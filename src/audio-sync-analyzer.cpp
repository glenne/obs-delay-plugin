/*
Audio Sync Analyzer - High Quality Correlation
Copyright (C) 2025 

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.
*/

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <complex>
#include <vector>

#include <media-io/audio-io.h>
#include <obs-module.h>
#include <util/bmem.h>
#include <util/platform.h>

#include "pocketfft_hdronly.h"

#define BUFFER_SECONDS 5u
#define MIN_WINDOW_MS 200u
#define MAX_WINDOW_MS 3000u
#define DEFAULT_WINDOW_MS 1000u
#define MAX_LAG_MS 1500u
#define MIN_CORR_THRESHOLD 0.6f
#define PRE_EMPHASIS_ALPHA 0.95f

struct delay_meter_data {
	obs_source_t *context;
	obs_source_t *target;
	char *target_name;

	pthread_mutex_t lock;

	float *ref_buffer;
	float *tgt_buffer;
	size_t capacity;
	size_t ref_pos;
	size_t tgt_pos;
	size_t ref_count;
	size_t tgt_count;
	uint64_t last_ref_ns;
	uint64_t last_tgt_ns;

	uint32_t sample_rate;
	enum audio_format audio_format;
	uint32_t window_ms;
	uint32_t max_lag_ms;
	bool debug_enabled;

	char *last_delay_text;
	char *last_time_text;
	char *last_notes;
	double last_delay_ms;
	float last_correlation;
	bool ui_update_pending;
	bool last_delay_valid;
};

static size_t next_power_of_2(size_t n)
{
	size_t p = 1;
	while (p < n)
		p <<= 1;
	return p;
}

static size_t ms_to_samples(uint32_t ms, uint32_t sample_rate)
{
	return (size_t)(((uint64_t)ms * sample_rate + 500) / 1000);
}

static void apply_result_ui(void *param);

static void ring_write(float *buffer, size_t capacity, size_t *pos, size_t *count, const float *src, size_t frames)
{
	for (size_t i = 0; i < frames; ++i) {
		buffer[*pos] = src[i];
		*pos = (*pos + 1) % capacity;
		if (*count < capacity)
			(*count)++;
	}
}

static bool has_recent_audio(uint64_t last_ns, uint64_t now_ns, uint64_t max_age_ns)
{
	if (last_ns == 0 || now_ns < last_ns)
		return false;
	return (now_ns - last_ns) <= max_age_ns;
}

static const float *as_float_channel(const uint8_t *const *planes, enum audio_format format)
{
	if (!planes)
		return nullptr;

	if (format == AUDIO_FORMAT_FLOAT || format == AUDIO_FORMAT_FLOAT_PLANAR)
		return (const float *)planes[0];

	return nullptr;
}

static void apply_pre_emphasis(float *data, size_t frames)
{
	if (frames < 2)
		return;

	float prev = data[0];
	for (size_t i = 1; i < frames; ++i) {
		float next = data[i];
		data[i] = next - PRE_EMPHASIS_ALPHA * prev;
		prev = next;
	}
}

static void apply_hann_window(float *data, size_t frames)
{
	if (frames <= 1)
		return;

	for (size_t i = 0; i < frames; ++i) {
		double w = 0.5 * (1.0 - cos(2.0 * M_PI * (double)i / (frames - 1.0)));
		data[i] *= (float)w;
	}
}

static bool copy_recent(struct delay_meter_data *dm, float **out_ref, float **out_tgt, size_t *out_frames)
{
	float *ref = nullptr;
	float *tgt = nullptr;

	pthread_mutex_lock(&dm->lock);

	size_t available = dm->ref_count < dm->tgt_count ? dm->ref_count : dm->tgt_count;
	const size_t window_frames = ms_to_samples(dm->window_ms, dm->sample_rate);
	const size_t frames = available < window_frames ? available : window_frames;

	if (frames < 1024) {
		pthread_mutex_unlock(&dm->lock);
		return false;
	}

	ref = static_cast<float *>(bmalloc(frames * sizeof(float)));
	tgt = static_cast<float *>(bmalloc(frames * sizeof(float)));
	if (!ref || !tgt) {
		pthread_mutex_unlock(&dm->lock);
		bfree(ref);
		bfree(tgt);
		return false;
	}

	const size_t ref_start = (dm->ref_pos + dm->capacity - frames) % dm->capacity;
	const size_t tgt_start = (dm->tgt_pos + dm->capacity - frames) % dm->capacity;

	for (size_t i = 0; i < frames; ++i) {
		ref[i] = dm->ref_buffer[(ref_start + i) % dm->capacity];
		tgt[i] = dm->tgt_buffer[(tgt_start + i) % dm->capacity];
	}

	apply_pre_emphasis(ref, frames);
	apply_pre_emphasis(tgt, frames);
	apply_hann_window(ref, frames);
	apply_hann_window(tgt, frames);

	*out_ref = ref;
	*out_tgt = tgt;
	*out_frames = frames;
	pthread_mutex_unlock(&dm->lock);

	return true;
}

static bool estimate_delay(struct delay_meter_data *dm, double *delay_ms_out, double *corr_out)
{
	float *ref = nullptr;
	float *tgt = nullptr;
	size_t frames = 0;

	if (!copy_recent(dm, &ref, &tgt, &frames)) {
		blog(LOG_INFO, "[ADM]  copy_recent failed");
		return false;
	}

	if (dm->debug_enabled) {
		blog(LOG_INFO, "[ADM DEBUG] frames=%zu max_lag=%d", frames,
		     (int)ms_to_samples(dm->max_lag_ms, dm->sample_rate));
	}

	int max_lag = (int)ms_to_samples(dm->max_lag_ms, dm->sample_rate);
	max_lag = std::min(max_lag, (int)frames - 1);

	auto compute_mean = [](const float *data, size_t count) {
		if (!data || count == 0)
			return 0.0f;
		double sum = 0.0;
		for (size_t i = 0; i < count; ++i)
			sum += data[i];
		return (float)(sum / (double)count);
	};

	const float ref_mean = compute_mean(ref, frames);
	const float tgt_mean = compute_mean(tgt, frames);
	for (size_t i = 0; i < frames; ++i) {
		ref[i] -= ref_mean;
		tgt[i] -= tgt_mean;
	}

	std::vector<double> ref_prefix(frames + 1, 0.0);
	std::vector<double> tgt_prefix(frames + 1, 0.0);
	for (size_t i = 0; i < frames; ++i) {
		ref_prefix[i + 1] = ref_prefix[i] + (double)ref[i] * (double)ref[i];
		tgt_prefix[i + 1] = tgt_prefix[i] + (double)tgt[i] * (double)tgt[i];
	}

	const size_t nfft = next_power_of_2(frames * 2);
	const size_t spectrum_len = nfft / 2 + 1;

	std::vector<float> ref_padded(nfft, 0.0f);
	std::vector<float> tgt_padded(nfft, 0.0f);
	std::copy(ref, ref + frames, ref_padded.begin());
	std::copy(tgt, tgt + frames, tgt_padded.begin());

	std::vector<std::complex<float>> ref_fft(spectrum_len);
	std::vector<std::complex<float>> tgt_fft(spectrum_len);
	std::vector<std::complex<float>> cross_spectrum(spectrum_len);
	std::vector<float> corr_time(nfft, 0.0f);

	const pocketfft::shape_t shape{nfft};
	const pocketfft::stride_t stride_real{(ptrdiff_t)sizeof(float)};
	const pocketfft::stride_t stride_complex{(ptrdiff_t)sizeof(std::complex<float>)};

	pocketfft::r2c(shape, stride_real, stride_complex, 0, true, ref_padded.data(), ref_fft.data(), 1.0f);
	pocketfft::r2c(shape, stride_real, stride_complex, 0, true, tgt_padded.data(), tgt_fft.data(), 1.0f);

	for (size_t i = 0; i < spectrum_len; ++i)
		cross_spectrum[i] = ref_fft[i] * std::conj(tgt_fft[i]);

	const float ifft_scale = 1.0f / (float)nfft;
	pocketfft::c2r(shape, stride_complex, stride_real, 0, true, cross_spectrum.data(), corr_time.data(),
		       ifft_scale);

	double best_corr = -1.0;
	int best_lag = 0;
	int lag_count = 0;

	for (int lag = -max_lag; lag <= max_lag; ++lag) {
		const int abs_lag = std::abs(lag);
		const size_t overlap = frames - (size_t)abs_lag;
		if (overlap < 1024)
			continue;

		double energy_ref = 0.0;
		double energy_tgt = 0.0;

		if (lag >= 0) {
			energy_ref = ref_prefix[overlap] - ref_prefix[0];
			energy_tgt = tgt_prefix[(size_t)lag + overlap] - tgt_prefix[(size_t)lag];
		} else {
			const size_t start = (size_t)abs_lag;
			energy_ref = ref_prefix[start + overlap] - ref_prefix[start];
			energy_tgt = tgt_prefix[overlap] - tgt_prefix[0];
		}

		double denom = sqrt(energy_ref * energy_tgt);
		if (denom < 1e-8)
			continue;

		const size_t idx = lag >= 0 ? (size_t)lag : nfft - (size_t)abs_lag;
		double corr = corr_time[idx] / denom;
		lag_count++;

		if (corr > best_corr) {
			best_corr = corr;
			best_lag = lag;
		}
	}

	bfree(ref);
	bfree(tgt);

	if (dm->debug_enabled) {
		blog(LOG_INFO, "[ADM DEBUG] FINAL: best_corr=%.4f best_lag=%d lag_count=%d", best_corr, best_lag,
		     lag_count);
	}

	if (best_corr < MIN_CORR_THRESHOLD) {
		blog(LOG_INFO, "[ADM]  CORRELATION TOO LOW: %.4f < %.1f", best_corr, MIN_CORR_THRESHOLD);
		return false;
	}

	*delay_ms_out = ((double)best_lag * 1000.0) / (double)dm->sample_rate;
	*corr_out = best_corr;
	return true;
}

static void set_result(struct delay_meter_data *dm, const char *delay_text, const char *notes_text)
{
	if (!dm || !delay_text)
		return;

	blog(LOG_INFO, "[ADM DEBUG] Set Result=%s", delay_text);

	time_t now = time(nullptr);
	struct tm *tm_info = localtime(&now);
	char timestamp[10];
	strftime(timestamp, sizeof(timestamp), "%H:%M:%S", tm_info);

	bool queue = false;
	pthread_mutex_lock(&dm->lock);
	bfree(dm->last_delay_text);
	bfree(dm->last_time_text);
	bfree(dm->last_notes);
	dm->last_delay_text = bstrdup(delay_text);
	dm->last_time_text = bstrdup(timestamp);
	dm->last_notes = notes_text ? bstrdup(notes_text) : bstrdup("");
	dm->last_delay_valid = true;
	if (!dm->ui_update_pending) {
		dm->ui_update_pending = true;
		queue = true;
	}
	pthread_mutex_unlock(&dm->lock);

	if (queue)
		obs_queue_task(OBS_TASK_UI, apply_result_ui, dm, false);
}

static void apply_result_ui(void *param)
{
	auto *dm = static_cast<delay_meter_data *>(param);
	if (!dm || !dm->context)
		return;

	blog(LOG_INFO, "[ADM DEBUG] Apply Result");
	char *result = nullptr;
	char *time_txt = nullptr;
	char *notes_txt = nullptr;
	pthread_mutex_lock(&dm->lock);
	dm->ui_update_pending = false;
	if (dm->last_delay_text)
		result = bstrdup(dm->last_delay_text);
	if (dm->last_time_text)
		time_txt = bstrdup(dm->last_time_text);
	if (dm->last_notes)
		notes_txt = bstrdup(dm->last_notes);
	pthread_mutex_unlock(&dm->lock);

	if (!result)
		return;

	obs_data_t *settings = obs_source_get_settings(dm->context);
	if (settings) {
		obs_data_set_string(settings, "time_result", time_txt ? time_txt : "");
		obs_data_set_string(settings, "delay_result", result);
		obs_data_set_string(settings, "notes", notes_txt ? notes_txt : "");
		obs_source_update(dm->context, settings);
		obs_data_release(settings);
	}

	bfree(result);
	bfree(time_txt);
	bfree(notes_txt);
}

static void capture_target(void *param, obs_source_t *source, const struct audio_data *audio, bool muted)
{
	UNUSED_PARAMETER(source);
	UNUSED_PARAMETER(muted);

	auto *dm = static_cast<delay_meter_data *>(param);
	if (!dm)
		return;

	const float *samples = as_float_channel((const uint8_t *const *)audio->data, dm->audio_format);
	if (!samples || audio->frames == 0)
		return;

	pthread_mutex_lock(&dm->lock);
	ring_write(dm->tgt_buffer, dm->capacity, &dm->tgt_pos, &dm->tgt_count, samples, audio->frames);
	dm->last_tgt_ns = os_gettime_ns();
	// blog(LOG_INFO, "[DEBUG] Target buffer: %zu frames", dm->tgt_count);
	// if (audio->frames > 0) {
	//     blog(LOG_DEBUG, "[ADM] Target: %u → %zu", audio->frames, dm->tgt_count);
	// }
	pthread_mutex_unlock(&dm->lock);
}

static void connect_target(struct delay_meter_data *dm, const char *name, bool log_missing)
{
	if (dm->target && dm->target_name && name && strcmp(dm->target_name, name) == 0)
		return;

	if (dm->target) {
		blog(LOG_INFO, "Releasing prior audio callback");
		obs_source_remove_audio_capture_callback(dm->target, capture_target, dm);
		obs_source_release(dm->target);
		dm->target = nullptr;
		dm->tgt_count = 0;
		dm->tgt_pos = 0;
		dm->last_tgt_ns = 0;
	}

	blog(LOG_INFO, "[ADM Info] Connecting to %s", name);

	if (dm->target_name) {
		bfree(dm->target_name);
	}
	dm->target_name = name && *name ? bstrdup(name) : nullptr;

	if (!dm->target_name)
		return;

	obs_source_t *src = obs_get_source_by_name(dm->target_name);
	if (!src) {
		if (log_missing) {
			blog(LOG_INFO, "[ADM] Target '%s' not yet available", dm->target_name);
		}
		return;
	}

	dm->target = src;
	obs_source_add_audio_capture_callback(dm->target, capture_target, dm);
}

static void delay_meter_update(void *data, obs_data_t *settings)
{
	auto *dm = static_cast<delay_meter_data *>(data);
	if (!dm)
		return;

	blog(LOG_INFO, "[ADM TRACE] Meter Update");
	dm->window_ms = (uint32_t)obs_data_get_int(settings, "window_ms");
	dm->max_lag_ms = (uint32_t)obs_data_get_int(settings, "max_lag_ms");
	dm->debug_enabled = obs_data_get_bool(settings, "debug_enabled");

	const char *target = obs_data_get_string(settings, "target_source");
	connect_target(dm, target, true);
	dm->last_ref_ns = 0;
	dm->last_tgt_ns = 0;
}

static bool perform_measure(struct delay_meter_data *dm)
{
	if (dm->debug_enabled) {
		blog(LOG_INFO, "[ADM DIAG] Starting measurement");
	}

	if (!dm || !dm->target) {
		set_result(dm, "No target source", "Select a delayed source to compare against.");
		dm->last_delay_valid = false;
		return false;
	}

	uint64_t last_ref_ns = 0;
	uint64_t last_tgt_ns = 0;
	pthread_mutex_lock(&dm->lock);
	if (dm->debug_enabled) {
		blog(LOG_INFO, "[ADM DIAG] ref=%zu tgt=%zu", dm->ref_count, dm->tgt_count);
	}
	bool enough = dm->ref_count >= 1024 && dm->tgt_count >= 1024;
	last_ref_ns = dm->last_ref_ns;
	last_tgt_ns = dm->last_tgt_ns;
	pthread_mutex_unlock(&dm->lock);

	if (!enough) {
		set_result(dm, "Buffers too small",
			   "Need more buffered audio from both reference and target before measuring.");
		dm->last_delay_valid = false;
		return false;
	}

	const uint64_t now_ns = os_gettime_ns();
	const uint64_t max_age_ns = (uint64_t)(dm->window_ms + dm->max_lag_ms + 200u) * 1000000ULL; // grace window

	if (!has_recent_audio(last_ref_ns, now_ns, max_age_ns)) {
		set_result(dm, "Reference inactive", "No recent audio on reference source.");
		dm->last_delay_valid = false;
		return false;
	}

	if (!has_recent_audio(last_tgt_ns, now_ns, max_age_ns)) {
		set_result(dm, "Target inactive", "No recent audio on target source.");
		dm->last_delay_valid = false;
		return false;
	}

	double delay_ms = 0.0;
	double corr = 0.0;

	blog(LOG_INFO, "[ADM] Estimating Audio Delay");
	if (estimate_delay(dm, &delay_ms, &corr)) {
		char *target_name_copy = nullptr;

		pthread_mutex_lock(&dm->lock);
		target_name_copy = dm->target_name ? bstrdup(dm->target_name) : nullptr;
		pthread_mutex_unlock(&dm->lock);

		char buffer[256];
		snprintf(buffer, sizeof(buffer), "%+6.1f ms (correlation: %.2f)", delay_ms, corr);
		char notes_buf[256];
		if (delay_ms > 0) {
			snprintf(notes_buf, sizeof(notes_buf), "Target '%s' lags reference by %.1f ms",
				 target_name_copy ? target_name_copy : "<target>", delay_ms);
		} else if (delay_ms < 0) {
			snprintf(notes_buf, sizeof(notes_buf), "Target '%s' leads reference by %.1f ms",
				 target_name_copy ? target_name_copy : "<target>", fabs(delay_ms));
		} else {
			snprintf(notes_buf, sizeof(notes_buf), "Target '%s' is aligned with reference",
				 target_name_copy ? target_name_copy : "<target>");
		}

		set_result(dm, buffer, notes_buf);
		dm->last_delay_ms = delay_ms;
		dm->last_delay_valid = true;

		bfree(target_name_copy);
		return true;
	}

	set_result(dm, "Insufficient correlation - check audio levels and similarity",
		   "Insufficient correlation; ensure both sources carry similar program audio.");
	dm->last_delay_valid = false;
	return false;
}

// ✅ FIXED: Correct signature - only 3 parameters (void* data)
static bool measure_now_clicked(obs_properties_t *props, obs_property_t *property, void *data)
{
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	auto *dm = static_cast<delay_meter_data *>(data);
	/* Always return true so OBS refreshes the properties view even when measurement
	 * fails (e.g., missing target or not enough buffered audio). */
	perform_measure(dm);
	return true;
}

static bool apply_sync_offset_clicked(obs_properties_t *props, obs_property_t *property, void *data)
{
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	auto *dm = static_cast<delay_meter_data *>(data);
	if (!dm)
		return true;

	pthread_mutex_lock(&dm->lock);
	bool valid = dm->last_delay_valid;
	double delay_ms = dm->last_delay_ms;
	pthread_mutex_unlock(&dm->lock);

	if (!valid) {
		set_result(dm, " No recent measurement", "Run Measure Now before applying offset.");
		return true;
	}

	obs_source_t *parent = obs_filter_get_parent(dm->context);
	if (!parent) {
		set_result(dm, " No parent source", "Cannot apply offset without a parent source.");
		return true;
	}

	parent = obs_source_get_ref(parent);
	if (!parent) {
		set_result(dm, " Parent unavailable", "Parent source vanished before applying offset.");
		return true;
	}

	int64_t offset_ns = (int64_t)llround(delay_ms * 1000000.0);
	obs_source_set_sync_offset(parent, offset_ns);
	obs_source_release(parent);

	char msg[128];
	snprintf(msg, sizeof(msg), "Applied %+0.1f ms to Sync Offset", delay_ms);
	set_result(dm, msg, "Sync Offset updated on reference source.");
	return true;
}

static struct obs_audio_data *delay_meter_filter_audio(void *data, struct obs_audio_data *audio)
{
	auto *dm = static_cast<delay_meter_data *>(data);
	if (!dm)
		return audio;

	if (!dm->target && dm->target_name)
		connect_target(dm, dm->target_name, false);

	if (!dm->target) {
		return audio;
	}

	const float *samples = as_float_channel((const uint8_t *const *)audio->data, dm->audio_format);
	if (samples && audio->frames > 0) {
		pthread_mutex_lock(&dm->lock);
		ring_write(dm->ref_buffer, dm->capacity, &dm->ref_pos, &dm->ref_count, samples, audio->frames);
		dm->last_ref_ns = os_gettime_ns();
		pthread_mutex_unlock(&dm->lock);
	}
	return audio;
}

static void delay_meter_destroy(void *data)
{
	blog(LOG_INFO, "[ADM Trace] Destroy");
	auto *dm = static_cast<delay_meter_data *>(data);
	if (!dm)
		return;

	if (dm->target) {
		obs_source_remove_audio_capture_callback(dm->target, capture_target, dm);
		obs_source_release(dm->target);
	}

	pthread_mutex_destroy(&dm->lock);
	bfree(dm->ref_buffer);
	bfree(dm->tgt_buffer);
	bfree(dm->target_name);
	bfree(dm->last_delay_text);
	bfree(dm->last_time_text);
	bfree(dm->last_notes);
	bfree(dm);
}

static void *delay_meter_create(obs_data_t *settings, obs_source_t *context)
{
	struct delay_meter_data *dm = static_cast<delay_meter_data *>(bzalloc(sizeof(*dm)));
	dm->context = context;
	pthread_mutex_init(&dm->lock, nullptr);

	dm->sample_rate = audio_output_get_sample_rate(obs_get_audio());
	dm->audio_format = AUDIO_FORMAT_FLOAT_PLANAR;

	dm->capacity = ms_to_samples(BUFFER_SECONDS * 1000u, dm->sample_rate);
	dm->ref_buffer = static_cast<float *>(bzalloc(dm->capacity * sizeof(float)));
	dm->tgt_buffer = static_cast<float *>(bzalloc(dm->capacity * sizeof(float)));
	dm->tgt_pos = dm->ref_pos = 0;
	dm->tgt_count = dm->ref_count = 0;

	dm->last_delay_valid = false;
	dm->debug_enabled = obs_data_get_bool(settings, "debug_enabled");
	dm->window_ms = (uint32_t)obs_data_get_int(settings, "window_ms");
	dm->max_lag_ms = (uint32_t)obs_data_get_int(settings, "max_lag_ms");

	obs_data_set_string(settings, "time_result", "");
	obs_data_set_string(settings, "delay_result", "Ready...");
	obs_data_set_string(settings, "notes", "");

	const char *target = obs_data_get_string(settings, "target_source");
	connect_target(dm, target, true);

	blog(LOG_INFO, "[ADM TRACE] Created");
	return dm;
}

static bool add_audio_sources(void *priv, obs_source_t *src)
{
	auto *list = static_cast<obs_property_t *>(priv);
	uint32_t flags = obs_source_get_output_flags(src);
	if (!(flags & OBS_SOURCE_AUDIO))
		return true;

	const char *name = obs_source_get_name(src);
	obs_property_list_add_string(list, name, name);
	return true;
}

static obs_properties_t *delay_meter_properties(void *data)
{
	UNUSED_PARAMETER(data);
	blog(LOG_INFO, "[ADM TRACE] Creating Properties");
	obs_properties_t *props = obs_properties_create();

	obs_property_t *list = obs_properties_add_list(props, "target_source", "Delayed Source", OBS_COMBO_TYPE_LIST,
						       OBS_COMBO_FORMAT_STRING);
	obs_enum_sources(add_audio_sources, list);

	obs_properties_add_int_slider(props, "window_ms", "Analysis Window (ms)", MIN_WINDOW_MS, MAX_WINDOW_MS, 50);
	obs_properties_add_int_slider(props, "max_lag_ms", "Max Lag Search (ms)", 50, MAX_LAG_MS, 25);

	obs_property_t *time_prop = obs_properties_add_text(props, "time_result", "Time", OBS_TEXT_INFO);
	obs_property_set_enabled(time_prop, false);

	obs_property_t *result = obs_properties_add_text(props, "delay_result", "Delay", OBS_TEXT_INFO);
	obs_property_set_enabled(result, false);

	obs_property_t *notes = obs_properties_add_text(props, "notes", "Notes", OBS_TEXT_INFO);
	obs_property_set_enabled(notes, false);

	obs_properties_add_button2(props, "measure_now", "Measure Now", measure_now_clicked, data);
	obs_properties_add_button2(props, "apply_sync_offset", "Apply to Sync Offset", apply_sync_offset_clicked, data);

	obs_properties_add_bool(props, "debug_enabled", "Enable Debug Logging");

	return props;
}

static void delay_meter_defaults(obs_data_t *settings)
{
	blog(LOG_INFO, "[ADM TRACE] Setting Defaults");
	obs_data_set_default_int(settings, "window_ms", DEFAULT_WINDOW_MS);
	obs_data_set_default_int(settings, "max_lag_ms", 500);
	obs_data_set_default_bool(settings, "debug_enabled", false);
	obs_data_set_default_string(settings, "delay_result", "Ready...");
	obs_data_set_default_string(settings, "time_result", "");
	obs_data_set_default_string(settings, "notes", "");
	obs_data_set_default_string(settings, "target_source", "");
}

static const char *delay_meter_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "Audio Sync Analyzer";
}

extern "C" {
static obs_source_info make_delay_meter_filter_info()
{
	obs_source_info info = {};
	info.id = "audio-sync-analyzer";
	info.type = OBS_SOURCE_TYPE_FILTER;
	info.output_flags = OBS_SOURCE_AUDIO;
	info.get_name = delay_meter_get_name;
	info.create = delay_meter_create;
	info.destroy = delay_meter_destroy;
	info.update = delay_meter_update;
	info.get_defaults = delay_meter_defaults;
	info.get_properties = delay_meter_properties;
	info.filter_audio = delay_meter_filter_audio;
	return info;
}

struct obs_source_info delay_meter_filter = make_delay_meter_filter_info();
}
