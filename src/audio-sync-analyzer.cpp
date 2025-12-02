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
#include <string>
#include <vector>

#include <obs-frontend-api.h>
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
#define MIN_CORR_THRESHOLD 0.4f
#define BANDPASS_LOW_Hz 200.0f
#define BANDPASS_HIGH_Hz 2000.0f

struct bandpass_coeffs {
	float b0, b1, b2, a1, a2;
};

struct audio_sync_data;
static audio_sync_data *g_dm = nullptr;

static void connect_ref(struct audio_sync_data *dm);
static void connect_target(struct audio_sync_data *dm);

struct audio_sync_data {
	obs_source_t *ref;
	obs_source_t *target;
	std::string ref_name;
	std::string target_name;
	std::string connected_ref;
	std::string connected_target;

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
	float corr_threshold;
	bool debug_enabled;

	// Bandpass filter coefficients (state is reset for each measurement)
	struct bandpass_coeffs bp_coeffs;

	std::string last_delay_text = "---";
	std::string last_time_text;
	std::string last_notes;
	double last_delay_ms;
	float last_correlation;
	bool last_delay_valid;
	bool average_in_progress;
	bool average_stop;
	bool average_thread_active;
	pthread_t average_thread;
};

static void update_dock_ui(audio_sync_data *dm);

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

static void design_bandpass_filter(float low_freq, float high_freq, uint32_t sample_rate,
				   struct bandpass_coeffs *coeffs)
{
	// Design a second-order bandpass filter using biquad structure
	// This implements a bandpass filter with center frequency and Q

	const float center_freq = sqrtf(low_freq * high_freq);
	const float bandwidth = high_freq - low_freq;
	const float Q = center_freq / bandwidth;

	const float w0 = 2.0f * (float)M_PI * center_freq / (float)sample_rate;
	const float cos_w0 = cosf(w0);
	const float sin_w0 = sinf(w0);
	const float alpha = sin_w0 / (2.0f * Q);

	const float b0_val = alpha;
	const float b1_val = 0.0f;
	const float b2_val = -alpha;
	const float a0_val = 1.0f + alpha;
	const float a1_val = -2.0f * cos_w0;
	const float a2_val = 1.0f - alpha;

	// Normalize coefficients
	coeffs->b0 = b0_val / a0_val;
	coeffs->b1 = b1_val / a0_val;
	coeffs->b2 = b2_val / a0_val;
	coeffs->a1 = a1_val / a0_val;
	coeffs->a2 = a2_val / a0_val;
}

static void apply_bandpass_filter(float *data, size_t samples, const struct bandpass_coeffs *coeffs)
{
	if (samples == 0)
		return;

	// Initialize filter state to zero for independent measurements
	float x_prev1 = 0.0f;
	float x_prev2 = 0.0f;
	float y_prev1 = 0.0f;
	float y_prev2 = 0.0f;

	for (size_t i = 0; i < samples; ++i) {
		const float x = data[i];

		// Direct Form II transposed biquad filter
		const float y = coeffs->b0 * x + coeffs->b1 * x_prev1 + coeffs->b2 * x_prev2 - coeffs->a1 * y_prev1 -
				coeffs->a2 * y_prev2;

		x_prev2 = x_prev1;
		x_prev1 = x;
		y_prev2 = y_prev1;
		y_prev1 = y;

		data[i] = y;
	}
}

static void apply_hann_window(float *data, size_t samples)
{
	if (samples <= 1)
		return;

	for (size_t i = 0; i < samples; ++i) {
		double w = 0.5 * (1.0 - cos(2.0 * M_PI * (double)i / (samples - 1.0)));
		data[i] *= (float)w;
	}
}

static void frontend_save_cb(obs_data_t *settings, bool saving, void *private_data)
{
	auto *dm = static_cast<audio_sync_data *>(private_data);
	if (!dm || !settings)
		return;

	const char *key = "audio_sync_analyzer";

	if (saving) {
		obs_data_t *obj = obs_data_create();
		pthread_mutex_lock(&dm->lock);
		obs_data_set_string(obj, "ref_name", dm->ref_name.c_str());
		obs_data_set_string(obj, "target_name", dm->target_name.c_str());
		obs_data_set_int(obj, "window_ms", dm->window_ms);
		obs_data_set_int(obj, "max_lag_ms", dm->max_lag_ms);
		obs_data_set_double(obj, "corr_threshold", dm->corr_threshold);
		obs_data_set_bool(obj, "debug_enabled", dm->debug_enabled);
		pthread_mutex_unlock(&dm->lock);

		obs_data_set_obj(settings, key, obj);
		obs_data_release(obj);
		blog(LOG_INFO, "[ASM] saved frontend settings");
	} else {
		obs_data_t *obj = obs_data_get_obj(settings, key);
		if (!obj)
			return;

		pthread_mutex_lock(&dm->lock);
		dm->ref_name = obs_data_get_string(obj, "ref_name");
		dm->target_name = obs_data_get_string(obj, "target_name");
		uint32_t win = (uint32_t)obs_data_get_int(obj, "window_ms");
		uint32_t lag = (uint32_t)obs_data_get_int(obj, "max_lag_ms");
		double corr = obs_data_get_double(obj, "corr_threshold");
		dm->window_ms = win ? win : DEFAULT_WINDOW_MS;
		dm->max_lag_ms = lag ? lag : 500;
		dm->corr_threshold = (float)(corr > 0.0 ? corr : MIN_CORR_THRESHOLD);
		dm->debug_enabled = obs_data_get_bool(obj, "debug_enabled");
		pthread_mutex_unlock(&dm->lock);

		obs_data_release(obj);
		connect_ref(dm);
		connect_target(dm);
		update_dock_ui(dm);
		blog(LOG_INFO, "[ASM] loaded frontend settings");
	}
}

static bool copy_recent(struct audio_sync_data *dm, float **out_ref, float **out_tgt, size_t *out_frames)
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

	// Apply bandpass filter instead of pre-emphasis
	apply_bandpass_filter(ref, frames, &dm->bp_coeffs);
	apply_bandpass_filter(tgt, frames, &dm->bp_coeffs);

	apply_hann_window(ref, frames);
	apply_hann_window(tgt, frames);

	*out_ref = ref;
	*out_tgt = tgt;
	*out_frames = frames;
	pthread_mutex_unlock(&dm->lock);

	return true;
}

static bool estimate_delay(struct audio_sync_data *dm, double *delay_ms_out, double *corr_out)
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

	if (best_corr < dm->corr_threshold) {
		blog(LOG_INFO, "[ADM]  CORRELATION TOO LOW: %.4f < %.2f", best_corr, dm->corr_threshold);
		return false;
	}

	*delay_ms_out = ((double)best_lag * 1000.0) / (double)dm->sample_rate;
	*corr_out = best_corr;
	return true;
}

static void set_result(struct audio_sync_data *dm, const char *delay_text, const char *notes_text, bool valid)
{
	if (!dm || !delay_text)
		return;

	time_t now = time(nullptr);
	struct tm *tm_info = localtime(&now);
	char timestamp[10];
	strftime(timestamp, sizeof(timestamp), "%H:%M:%S", tm_info);

	pthread_mutex_lock(&dm->lock);
	dm->last_delay_text = delay_text;
	dm->last_time_text = timestamp;
	dm->last_notes = notes_text ? notes_text : "";
	dm->last_delay_valid = valid;
	pthread_mutex_unlock(&dm->lock);

	update_dock_ui(dm);
}

static void capture_target(void *param, obs_source_t *source, const struct audio_data *audio, bool muted)
{
	UNUSED_PARAMETER(source);
	UNUSED_PARAMETER(muted);

	auto *dm = static_cast<audio_sync_data *>(param);
	if (!dm)
		return;

	const float *samples = as_float_channel((const uint8_t *const *)audio->data, dm->audio_format);
	if (!samples || audio->frames == 0)
		return;

	pthread_mutex_lock(&dm->lock);
	ring_write(dm->tgt_buffer, dm->capacity, &dm->tgt_pos, &dm->tgt_count, samples, audio->frames);
	dm->last_tgt_ns = os_gettime_ns();
	pthread_mutex_unlock(&dm->lock);
}

static void capture_ref(void *param, obs_source_t *source, const struct audio_data *audio, bool muted)
{
	UNUSED_PARAMETER(source);
	UNUSED_PARAMETER(muted);

	auto *dm = static_cast<audio_sync_data *>(param);
	if (!dm)
		return;

	const float *samples = as_float_channel((const uint8_t *const *)audio->data, dm->audio_format);
	if (!samples || audio->frames == 0)
		return;

	pthread_mutex_lock(&dm->lock);
	ring_write(dm->ref_buffer, dm->capacity, &dm->ref_pos, &dm->ref_count, samples, audio->frames);
	dm->last_ref_ns = os_gettime_ns();
	pthread_mutex_unlock(&dm->lock);
}

static void connect_ref(struct audio_sync_data *dm)
{
	if (dm->ref && dm->ref_name == dm->connected_ref) {
		return;
	}
	if (dm->ref_name.empty())
		return;
	if (dm->debug_enabled) {
		blog(LOG_INFO, "[ADM TRACE] Connect Ref");
	}

	if (dm->ref) {
		obs_source_remove_audio_capture_callback(dm->ref, capture_ref, dm);
		obs_source_release(dm->ref);
		dm->ref = nullptr;
		dm->ref_count = 0;
		dm->ref_pos = 0;
		dm->last_ref_ns = 0;
	}

	obs_source_t *src = obs_get_source_by_name(dm->ref_name.c_str());
	if (!src) {
		blog(LOG_INFO, "[ADM] Ref '%s' not yet available", dm->ref_name.c_str());
		return;
	}

	dm->ref = src;
	dm->connected_ref = dm->ref_name;
	obs_source_add_audio_capture_callback(dm->ref, capture_ref, dm);
}

static void connect_target(struct audio_sync_data *dm)
{
	if (dm->target && dm->connected_target == dm->target_name) {
		// nothing has changed, ignore
		return;
	}
	if (dm->target_name.empty())
		return;
	if (dm->debug_enabled) {
		blog(LOG_INFO, "[ADM TRACE] Connect Target");
	}

	if (dm->target) {
		blog(LOG_INFO, "Releasing prior audio callback");
		obs_source_remove_audio_capture_callback(dm->target, capture_target, dm);
		obs_source_release(dm->target);
		dm->target = nullptr;
		dm->connected_target = "";
		dm->tgt_count = 0;
		dm->tgt_pos = 0;
		dm->last_tgt_ns = 0;
	}

	blog(LOG_INFO, "[ADM Info] Connecting to %s", dm->target_name.c_str());

	obs_source_t *src = obs_get_source_by_name(dm->target_name.c_str());
	if (!src) {
		blog(LOG_INFO, "[ADM] Target '%s' not yet available", dm->target_name.c_str());
		// OK to leave target_name set
		return;
	}

	dm->target = src;
	dm->connected_target = dm->target_name;
	obs_source_add_audio_capture_callback(dm->target, capture_target, dm);
	blog(LOG_INFO, "[ADM Info] Connected to %s", dm->target_name.c_str());
}

struct measurement_sample {
	double delay_ms = 0.0;
	double correlation = 0.0;
	bool success = false;
	std::string status;
};

static bool try_measure_once(struct audio_sync_data *dm, measurement_sample &out)
{
	if (!dm || !dm->target) {
		out.status = "No target source";
		return false;
	}

	uint64_t last_ref_ns = 0;
	uint64_t last_tgt_ns = 0;
	size_t ref_count = 0;
	size_t tgt_count = 0;

	pthread_mutex_lock(&dm->lock);
	ref_count = dm->ref_count;
	tgt_count = dm->tgt_count;
	last_ref_ns = dm->last_ref_ns;
	last_tgt_ns = dm->last_tgt_ns;
	pthread_mutex_unlock(&dm->lock);

	if (ref_count < 1024 || tgt_count < 1024) {
		out.status = "Buffers too small";
		return false;
	}

	const uint64_t now_ns = os_gettime_ns();
	const uint64_t max_age_ns = (uint64_t)(dm->window_ms + dm->max_lag_ms + 200u) * 1000000ULL;

	if (!has_recent_audio(last_ref_ns, now_ns, max_age_ns)) {
		out.status = "Reference inactive";
		return false;
	}

	if (!has_recent_audio(last_tgt_ns, now_ns, max_age_ns)) {
		out.status = "Target inactive";
		return false;
	}

	double delay_ms = 0.0;
	double corr = 0.0;
	if (!estimate_delay(dm, &delay_ms, &corr)) {
		out.status = "Insufficient correlation";
		return false;
	}

	out.delay_ms = delay_ms;
	out.correlation = corr;
	out.success = true;
	out.status.clear();
	return true;
}

static bool perform_measure(struct audio_sync_data *dm)
{
	if (dm->debug_enabled) {
		blog(LOG_INFO, "[ADM DIAG] Starting measurement");
	}

	if (!dm->ref && !dm->ref_name.empty())
		connect_ref(dm);
	if (!dm->target && !dm->target_name.empty())
		connect_target(dm);

	if (!dm || !dm->target || !dm->ref) {
		set_result(dm, "---", "Select both reference and target sources.", false);
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

	const uint64_t now_ns = os_gettime_ns();
	const uint64_t max_age_ns = (uint64_t)(dm->window_ms + dm->max_lag_ms + 200u) * 1000000ULL; // grace window

	if (!has_recent_audio(last_ref_ns, now_ns, max_age_ns)) {
		set_result(dm, "---", "No recent audio on reference source.", false);
		return false;
	}

	if (!has_recent_audio(last_tgt_ns, now_ns, max_age_ns)) {
		set_result(dm, "---", "No recent audio on target source.", false);
		return false;
	}

	if (!enough) {
		set_result(dm, "---", "Need more buffered audio from both reference and target before measuring.",
			   false);
		return false;
	}

	double delay_ms = 0.0;
	double corr = 0.0;

	blog(LOG_INFO, "[ADM] Estimating Audio Delay");
	if (estimate_delay(dm, &delay_ms, &corr)) {
		std::string target_name_copy;

		pthread_mutex_lock(&dm->lock);
		target_name_copy = dm->target_name;
		dm->last_correlation = (float)corr;
		pthread_mutex_unlock(&dm->lock);

		char buffer[256];
		snprintf(buffer, sizeof(buffer), "%+6.1f ms", delay_ms);
		char notes_buf[256];
		if (delay_ms > 0) {
			snprintf(notes_buf, sizeof(notes_buf), "Target '%s' lags reference by %.1f ms (corr=%.2f)",
				 !target_name_copy.empty() ? target_name_copy.c_str() : "<target>", delay_ms, corr);
		} else if (delay_ms < 0) {
			snprintf(notes_buf, sizeof(notes_buf), "Target '%s' leads reference by %.1f ms (corr=%.2f)",
				 !target_name_copy.empty() ? target_name_copy.c_str() : "<target>", fabs(delay_ms),
				 corr);
		} else {
			snprintf(notes_buf, sizeof(notes_buf), "Target '%s' is aligned with reference (corr=%.2f)",
				 !target_name_copy.empty() ? target_name_copy.c_str() : "<target>", corr);
		}

		set_result(dm, buffer, notes_buf, true);
		dm->last_delay_ms = delay_ms;
		dm->last_delay_valid = true;
		return true;
	}

	set_result(dm, "---", "Insufficient correlation; ensure both sources carry similar program audio.", false);
	return false;
}

static void measure_now(audio_sync_data *dm)
{
	perform_measure(dm);
}

static void *measure_average_worker(void *param)
{
	auto *dm = static_cast<audio_sync_data *>(param);
	if (!dm)
		return nullptr;

	std::vector<measurement_sample> samples;
	samples.reserve(10);

	for (int i = 0; i < 10; ++i) {
		pthread_mutex_lock(&dm->lock);
		bool stop = dm->average_stop;
		pthread_mutex_unlock(&dm->lock);
		if (stop)
			break;

		measurement_sample sample;
		try_measure_once(dm, sample);
		samples.push_back(sample);

		if (i < 9)
			os_sleep_ms(400);
	}

	std::string notes;
	std::vector<measurement_sample> successes;
	successes.reserve(samples.size());

	for (size_t i = 0; i < samples.size(); ++i) {
		const auto &s = samples[i];
		if (s.success) {
			successes.push_back(s);
		}
		if (dm->debug_enabled) {
			char line[160];
			if (s.success) {
				snprintf(line, sizeof(line), "%2zu: %+6.1f ms (corr=%.2f)", i + 1, s.delay_ms,
					 s.correlation);
			} else {
				snprintf(line, sizeof(line), "%2zu: fail (%s)", i + 1, s.status.c_str());
			}
			if (!notes.empty())
				notes += "\n";
			notes += line;
		}
	}

	double avg_delay = 0.0;
	double avg_corr = 0.0;
	bool have_result = false;

	if (!successes.empty()) {
		std::sort(successes.begin(), successes.end(),
			  [](const measurement_sample &a, const measurement_sample &b) {
				  return a.correlation > b.correlation;
			  });
		const size_t take = std::min<size_t>(4, successes.size());
		double sum_delay = 0.0;
		double sum_corr = 0.0;
		for (size_t i = 0; i < take; ++i) {
			sum_delay += successes[i].delay_ms;
			sum_corr += successes[i].correlation;
		}
		avg_delay = sum_delay / (double)take;
		avg_corr = sum_corr / (double)take;
		have_result = true;
	}

	if (have_result) {
		char buffer[256];
		snprintf(buffer, sizeof(buffer), "%+6.1f ms", avg_delay);
		const char *details = dm->debug_enabled ? notes.c_str() : "Average completed (top 4 used).";
		set_result(dm, buffer, details, true);
		pthread_mutex_lock(&dm->lock);
		dm->last_delay_ms = avg_delay;
		dm->last_delay_valid = true;
		dm->last_correlation = (float)avg_corr;
		pthread_mutex_unlock(&dm->lock);
	} else {
		set_result(dm, "Average failed", notes.empty() ? "No successful measurements" : notes.c_str(), false);
		pthread_mutex_lock(&dm->lock);
		dm->last_delay_valid = false;
		pthread_mutex_unlock(&dm->lock);
	}

	pthread_mutex_lock(&dm->lock);
	dm->average_in_progress = false;
	dm->average_thread_active = false;
	pthread_mutex_unlock(&dm->lock);
	blog(LOG_INFO, "[ADM Trace] Measure Thread Complete");
	return nullptr;
}

static void measure_average(audio_sync_data *dm)
{
	if (!dm)
		return;

	pthread_mutex_lock(&dm->lock);
	if (dm->average_in_progress) {
		pthread_mutex_unlock(&dm->lock);
		return;
	}
	dm->average_in_progress = true;
	dm->average_stop = false;
	pthread_mutex_unlock(&dm->lock);

	if (pthread_create(&dm->average_thread, nullptr, measure_average_worker, dm) != 0) {
		pthread_mutex_lock(&dm->lock);
		dm->average_in_progress = false;
		pthread_mutex_unlock(&dm->lock);
		set_result(dm, "Error", "Could not start background measurement thread.", false);
		return;
	}

	pthread_mutex_lock(&dm->lock);
	dm->average_thread_active = true;
	pthread_mutex_unlock(&dm->lock);

	set_result(dm, "Averaging...", "Collecting 10 measurements over 1 second.", false);
	pthread_mutex_lock(&dm->lock);
	dm->last_delay_valid = false;
	pthread_mutex_unlock(&dm->lock);
}

static void apply_sync_offset(audio_sync_data *dm)
{
	if (!dm)
		return;

	pthread_mutex_lock(&dm->lock);
	bool valid = dm->last_delay_valid;
	double delay_ms = dm->last_delay_ms;
	pthread_mutex_unlock(&dm->lock);

	if (!valid) {
		set_result(dm, "No Data", "Run Measure Now before applying offset.", false);
		return;
	}

	obs_source_t *ref = nullptr;
	pthread_mutex_lock(&dm->lock);
	ref = dm->ref ? obs_source_get_ref(dm->ref) : nullptr;
	pthread_mutex_unlock(&dm->lock);

	if (!ref) {
		set_result(dm, "No Reference", "Select a reference source first.", false);
		return;
	}

	int64_t offset_ns = (int64_t)llround(delay_ms * 1000000.0);
	obs_source_set_sync_offset(ref, offset_ns);
	obs_source_release(ref);

	char msg[128];
	snprintf(msg, sizeof(msg), "✓ %+0.1f", delay_ms);
	set_result(dm, msg, "Sync Offset updated on reference source.", true);
}

// ──────────────────────────────────────────────────────────────
//  Dockable Live Analyzer
// ──────────────────────────────────────────────────────────────
#include <QDockWidget>
#include <QPointer>
#include <QLabel>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QTextEdit>
#include <QScrollBar>
#include <QVBoxLayout>
#include <QMainWindow>
#include <QWidget>
#include <QToolButton>
#include <QStyle>

static void populate_source_combo(QComboBox *combo, const std::string &current)
{
	combo->clear();
	obs_enum_sources(
		[](void *data, obs_source_t *src) {
			auto *c = static_cast<QComboBox *>(data);
			uint32_t flags = obs_source_get_output_flags(src);
			if (!(flags & OBS_SOURCE_AUDIO))
				return true;
			const char *name = obs_source_get_name(src);
			c->addItem(QString::fromUtf8(name));
			return true;
		},
		combo);
	int idx = combo->findText(QString::fromStdString(current));
	if (idx >= 0)
		combo->setCurrentIndex(idx);
}

class SyncDockWidget : public QWidget {
	Q_OBJECT
public:
	QLabel *delayLabel;
	QLabel *corrLabel;
	QLabel *sourceLabel;
	QTextEdit *logView;
	audio_sync_data *dm;
	QToolButton *btnSettings;
	QPushButton *btnApply;

	explicit SyncDockWidget(audio_sync_data *data) : QWidget(nullptr), dm(data)
	{
		setWindowTitle("Audio Sync Analyzer");
		auto *lay = new QVBoxLayout(this);

		sourceLabel = new QLabel("Reference ↔ Target");
		sourceLabel->setAlignment(Qt::AlignCenter);
		sourceLabel->setStyleSheet(
			"font-weight: 600; border: 1px solid #666; border-radius: 4px; padding: 4px; background: transparent;");
		delayLabel = new QLabel("---");
		delayLabel->setStyleSheet(
			"font-size: 14px; font-weight: bold; color: #2e9afe; border: 1px solid #666; border-radius: 4px; padding: 4px; ");
		delayLabel->setAlignment(Qt::AlignCenter);
		corrLabel = new QLabel("Corr: --");
		corrLabel->setAlignment(Qt::AlignCenter);
		corrLabel->setStyleSheet("border: 1px solid #666; border-radius: 4px; padding: 4px; ");

		auto *btnMeasure = new QPushButton("Measure");
		btnMeasure->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
		auto *btnAvg = new QPushButton("Avg");
		btnAvg->setMinimumWidth(56);

		btnSettings = new QToolButton();
		btnSettings->setText(QString::fromUtf8("\u2699")); // gear symbol
		btnSettings->setStyleSheet("font-size: 18px;");
		btnSettings->setToolButtonStyle(Qt::ToolButtonTextOnly);
		btnSettings->setAutoRaise(true);
		btnSettings->setFixedSize(28, 28);
		btnSettings->setToolTip("Settings");

		btnApply = new QPushButton("Apply");
		btnApply->setEnabled(false);

		logView = new QTextEdit;
		logView->setReadOnly(true);
		logView->setMaximumHeight(160);
		logView->setMaximumWidth(360);
		logView->setLineWrapMode(QTextEdit::WidgetWidth);

		// Top row: sources + settings on the right
		auto *topRow = new QHBoxLayout();
		topRow->addWidget(sourceLabel, 1);
		topRow->addWidget(btnSettings, 0, Qt::AlignRight);
		lay->addLayout(topRow);

		// Result row: delay/corr with Apply on the right
		auto *resultCol = new QVBoxLayout();
		resultCol->addWidget(delayLabel);
		resultCol->addWidget(corrLabel);
		auto *resultRow = new QHBoxLayout();
		resultRow->addLayout(resultCol, 1);
		resultRow->addWidget(btnApply);
		lay->addLayout(resultRow);

		auto *row = new QHBoxLayout();
		row->addWidget(btnMeasure, 1);
		row->addWidget(btnAvg);
		lay->addLayout(row);
		lay->addWidget(logView);
		lay->addStretch();

		connect(btnMeasure, &QPushButton::clicked, this, [this]() { measure_now(dm); });
		connect(btnAvg, &QPushButton::clicked, this, [this]() { measure_average(dm); });
		connect(btnApply, &QPushButton::clicked, this, [this]() { apply_sync_offset(dm); });
		connect(btnSettings, &QPushButton::clicked, this, [this]() { openSettingsDialog(); });

		// Initialize labels with current names
		updateSourceNames(QString::fromStdString(dm->ref_name), QString::fromStdString(dm->target_name));
	}

public slots:
	void updateResult(const QString &delay, const QString &notes, bool valid)
	{
		delayLabel->setText(delay);
		pthread_mutex_lock(&dm->lock);
		const double corr = dm ? dm->last_correlation : 0.0;
		const QString ref = QString::fromStdString(dm->ref_name.empty() ? "<ref>" : dm->ref_name);
		const QString tgt = QString::fromStdString(dm->target_name.empty() ? "<target>" : dm->target_name);
		pthread_mutex_unlock(&dm->lock);
		corrLabel->setText(QString("Corr: %1").arg(corr, 0, 'f', 2));
		updateSourceNames(ref, tgt);
		logView->setPlainText(notes);
		logView->verticalScrollBar()->setValue(logView->verticalScrollBar()->maximum());
		btnApply->setEnabled(valid);
	}

	void updateSourceNames(const QString &ref, const QString &tgt)
	{
		const QString refSafe = ref.isEmpty() ? "<ref>" : ref;
		const QString tgtSafe = tgt.isEmpty() ? "<target>" : tgt;
		sourceLabel->setText(QString("%1  ↔  %2").arg(refSafe, tgtSafe));
	}

private:
	void openSettingsDialog()
	{
		if (!dm)
			return;

		QDialog dlg(this);
		dlg.setWindowTitle("Audio Sync Analyzer Settings");
		auto *layout = new QFormLayout(&dlg);

		auto *refCombo = new QComboBox(&dlg);
		auto *tgtCombo = new QComboBox(&dlg);
		auto *winSpin = new QSpinBox(&dlg);
		auto *lagSpin = new QSpinBox(&dlg);
		auto *corrSpin = new QDoubleSpinBox(&dlg);

		winSpin->setMinimumWidth(120);
		lagSpin->setMinimumWidth(120);
		corrSpin->setMinimumWidth(120);

		winSpin->setRange(MIN_WINDOW_MS, MAX_WINDOW_MS);
		winSpin->setSingleStep(50);
		lagSpin->setRange(50, MAX_LAG_MS);
		lagSpin->setSingleStep(25);
		corrSpin->setRange(0.0, 1.0);
		corrSpin->setSingleStep(0.01);
		corrSpin->setDecimals(2);

		std::string ref_name;
		std::string tgt_name;
		uint32_t window_ms = DEFAULT_WINDOW_MS;
		uint32_t max_lag_ms = 500;
		float corr_threshold = MIN_CORR_THRESHOLD;

		pthread_mutex_lock(&dm->lock);
		ref_name = dm->ref_name;
		tgt_name = dm->target_name;
		window_ms = dm->window_ms;
		max_lag_ms = dm->max_lag_ms;
		corr_threshold = dm->corr_threshold;
		pthread_mutex_unlock(&dm->lock);

		populate_source_combo(refCombo, ref_name);
		populate_source_combo(tgtCombo, tgt_name);
		winSpin->setValue((int)window_ms);
		lagSpin->setValue((int)max_lag_ms);
		corrSpin->setValue((double)corr_threshold);

		layout->addRow("Reference Source", refCombo);
		layout->addRow("Target Source", tgtCombo);
		layout->addRow("Analysis Window (ms)", winSpin);
		layout->addRow("Max Lag (ms)", lagSpin);
		layout->addRow("Correlation Threshold", corrSpin);

		auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
		layout->addWidget(buttons);
		QObject::connect(buttons, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
		QObject::connect(buttons, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);

		if (dlg.exec() != QDialog::Accepted)
			return;

		std::string new_ref = refCombo->currentText().toStdString();
		std::string new_tgt = tgtCombo->currentText().toStdString();
		uint32_t new_win = (uint32_t)winSpin->value();
		uint32_t new_lag = (uint32_t)lagSpin->value();
		float new_corr = (float)corrSpin->value();

		pthread_mutex_lock(&dm->lock);
		dm->ref_name = new_ref;
		dm->target_name = new_tgt;
		dm->window_ms = new_win;
		dm->max_lag_ms = new_lag;
		dm->corr_threshold = new_corr;
		pthread_mutex_unlock(&dm->lock);

		connect_ref(dm);
		connect_target(dm);
	}
};

static QPointer<QDockWidget> g_syncDock;

// Called from set_result after every measurement; pulls copies from dm to avoid dangling pointers
static void update_dock_ui(audio_sync_data *dm)
{
	blog(LOG_INFO, "[ADM DEBUG] Entering update dock ui");
	if (!g_syncDock)
		return;

	auto *widget = qobject_cast<SyncDockWidget *>(g_syncDock->widget());
	if (!widget)
		return;

	std::string delay_copy;
	std::string notes_copy;
	double corr_copy = 0.0;
	std::string ref_name;
	std::string tgt_name;
	pthread_mutex_lock(&dm->lock);
	delay_copy = dm->last_delay_text;
	notes_copy = dm->last_delay_valid ? dm->last_notes : "";
	corr_copy = dm->last_correlation;
	ref_name = dm->ref_name;
	tgt_name = dm->target_name;
	bool valid = dm->last_delay_valid;
	pthread_mutex_unlock(&dm->lock);

	QMetaObject::invokeMethod(
		widget,
		[widget, delay_copy, notes_copy, corr_copy, ref_name, tgt_name, valid] {
			Q_UNUSED(corr_copy);
			widget->updateResult(QString::fromStdString(delay_copy), QString::fromStdString(notes_copy),
					     valid);
			widget->updateSourceNames(QString::fromStdString(ref_name), QString::fromStdString(tgt_name));
		},
		Qt::QueuedConnection);
}

static const char *k_dock_id = "audio-sync-analyzer";

static void create_dock_widget()
{
	if (!g_dm || g_syncDock)
		return;

	g_syncDock = new QDockWidget("Audio Sync Analyzer", nullptr);
	g_syncDock->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetMovable |
				QDockWidget::DockWidgetFloatable);
	g_syncDock->setWidget(new SyncDockWidget(g_dm));
	g_syncDock->resize(360, 500);
	g_syncDock->setMaximumWidth(420);

	obs_frontend_add_custom_qdock(k_dock_id, g_syncDock);

	QObject::connect(g_syncDock, &QDockWidget::destroyed, [] {
		obs_frontend_remove_dock(k_dock_id);
		g_syncDock = nullptr;
	});
}

static void destroy_dock_widget()
{
	if (g_syncDock) {
		obs_frontend_remove_dock(k_dock_id);
		g_syncDock = nullptr;
	}
}

static void audio_sync_frontend_shutdown()
{
	if (!g_dm)
		return;

	destroy_dock_widget();

	pthread_mutex_lock(&g_dm->lock);
	bool join_needed = g_dm->average_thread_active;
	g_dm->average_stop = true;
	pthread_mutex_unlock(&g_dm->lock);
	if (join_needed)
		pthread_join(g_dm->average_thread, nullptr);

	if (g_dm->target) {
		obs_source_remove_audio_capture_callback(g_dm->target, capture_target, g_dm);
		obs_source_release(g_dm->target);
		g_dm->target = nullptr;
	}
	if (g_dm->ref) {
		obs_source_remove_audio_capture_callback(g_dm->ref, capture_ref, g_dm);
		obs_source_release(g_dm->ref);
		g_dm->ref = nullptr;
	}

	pthread_mutex_destroy(&g_dm->lock);
	bfree(g_dm->ref_buffer);
	bfree(g_dm->tgt_buffer);
	delete g_dm;
	g_dm = nullptr;
}

static void tools_menu_action(void *data)
{
	UNUSED_PARAMETER(data);
	if (!g_dm)
		return;

	if (!g_syncDock)
		create_dock_widget();

	if (g_syncDock) {
		g_syncDock->show();
		g_syncDock->raise();
		g_syncDock->activateWindow();
	}
}

static void audio_sync_frontend_init()
{
	if (g_dm)
		return;

	g_dm = new audio_sync_data();
	pthread_mutex_init(&g_dm->lock, nullptr);
	g_dm->sample_rate = audio_output_get_sample_rate(obs_get_audio());
	g_dm->audio_format = AUDIO_FORMAT_FLOAT_PLANAR;
	g_dm->capacity = ms_to_samples(BUFFER_SECONDS * 1000u, g_dm->sample_rate);
	g_dm->ref_buffer = static_cast<float *>(bzalloc(g_dm->capacity * sizeof(float)));
	g_dm->tgt_buffer = static_cast<float *>(bzalloc(g_dm->capacity * sizeof(float)));
	g_dm->tgt_pos = g_dm->ref_pos = 0;
	g_dm->tgt_count = g_dm->ref_count = 0;
	g_dm->last_delay_valid = false;
	g_dm->window_ms = DEFAULT_WINDOW_MS;
	g_dm->max_lag_ms = 500;
	g_dm->corr_threshold = MIN_CORR_THRESHOLD;
	g_dm->debug_enabled = false;
	g_dm->average_in_progress = false;
	g_dm->average_stop = false;
	g_dm->average_thread_active = false;

	design_bandpass_filter(BANDPASS_LOW_Hz, BANDPASS_HIGH_Hz, g_dm->sample_rate, &g_dm->bp_coeffs);

	obs_frontend_add_save_callback(frontend_save_cb, g_dm);
	obs_frontend_add_tools_menu_item("Audio Sync Analyzer", tools_menu_action, nullptr);

	create_dock_widget();
	update_dock_ui(g_dm);
}

extern "C" void audio_sync_frontend_init_module()
{
	audio_sync_frontend_init();
}

extern "C" void audio_sync_frontend_shutdown_module()
{
	obs_frontend_remove_save_callback(frontend_save_cb, g_dm);
	audio_sync_frontend_shutdown();
}

#include "audio-sync-analyzer.moc"
