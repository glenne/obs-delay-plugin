# Simple helper Makefile for local builds

CMAKE ?= cmake
PRESET ?= macos
BUILD_DIR ?= build_macos
CONFIG ?= RelWithDebInfo
INSTALL_PREFIX ?= $(CURDIR)/release/$(CONFIG)

.PHONY: all configure build install clean

mac:
	CI=1 ./.github/scripts/build-macos

format:
	clang-format -i src/plugin* src/audio*

all: build

configure:
	$(CMAKE) --preset $(PRESET)

buildonly: 
	$(CMAKE) --build --preset $(PRESET)

build: configure
	$(CMAKE) --build --preset $(PRESET)

install: build
	$(CMAKE) --install $(BUILD_DIR) --config $(CONFIG) --prefix $(INSTALL_PREFIX)

clean:
	rm -rf $(BUILD_DIR)
