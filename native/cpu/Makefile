SHELL := /bin/sh

CC := xcrun clang
PYTHON ?= python3
BUILD_DIR := build
ROOT_DIR := $(abspath ../..)
INCLUDE_DIR := include
SOURCE := src/blackhole_cpu.c
HEADER := include/blackhole_cpu.h
LIBRARY := $(BUILD_DIR)/libblackhole_cpu.dylib
CONTRACT_TEST := $(BUILD_DIR)/test_contract
SANITIZER_TEST := $(BUILD_DIR)/test_contract_sanitized

STRICT_FP_FLAGS := \
	-std=c17 -O3 -g0 -fPIC \
	-fno-fast-math -ffp-contract=off -fno-associative-math \
	-fno-reciprocal-math -fno-unsafe-math-optimizations \
	-frounding-math -fno-finite-math-only -fno-lto
WARNING_FLAGS := \
	-Wall -Wextra -Wpedantic -Werror -Wconversion -Wsign-conversion \
	-Wshadow -Wstrict-prototypes -Wmissing-prototypes -Wundef \
	-Wdouble-promotion -Wformat=2 -Wcast-align
CPPFLAGS := -I$(INCLUDE_DIR)
CFLAGS := $(STRICT_FP_FLAGS) $(WARNING_FLAGS)

.PHONY: all clean test audit sanitize benchmark

all: $(LIBRARY) $(CONTRACT_TEST)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIBRARY): $(SOURCE) $(HEADER) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -dynamiclib \
		-Wl,-install_name,@rpath/libblackhole_cpu.dylib \
		-Wl,-dead_strip -o $@ $(SOURCE) -lm

$(CONTRACT_TEST): tests/test_contract.c $(SOURCE) $(HEADER) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ tests/test_contract.c $(SOURCE) -lm

$(SANITIZER_TEST): tests/test_contract.c $(SOURCE) $(HEADER) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -g -fno-omit-frame-pointer \
		-fsanitize=address,undefined -o $@ tests/test_contract.c $(SOURCE) -lm

test: all audit sanitize
	$(CONTRACT_TEST)
	cd $(ROOT_DIR) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest -v \
		native.cpu.tests.test_parity \
		native.cpu.tests.test_whole_ray_parity

audit: $(LIBRARY)
	cd $(ROOT_DIR) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) \
		native/cpu/tools/audit_binary.py native/cpu/$(LIBRARY)

sanitize: $(SANITIZER_TEST)
	ASAN_OPTIONS=halt_on_error=1 \
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
		$(SANITIZER_TEST)

benchmark: $(LIBRARY)
	cd $(ROOT_DIR) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) \
		-m native.cpu.tests.benchmark_rhs
	cd $(ROOT_DIR) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) \
		-m native.cpu.tests.benchmark_dopri

clean:
	rm -rf $(BUILD_DIR)
