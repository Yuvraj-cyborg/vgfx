CARGO ?= cargo
PYTHON ?= python3

.PHONY: all build test run-2d run-3d run-windowed ffi-c ffi-py clean

all: build

build:
	$(CARGO) build

test:
	$(CARGO) test

run-2d:
	$(CARGO) run --example offscreen_2d

run-3d:
	$(CARGO) run --example offscreen_3d

run-windowed:
	$(CARGO) run --example windowed

ffi-c:
	$(MAKE) -C examples/c_example run

ffi-py: build
	$(PYTHON) examples/ffi_python/render.py

clean:
	$(CARGO) clean
	$(MAKE) -C examples/c_example clean