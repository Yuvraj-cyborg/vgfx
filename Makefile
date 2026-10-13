CARGO ?= cargo
PYTHON ?= python3

.PHONY: all build test run-2d run-3d run-windowed run-doom run-model ffi-c ffi-py \
	check-all cross-linux-aarch64 cross-linux-x86_64 cross-windows clean

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

run-doom:
	$(CARGO) run --example doom

run-model:
	$(CARGO) run --example offscreen_model

ffi-c:
	$(MAKE) -C examples/c_example run

ffi-py: build
	$(PYTHON) examples/ffi_python/render.py

check-all:
	$(CARGO) check --target aarch64-apple-darwin
	$(CARGO) check --target x86_64-unknown-linux-gnu

cross-linux-aarch64:
	cross build --target aarch64-unknown-linux-gnu

cross-linux-x86_64:
	cross build --target x86_64-unknown-linux-gnu

cross-windows:
	cross build --target x86_64-pc-windows-gnu

clean:
	$(CARGO) clean
	$(MAKE) -C examples/c_example clean