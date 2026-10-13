CARGO ?= cargo
CC ?= cc
ROOT := ../../

DYLIB := $(ROOT)target/debug/libvgfx.dylib
INC := -I$(ROOT)include

# Matches the linking notes in include/vgfx.h.
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
LDLIBS += -lobjc -framework Metal -framework Foundation -framework QuartzCore -framework AppKit
endif

.PHONY: all run clean rust-build

all: c_example

c_example: main.c $(ROOT)include/vgfx.h rust-build
	$(CC) -O2 $(INC) main.c -L$(ROOT)target/debug -lvgfx $(LDLIBS) -o c_example

rust-build:
	$(CARGO) build --manifest-path $(ROOT)Cargo.toml

run: c_example
	./c_example

clean:
	rm -f c_example c_example.png
