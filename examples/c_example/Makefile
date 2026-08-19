CARGO ?= cargo
CC ?= cc
ROOT := ../../

DYLIB := $(ROOT)target/debug/libvgfx.dylib
INC := -I$(ROOT)include

.PHONY: all run clean

all: c_example

c_example: main.c $(DYLIB)
	$(CC) -O2 $(INC) main.c -L$(ROOT)target/debug -lvgfx -o c_example

$(DYLIB):
	$(CARGO) build --manifest-path $(ROOT)Cargo.toml

run: c_example
	./c_example

clean:
	rm -f c_example c_example.png
