#
# MicroGPT
#

.DEFAULT_GOAL := all
.NOTPARALLEL:

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BINARY      := ./zig-out/bin/microgpt
OUTDIR      := zig-out

CORPUS      := input.txt
CORPUS_URL  := https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt
REPO_URL    := https://github.com/mlavergn/microgptzig

# The Zig sources to format and lint. Named explicitly rather than `.` because
# speed/ is a separate repository whose code is not ours to rewrite or gate on.
ZIG_SOURCES := build.zig $(wildcard src/*.zig) $(wildcard src/port/*.zig)

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------

all: clean build run

clean:
	rm -rf $(OUTDIR)

build:
	zig build -Drelease

run:
	time $(BINARY)

# Build the straight port (src/port/gpt_main.zig) on its own, as zig-out/bin/port_gpt_main.
port:
	zig build port-build -Drelease

test:
	zig build test

# Time each component against its reference port, then the CLI end to end
# (train + sample). Always ReleaseFast for the native CPU; see src/gpt_bench.zig.
bench:
	zig build bench

# Full pre-commit gate. `.NOTPARALLEL` keeps the prerequisites in order.
validate: clean format lint build test
	@echo "done"

# -----------------------------------------------------------------------------
# Format & lint
# -----------------------------------------------------------------------------

format:
	zig fmt $(ZIG_SOURCES)

# Token-level style, then the rule set in styleguide/. zlintpre exits 0 whatever
# it finds, so its summary line is what gates. zlint gets its files on stdin:
# walking the tree on its own, it would also lint the untracked speed/.
lint:
	@test -f styleguide/zlint.json || { echo "lint: styleguide/ missing, run: make subpull"; exit 1; }
	@output=$$(zlintpre $(ZIG_SOURCES) 2>&1); echo "$$output"; \
	echo "$$output" | grep -q 'found 0 failures' || { echo "lint: zlintpre findings above"; exit 1; }
	printf '%s\n' $(ZIG_SOURCES) | zlint -c styleguide --deny-warnings --stdin

# -----------------------------------------------------------------------------
# Reference
# -----------------------------------------------------------------------------

python:
	time python3 microgpt.py

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

data:
	curl -o $(CORPUS) "$(CORPUS_URL)"

# -----------------------------------------------------------------------------
# Submodules
# -----------------------------------------------------------------------------

# Init/update all submodules (the shared styleguide that `lint` reads).
subpull:
	git submodule update --init --recursive

# -----------------------------------------------------------------------------
# Links
# -----------------------------------------------------------------------------

repo:
	open "$(REPO_URL)"

.PHONY: all bench build clean data format lint port python repo run subpull test validate
