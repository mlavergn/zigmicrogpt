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

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------

all: clean build run

clean:
	rm -rf $(OUTDIR)

build:
	zig build

run:
	time $(BINARY)

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
# Links
# -----------------------------------------------------------------------------

repo:
	open "$(REPO_URL)"

.PHONY: all build clean data python repo run
