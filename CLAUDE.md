# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A study copy of Karpathy's `microgpt` — a character-level GPT in pure, dependency-free Python
(scalar autograd, hand-rolled Adam, no NumPy/PyTorch, CPU only). At the root: `microgpt.py`,
`input.txt` (names corpus), `Makefile`. Plus `microgptzig/`, a Zig 0.16 port (see below).

## Vendored, not authored

`microgpt.py` and `input.txt` are **downloaded verbatim from upstream** and intended to stay that
way. Do not edit `microgpt.py`, reformat it, or "clean it up" — put experiments in a new file
instead. This also means: no linter, no formatter, no type hints, no `requirements.txt` for the
Python side. Its style (one-line `def` bodies, >130-char lines, column-aligned comments, literate
section headers) is deliberate; black/ruff would rewrite most of the file.

This rule covers the root Python only. `microgptzig/` is our own code — `zig fmt` it normally.

## microgptzig/ — the Zig 0.16 port

`zig build test` (CPython parity + autograd checks), `zig build run` (train and sample, ~7s).
The binary reads `input.txt` from the CWD, falling back to `../`, so it works from either
directory. It does not download the corpus.

The port is a **bit-for-bit** reimplementation, not a paraphrase: it reimplements CPython's
MT19937 and its `gauss`/`shuffle`/`choices`, hand-rolls `pow` rather than using `std.math.pow`
(which disagrees in the last bit — `std.math.pow(2, -0.5)` is one ulp low), and reproduces microgpt.py's computation
graph node-for-node, including the `term + 0` node that Python's builtin `sum` implicitly creates.
Comments marked `parity:` mark each such concession. Changing any of them silently breaks the
correspondence with the Python, so treat them as load-bearing.

Two intentional deviations from the Python, both semantics-preserving: `Value` objects become
indices into a flat `Tape` that is truncated back to the parameters each step, and `build_topo`'s
recursion becomes an explicit stack (same post-order, no recursion limit).

This directory is **not a git repository**. There is no way to recover an overwritten or deleted
file. Confirm before anything destructive.

## Commands

- `make train` — run training + sampling (`python3 microgpt.py`). Must run from the repo root:
  `input.txt` is opened by relative path. No shebang despite the executable bit — always `python3`.
- `make code` / `make data` — **re-fetch from upstream, overwriting `microgpt.py` / `input.txt`.**
  Not build steps. Never run to "refresh" without checking for local changes first.
- `make docs` uses macOS `open`; it fails on Linux.
- `make musictrain` is broken — it runs `musictrain.py`, but `make music` downloads
  `micromusic.py`. Use `python3 micromusic.py` (the Makefile has not been corrected).

There is no build, test, or lint target, and no CI.

## Gotchas

- **No checkpointing.** Train and inference are one process; weights are lost on exit. Sampling
  again means retraining.
- **Slow by design** — scalar Python autograd, ~1000 steps single-threaded. Expect minutes. Don't
  launch a training run to verify an unrelated change.
- Progress prints with `end='\r'`; piped output becomes one giant line. Pipe through
  `tr '\r' '\n'` to read it.
- `random.seed(42)` is set at import, so runs are reproducible.
- `make data` pulls names.txt from a moving branch ref, while `microgpt.py`'s built-in fallback
  download pins commit `988aa59`. They can drift apart.
- `build_topo` recurses over the whole computation graph; raising `block_size` or `n_layer` risks
  hitting Python's recursion limit.
