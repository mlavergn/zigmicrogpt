# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Zig 0.17 port of Karpathy's `microgpt`, a character-level GPT in pure, dependency-free Python
(scalar autograd, hand-rolled Adam, no NumPy/PyTorch, CPU only). The Python original
`microgpt.py` and its names corpus `input.txt` sit at the root as the reference; the Zig lives
in `src/`. `build.zig`, `build.zig.zon`, and the `Makefile` stay at the root.

## Layout

| Path | What it is |
| :--- | :--- |
| `src/module.zig` | Barrel: re-exports every type below; `zig build test` root (`refAllDecls`) |
| `src/gpt_main.zig` | The optimized CLI's entry point (`zig build` root): `std_options` and `main` only |
| `src/gpt_tape.zig` | `Tape`: the autograd graph, its ops and `backward` |
| `src/gpt_model.zig` | `Model` (plus `Matrix`, `Layer`, `Cache`): weights, `gpt` forward pass, `softmax`, `rmsnorm` |
| `src/gpt_trainer.zig`, `src/gpt_sampler.zig` | `Trainer` (training loop + Adam), `Sampler` (inference) |
| `src/gpt_tokenizer.zig` | `Tokenizer` |
| `src/gpt_random.zig` | `Random`: vectorized CPython-compatible RNG for the optimized build |
| `src/gpt_pow.zig` | `Pow`: `Pow.pow` is the only power function the Zig uses (see below) |
| `src/gpt_bench.zig` | `Bench`, plus `main`: the `zig build bench` executable's root |
| `src/port/` | The straight port of microgpt.py; see its `README.md` |
| `src/port/gpt_main.zig`, `gpt_docs.zig` | Single-file port; `_docs` is the same code with full headerdoc. Not to be confused with `src/gpt_main.zig` |
| `src/port/gpt_random.zig` | The port's RNG, plain; also the reference `src/gpt_random.zig` is tested against |
| `styleguide/` | Submodule (`inferise/styleguide`): `ZIGSTYLE.md`, which `src/` follows, and `zlint.json` |
| `speed/` | A **separate, untracked git repo** with its own `build.zig`. Not part of this project |

Run the straight port with `zig build port -Drelease`. Not `zig run`: the ports import
`src/gpt_pow.zig`, which is outside their directory, so `build.zig` passes it in as the named
module `gpt_pow`, imported as `const Pow = @import("gpt_pow").Pow;`. Zig refuses a relative
`../` import. `zig build test` compiles both ports. `port/gpt_main.zig` and `gpt_docs.zig` must
stay identical apart from comments. Every file shares the `gpt_` prefix; keep it for new ones.

## Vendored, not authored

`microgpt.py` and `input.txt` are **downloaded verbatim from upstream** and intended to stay that
way. Do not edit `microgpt.py`, reformat it, or "clean it up". Put experiments in a new file
(the scratchpad works). No linter, formatter, type hints, or `requirements.txt` for the Python.
Its style (one-line `def` bodies, >130-char lines, column-aligned comments) is deliberate.

This rule covers the root Python only. `src/` is our own code: `zig fmt` it normally.

## Zig style

`src/` follows `styleguide/ZIGSTYLE.md` (the submodule; read it before adding code). Decisions
made applying it here:

- **`src/port/` is exempt from the structural rules** (barrel, one struct per file, `Self`,
  methods, logging). It is a line-by-line transcription of microgpt.py in one file, and
  restructuring it would break that mapping. Fix only naming and test dividers there.
- **Struct names drop the `gpt_` prefix**: `gpt_tape.zig` holds `Tape`, not `GptTape`. The
  prefix is a file-name convention only.
- **Logging**: every file declares `std.log.scoped(.microgptzig_<file>)`. Entry traces go on
  per-run and per-step functions only (`init`, `deinit`, `run`, `rewind`, `backward`, ...),
  never on per-node, per-draw or per-call ones (tape ops, RNG draws, `Pow`, `gpt`, `softmax`).
  Both executables set `std_options.log_level = .info`, which compiles the traces out; set
  `.debug` to see them (~2k lines per run).
- zlint fails on an unused `log`, so a file whose functions are all hot logs something cold
  (`Pow.parity` logs its rare fallback to std).
- `Random.seed` runs at comptime inside `seeded`; its trace is guarded by `@inComptime()`
  because logging cannot be evaluated at comptime in test builds.
- A new public type goes in `src/module.zig`, or its tests are never discovered. Tests sit at
  the bottom of their struct's file under `// Unit Tests`.

## Parity

The port is a **bit-for-bit** reimplementation, not a paraphrase: it reimplements CPython's
MT19937 and its `gauss`/`shuffle`/`choices`, and reproduces microgpt.py's computation graph
node-for-node, including the `term + 0` node that Python's builtin `sum` implicitly creates.
Comments marked `parity:` mark each such concession. Changing any of them silently breaks the
correspondence with the Python, so treat them as load-bearing.

Two intentional deviations, both semantics-preserving: Python's `Value` objects become `u32`
indices into a flat tape (`Tape` in `gpt_tape.zig`; `Value` in the port), truncated
back to the parameters each step; and `build_topo`'s recursion becomes an explicit stack (same
post-order, no recursion limit). `backward`'s walk also skips parameters (index below
`n_params`) and reads each node once; that halved the sort, which had been half of a run. Its
order is pinned by the test "backward sums gradients in build_topo's order", which checks it
bit for bit against a literal recursive `build_topo` on a random shared graph. Nothing else
catches an order change: a swapped child order and a reverse tape scan both pass every other
test.

### `gpt_pow.zig`

Never call `std.math.pow` directly; call `mod.Pow.pow(x, y)` (`Pow.pow` in the port).

- `Pow.impl` selects the implementation at comptime: `.parity` (default) or `.std`
  (`std.math.pow`). Flip it to A/B with `make bench`'s `gpt_main` row; the measured
  difference is within noise (~0.58 s vs ~0.59 s). Ship `.parity`.
- Speed is not a reason to prefer `.std`. Calls with constant exponents (Adam's 8.4M `2` and
  `0.5`) inline to a single operation. Of the runtime-exponent calls `make bench` times,
  `.parity` beats std ~2.2x on `-2` and `-1.5`, ties on `-0.5`, and loses only on Adam's bias
  correction (`y=1000`, ~2k calls per run). All of `pow` is ~0.2% of a run.
- `.parity` is double-double repeated squaring, correctly rounded, **for exponents that are
  multiples of 0.5 only** (it asserts). Every exponent the program uses qualifies.
- `std.math.pow` is not good enough: `1 / @sqrt` rounds twice (`2 ** -0.5` lands one ulp low),
  and plain-f64 squaring reaches 209 ulps off at `0.9 ** 1000` (Adam's bias correction).
- **macOS's libm `pow`, which CPython calls, is not correctly rounded either.** It misrounds
  ~0.13% of inputs, even `pow(x, 2)` against `x * x`. To verify a pow change, compare against
  exact rounding (`fractions.Fraction` / `decimal`), never against Python's `x ** y`.
- The training printout (4-decimal losses, sampled names) matches `microgpt.py` with **either**
  implementation, so it cannot detect a pow regression. The parity test can.

## Commands

- `make build` / `make run`: ReleaseFast build (`zig build -Drelease`), then a timed run.
  `make` alone is `clean build run`.
- `make test`: `zig build test`. `make validate`: clean, format, lint, build, test (the gate).
- `make bench` (= `zig build bench`): always ReleaseFast. Times each component against its
  reference (min/median/mean/max ns per op, speedup of medians): `random` against the plain
  port, `pow` (`.parity`) against `std.math.pow`. Then the CLIs end to end: 5 runs each of
  the optimized build and `port/gpt_main.zig`, alternating, with the optimized build's
  speedup (~2.0x: ~0.46 s vs ~0.92 s).
  Zig has no built-in benchmark harness; this uses `std.Io.Clock.awake`. To add a component,
  give `Bench` a method like `random` and call it from `components`. Every case must thread mutable
  state through its calls: a pure call with fixed inputs gets hoisted out of the timed region
  and reports ~0 ns.
- `make format` / `make lint`: `zig fmt`, then `zlintpre` + `zlint -c styleguide`, both scoped
  to `build.zig`, `src/*.zig` and `src/port/*.zig` via `ZIG_SOURCES`. Don't widen them to
  `.`: that sweeps in `speed/`. `make lint` needs the submodule: `make subpull`.
- `make python`: run `microgpt.py` (~60 s). `make data` **overwrites `input.txt`** from upstream.

Optimization is `-Drelease` (meaning ReleaseFast), not `-Doptimize`: `build.zig` sets
`preferred_optimize_mode`, which replaces the `-Doptimize` option. Plain `zig build` is Debug
(~4 s per run vs ~0.46 s). The binary reads `input.txt` from the CWD, falling back to `../`.

## Gotchas

- Tape ops other than `leaf` need `tape.reserve(n)` first; they push into preallocated
  capacity. Tests building a tape by hand must call it.
- No checkpointing. Train and inference are one process; weights are lost on exit.
- Progress prints with `\r` in both languages; pipe through `tr '\r' '\n'` to read or diff it.
  Zig and Python output are line-for-line comparable that way.
- `random.seed(42)` is fixed, so runs are reproducible.
- `make data` pulls names.txt from a moving branch ref, while `microgpt.py`'s built-in fallback
  download pins commit `988aa59`. They can drift apart.
- `build_topo` in the Python recurses over the whole graph; raising `block_size` or `n_layer`
  risks Python's recursion limit. The Zig has no such limit.
