# MicroGPT in Zig

This project contains a Zig port of microgpt.py from @karpathy.

The microgpt.py project is a spectacular piece of work that reduces the fundamentals behind a general purpose transformer (GPT) to its bare minimum while eschewing any pre-built dependencies. The entire process is plainly visible.

In the spirit of Andrej's project, this project is also open source.

## What it does

The program trains a small character-level GPT from scratch on a corpus of 32,000 names, then uses what it learned to invent new ones. Training and inference happen in a single run, taking nuder a minute from start to finish.

```
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 2.6497

--- inference (new, hallucinated names) ---
sample  1: kamon
sample  2: ann
sample  3: karai
sample  4: jaire
sample  5: vialan
```

Nothing is downloaded at runtime, nothing is saved to disk, and there is no model to load. Every weight the model has is created, trained, and discarded within the one process.

## Running it

Requires a Zig toolchain. The name corpus (`input.txt`) is included; `make data` re-fetches it from upstream if needed.

```sh
make          # clean, build, and run
make build    # build only
make run      # run the already-built binary
make python   # run the original microgpt.py for comparison
```

## Why Zig?

I chose Zig to give this framework maximum hardware efficiency with zero runtime overhead. By stripping away hidden allocations, implicit control flow, and garbage collection, Zig provides the deterministic execution deep learning demands, outperforming similar C++ and Rust implementations. Furthermore, its first-class C interoperability allows seamless integration with GPU drivers, while instant cross-compilation yields lean, self-contained binaries optimized for large-scale compute environments.

[https://ziglang.org](https://ziglang.org)

The port is a faithful one. Both programs are seeded identically and reimplement the same arithmetic in the same order, so the Zig and the Python produce the same model and the same names — the Zig just gets there 10x sooner.

## What's here

| File              | Purpose                                              |
| :---------------- | :--------------------------------------------------- |
| `microgpt.zig`    | The port. The complete algorithm, start to finish.   |
| `microgptdoc.zig` | The port with full headerdoc.                        |
| `microgpt.py`     | Karpathy's original, kept verbatim as the reference. |
| `input.txt`       | The corpus of names the model learns from.           |

## Optimized

| `microgptmax.zig` | Optimized rewrite of the port.                       |
| `random.zig`      | The random number generation the port depends on.    |

## Performance

| Implementation                 |  Wall time | CPU user time | CPU kernel time |
| :----------------------------- | ---------: | ------------: | --------------: |
| Zig with Random + std.math.Pow |  5.75 real |     5.24 user |        0.02 sys |
| Python                         | 58.31 real |    57.79 user |        0.34 sys |

## Credits

The original [microgpt.py](https://github.com/karpathy) is the work of Andrej Karpathy, as is the names corpus, which comes from his makemore project. This port is released under the terms in [LICENSE.md](LICENSE.md).
