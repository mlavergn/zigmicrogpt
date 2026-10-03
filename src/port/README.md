# port

The straight port of Karpathy's [`microgpt.py`](../../microgpt.py) to Zig: the same
algorithm, transcribed line by line into a single file, with no optimization. Read this to
see how the Python maps onto Zig. The optimized build in `src/` is what `zig build` ships.

| File             | Purpose                                                         |
| :--------------- | :-------------------------------------------------------------- |
| `gpt_main.zig`   | The straight port. The complete algorithm, start to finish.     |
| `gpt_docs.zig`   | The same code as `gpt_main.zig`, with full headerdoc.           |
| `gpt_random.zig` | The parts of CPython's `random` module that `microgpt.py` uses. |

Seeded identically and doing the same arithmetic in the same order, the port prints the same
losses and the same names as `microgpt.py`.

## Running

From the repository root:

```sh
zig build port -Drelease
```

The port reads `input.txt` from the current directory. `zig build test` compiles both
`gpt_main.zig` and `gpt_docs.zig`, so a change that breaks either one fails the tests.

## Notes

- This directory is exempt from the structural rules in `styleguide/ZIGSTYLE.md` (module
  barrel, one struct per file, `Self` and methods, logging), which the rest of `src/` follows.
  Splitting the port across files would break its line-by-line mapping to `microgpt.py`.
- `gpt_main.zig` here shares its name with the optimized `../gpt_main.zig`, the program
  `zig build` ships. The directory is what tells them apart.
- `gpt_main.zig` and `gpt_docs.zig` must stay identical apart from comments. A change to one
  belongs in the other.
- The port shares `../gpt_pow.zig` with the optimized build. Zig won't import a file outside
  the importing module's directory, so `build.zig` passes it in as the module `gpt_pow`.
  That's also why the port builds through `zig build` rather than `zig run`.
- `gpt_random.zig` is also the reference for the optimized `../gpt_random.zig`, which
  is tested against it draw for draw and benchmarked against it by `make bench`.
