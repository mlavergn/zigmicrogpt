// ---------------------------------------------------------------------------
// Benchmarks: each component against its reference (the plain port, or std
// where there is no port), then the CLI end to end. Run with `zig build
// bench`, which always builds ReleaseFast and passes the optimized CLI's path,
// then the straight port's, as arguments. This file is also that executable's
// root: `main` sits below `Bench`.
// ---------------------------------------------------------------------------

const std = @import("std");
const log = std.log.scoped(.microgptzig_bench);
const mod = @import("module.zig");

/// The straight port's RNG, the reference `mod.Random` is timed against.
const PortRandom = @import("port/gpt_random.zig").Random;

/// Summary of one case's samples, in nanoseconds per operation.
const Stats = struct {
    const Self = @This();
    min: f64,
    median: f64,
    mean: f64,
    max: f64,

    /// Sorts `xs` in place and summarizes it.
    fn of(xs: []f64) Self {
        std.mem.sort(f64, xs, {}, std.sort.asc(f64));
        var sum: f64 = 0;
        for (xs) |x| sum += x;
        return .{
            .min = xs[0],
            .median = xs[xs.len / 2],
            .mean = sum / @as(f64, @floatFromInt(xs.len)),
            .max = xs[xs.len - 1],
        };
    }
};

/// One exponent `Bench` times `pow` at, with inputs drawn from `[lo, hi)`.
const PowCase = struct {
    const Self = @This();
    name: []const u8,
    y: f64,
    lo: f64,
    hi: f64,
};

/// Times each component against its reference, then the CLIs end to end,
/// printing a table to `out`.
pub const Bench = struct {
    const Self = @This();

    /// Timed samples per case, after one untimed warm-up. Min is the least
    /// noisy estimate of what the code can do; median shows what a typical
    /// run sees.
    const samples = 15;
    /// Full runs of each CLI. Each trains the model, so fewer of them.
    const cli_samples = 5;

    // `random`: sized like the program's own use of each call, scaled up until
    // one sample takes long enough for the clock to resolve.
    const seed_ops = 1_000;
    const gauss_ops = 100_000; // weight init draws ~4k per run
    const shuffle_len = 32_033; // the number of names in input.txt
    const choices_ops = 100_000;
    const vocab = 27; // 26 letters plus BOS

    // `pow`: only the exponents the program calls `pow` with at runtime.
    // Constant ones (Adam's 2 and 0.5, softmax's -1) are inlined and folded to
    // one operation.
    const pow_ops = 100_000;
    /// Inputs per case, a power of two so indexing is a mask, not a divide.
    const pow_len = 1024;
    const pow_cases = [_]PowCase{
        .{ .name = "y=-2", .y = -2, .lo = 0.5, .hi = 2.5 }, // softmax's derivative
        .{ .name = "y=-0.5", .y = -0.5, .lo = 0.5, .hi = 2.5 }, // rmsnorm
        .{ .name = "y=-1.5", .y = -1.5, .lo = 0.5, .hi = 2.5 }, // rmsnorm's derivative
        .{ .name = "y=1000", .y = 1000, .lo = 0.85, .hi = 0.99 }, // Adam's bias, last step
    };

    io: std.Io,
    out: *std.Io.Writer,

    /// Creates a benchmark that times with `io`'s clock and prints to `out`.
    pub fn init(io: std.Io, out: *std.Io.Writer) Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        return .{ .io = io, .out = out };
    }

    /// Releases nothing: `out` belongs to the caller.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        _ = self;
    }

    /// Times every component against its reference.
    pub fn components(self: *Self) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        try self.out.print("{s:<19} {s:>9} {s:>9} {s:>9} {s:>9} {s:>9}  {s:>7}\n", .{ "component (ns/op)", "min", "median", "mean", "max", "ref med", "speedup" });
        try self.random();
        try self.pow();
        try self.out.flush();
    }

    /// Times full runs of the optimized CLI and the straight port, alternating
    /// between them so drift in the machine (heat, other load) falls on both.
    pub fn cli(self: *Self, optimized: []const u8, straight: []const u8) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        // Warm the page cache and input.txt for both.
        _ = try self.runCli(optimized);
        _ = try self.runCli(straight);
        var opt_secs: [cli_samples]f64 = undefined;
        var port_secs: [cli_samples]f64 = undefined;
        for (&opt_secs, &port_secs) |*o, *p| {
            o.* = try self.runCli(optimized);
            p.* = try self.runCli(straight);
        }
        const opt = Stats.of(&opt_secs);
        const port = Stats.of(&port_secs);
        try self.out.print("\n{s:<19} {s:>9} {s:>9} {s:>9} {s:>9}  {s:>7}\n", .{ "cli (seconds)", "min", "median", "mean", "max", "speedup" });
        try self.cliRow("gpt_main", opt, port.median / opt.median);
        try self.cliRow("port/gpt_main", port, 1);
    }

    /// Times `samples` calls of `run`, each performing `ops` operations, and
    /// returns per-operation statistics. `run` returns a value derived from its
    /// work so the optimizer cannot delete the loop.
    fn measure(self: *const Self, ops: usize, comptime run: anytype, args: anytype) Stats {
        std.mem.doNotOptimizeAway(@call(.auto, run, args));
        var ns: [samples]f64 = undefined;
        for (&ns) |*s| {
            const t0 = std.Io.Clock.awake.now(self.io);
            std.mem.doNotOptimizeAway(@call(.auto, run, args));
            const elapsed = t0.durationTo(std.Io.Clock.awake.now(self.io)).nanoseconds;
            s.* = @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(ops));
        }
        return Stats.of(&ns);
    }

    /// Prints a component case: the optimized implementation, its reference,
    /// and the speedup between their medians.
    fn report(self: *Self, component: []const u8, case: []const u8, fast: Stats, ref: Stats) !void {
        try self.out.print("{s:<8} {s:<10} {d:>9.2} {d:>9.2} {d:>9.2} {d:>9.2} {d:>9.2}  {d:>6.2}x\n", .{
            component, case, fast.min, fast.median, fast.mean, fast.max, ref.median, ref.median / fast.median,
        });
    }

    /// `mod.Random` against `PortRandom`, call by call.
    fn random(self: *Self) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var fast: mod.Random = .{};
        var ref: PortRandom = .{};
        fast.seed(42);
        ref.seed(42);

        var next_fast: u64 = 42;
        var next_ref: u64 = 42;
        try self.report("random", "seed", self.measure(seed_ops, randomSeed, .{ mod.Random, &fast, &next_fast }), self.measure(seed_ops, randomSeed, .{ PortRandom, &ref, &next_ref }));
        try self.report("random", "gauss", self.measure(gauss_ops, randomGauss, .{ mod.Random, &fast }), self.measure(gauss_ops, randomGauss, .{ PortRandom, &ref }));

        var xs: [shuffle_len]u32 = undefined;
        for (&xs, 0..) |*x, i| x.* = @intCast(i);
        try self.report("random", "shuffle", self.measure(shuffle_len, randomShuffle, .{ mod.Random, &fast, &xs }), self.measure(shuffle_len, randomShuffle, .{ PortRandom, &ref, &xs }));

        var weights: [vocab]f64 = undefined;
        for (&weights, 0..) |*w, i| w.* = 1.0 / @as(f64, @floatFromInt(i + 1));
        var cum: [vocab]f64 = undefined;
        try self.report("random", "choices", self.measure(choices_ops, randomChoices, .{ mod.Random, &fast, &weights, &cum }), self.measure(choices_ops, randomChoices, .{ PortRandom, &ref, &weights, &cum }));
    }

    /// `mod.Pow.parity` against `std.math.pow`, exponent by exponent.
    fn pow(self: *Self) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        for (pow_cases) |c| {
            var xs: [pow_len]f64 = undefined;
            for (&xs, 0..) |*x, i| x.* = c.lo + (c.hi - c.lo) * @as(f64, @floatFromInt(i)) / pow_len;
            var y = c.y;
            std.mem.doNotOptimizeAway(&y);
            var at_parity: usize = 0;
            var at_std: usize = 0;
            try self.report("pow", c.name, self.measure(pow_ops, powCalls, .{ mod.Pow.parity, &xs, &y, &at_parity }), self.measure(pow_ops, powCalls, .{ stdPow, &xs, &y, &at_std }));
        }
    }

    // `next` carries over between calls, so every sample seeds with new values.
    // With a fixed input the call is pure, and the optimizer hoists it out of
    // the timed region entirely.
    fn randomSeed(comptime R: type, r: *R, next: *u64) u64 {
        var acc: u64 = 0;
        for (0..seed_ops) |_| {
            r.seed(next.*);
            next.* +%= 1;
            acc +%= @bitCast(r.gauss(0, 1));
        }
        return acc;
    }

    fn randomGauss(comptime R: type, r: *R) u64 {
        var acc: f64 = 0;
        for (0..gauss_ops) |_| acc += r.gauss(0, 0.08);
        return @bitCast(acc);
    }

    fn randomShuffle(comptime R: type, r: *R, xs: []u32) u64 {
        r.shuffle(u32, xs);
        return xs[0];
    }

    fn randomChoices(comptime R: type, r: *R, weights: []const f64, cum: []f64) u64 {
        var acc: usize = 0;
        for (0..choices_ops) |_| acc +%= r.choices(weights, cum);
        return acc;
    }

    fn stdPow(x: f64, y: f64) f64 {
        return std.math.pow(f64, x, y);
    }

    // `at` advances between calls, so no two samples see the same inputs, and
    // `y` arrives through a pointer, so the exponent is not a constant to fold,
    // as at the program's own out-of-line call sites.
    fn powCalls(comptime f: anytype, xs: *const [pow_len]f64, y: *const f64, at: *usize) u64 {
        var acc: f64 = 0;
        for (0..pow_ops) |i| acc += f(xs[(at.* + i) & (pow_len - 1)], y.*);
        at.* +%= 1;
        return @bitCast(acc);
    }

    /// Runs the CLI binary at `path` to completion with stdout discarded, so
    /// the terminal redrawing a progress line every step is not part of the
    /// time. Returns the elapsed seconds.
    fn runCli(self: *Self, path: []const u8) !f64 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const t0 = std.Io.Clock.awake.now(self.io);
        var child = try std.process.spawn(self.io, .{ .argv = &.{path}, .stdout = .ignore });
        const term = try child.wait(self.io);
        const elapsed = t0.durationTo(std.Io.Clock.awake.now(self.io)).nanoseconds;
        switch (term) {
            .exited => |code| if (code != 0) return error.CliFailed,
            else => return error.CliFailed,
        }
        return @as(f64, @floatFromInt(elapsed)) / std.time.ns_per_s;
    }

    /// Prints one CLI's run times, with its speedup over the straight port.
    fn cliRow(self: *Self, name: []const u8, st: Stats, speedup: f64) !void {
        try self.out.print("{s:<19} {d:>9.3} {d:>9.3} {d:>9.3} {d:>9.3}  {d:>6.2}x\n", .{ name, st.min, st.median, st.mean, st.max, speedup });
    }
};

// -----------------------------------------------------------------------------
// Entry point

/// `.info` compiles every function-entry trace out; `.debug` turns them on.
pub const std_options: std.Options = .{ .log_level = .info };

/// Times every component, then, given the paths of the optimized CLI and the
/// straight port as arguments, both CLIs end to end.
pub fn main(init: std.process.Init) !void {
    log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
    const io = init.io;
    var buffer: [4096]u8 = undefined;
    var writer: std.Io.File.Writer = .init(.stdout(), io, &buffer);
    const out = &writer.interface;

    var bench = Bench.init(io, out);
    defer bench.deinit();
    try bench.components();

    var args = init.minimal.args.iterate();
    _ = args.skip();
    if (args.next()) |optimized| {
        const straight = args.next() orelse return error.MissingPortCli;
        try bench.cli(optimized, straight);
    }
    try out.flush();
}
