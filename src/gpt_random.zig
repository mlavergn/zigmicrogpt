// ---------------------------------------------------------------------------
// CPython's random module, vectorized
// ---------------------------------------------------------------------------
//
// A drop-in replacement for port/gpt_random.zig with an identical public API
// and a bit-identical output stream. Every optimization here is required to be
// value-preserving; the moment one is not, the port stops matching microgpt.py
// and the speed is worthless.
//
// What is optimized, and what cannot be:
//
//   * The twist (the 624-word state refill) is SIMD. Its three index ranges
//     have no intra-vector write-then-read hazard, so whole vectors of words
//     advance at once. This is the bulk of the work.
//   * Tempering is SIMD, applied to the whole block at refill time instead of
//     one word per draw. Each word tempers independently, so the values are
//     unchanged.
//   * Seeding is a serial chain, so it is made cheap per step instead: the
//     seed-independent `initGenrand(19650218)` table is a comptime constant,
//     the key length is comptime (no divide), and the chain runs through a
//     register with the fold written out. Seeding alone runs about 2.4x the
//     plain port (2.2x in `make bench`, whose seed case includes the first
//     draw). It can also run at comptime, baking the state into the binary.
//   * The stream itself is strictly sequential and cannot be threaded: word i
//     depends on the state left by word i-1, and `gauss`/`shuffle`/`choices`
//     must consume draws in CPython's exact order. Threads would also lose to
//     their own launch overhead here — a refill is 624 words, microseconds of
//     work. SIMD is the parallelism that fits this problem.
//   * `choices` sums weights sequentially on purpose. Vectorizing a float sum
//     reassociates it, which changes the last bits and breaks parity.

const std = @import("std");
const log = std.log.scoped(.microgptzig_random);

/// CPython's `random.Random`: MT19937 plus the algorithms layered on it.
pub const Random = struct {
    const Self = @This();
    const n = 624;
    const m = 397;
    const matrix_a: u32 = 0x9908b0df;
    const upper_mask: u32 = 0x80000000;
    const lower_mask: u32 = 0x7fffffff;

    /// Widest u32 vector this target retires in one instruction. Both the
    /// twist and the tempering are written against it.
    const w = std.simd.suggestVectorLength(u32) orelse 4;
    const V = @Vector(w, u32);
    const Shift = @Vector(w, u5);

    comptime {
        // Phase 2 of the twist reads `n - m` words behind the word it writes.
        // A vector wider than that gap would have a lane reading a value its
        // own store had not yet produced.
        if (w > n - m) @compileError("vector width exceeds the twist's dependency distance");
    }

    /// Zeroed rather than `undefined` so the whole struct is comptime-copyable,
    /// which is what lets `seeded` bake a pre-seeded state into the binary.
    mt: [n]u32 align(64) = @splat(0),
    /// The tempered form of `mt`, produced a whole block at a time.
    out: [n]u32 align(64) = @splat(0),
    mti: usize = n + 1,
    /// `random.gauss` generates two normals at a time and caches the spare.
    gauss_next: ?f64 = null,

    /// Seeds the state from a single 32-bit value.
    fn initGenrand(self: *Self, s: u32) void {
        self.mt[0] = s;
        for (1..n) |i| {
            const prev = self.mt[i - 1];
            self.mt[i] = 1812433253 *% (prev ^ (prev >> 30)) +% @as(u32, @intCast(i));
        }
        self.mti = n;
    }

    /// The state `initByArray` starts from: `initGenrand(19650218)`. The
    /// argument is a constant whatever the seed, so the table is too.
    const init_state: [n]u32 = blk: {
        @setEvalBranchQuota(10_000);
        var r: Self = .{};
        r.initGenrand(19650218);
        break :blk r.mt;
    };

    // Each step depends on the word the step before it wrote, so seeding is a
    // serial chain and the work is in making each link cheap. `prev` carries
    // that word in a register rather than through memory. The fold back to
    // `mt[1]`, which CPython tests for on every step, is written out where it
    // happens. With `key_len` comptime, `% key_len` is a mask, not a divide.
    /// Seeds the state from a key of one or two words, as CPython's
    /// `init_by_array` does for integer seeds.
    fn initByArray(self: *Self, comptime key_len: usize, key: [key_len]u32) void {
        comptime std.debug.assert(key_len == 1 or key_len == 2);
        self.mti = n;
        // Pass 1 rewrites every word, so it reads the table rather than a copy:
        // n steps over mt[1..n), then mt[1] again after the fold.
        var prev = init_state[0];
        for (1..n) |i| {
            const j = (i - 1) % key_len;
            prev = (init_state[i] ^ ((prev ^ (prev >> 30)) *% 1664525)) +% key[j] +% @as(u32, @intCast(j));
            self.mt[i] = prev;
        }
        self.mt[0] = prev;
        {
            const j = (n - 1) % key_len;
            prev = (self.mt[1] ^ ((prev ^ (prev >> 30)) *% 1664525)) +% key[j] +% @as(u32, @intCast(j));
            self.mt[1] = prev;
        }
        // Pass 2: n - 1 steps over mt[2..n), then mt[1] after the fold.
        for (2..n) |i| {
            prev = (self.mt[i] ^ ((prev ^ (prev >> 30)) *% 1566083941)) -% @as(u32, @intCast(i));
            self.mt[i] = prev;
        }
        self.mt[1] = (self.mt[1] ^ ((prev ^ (prev >> 30)) *% 1566083941)) -% 1;
        self.mt[0] = 0x80000000;
    }

    // CPython feeds the absolute value to `init_by_array` as little-endian
    // 32-bit words; the key length is what distinguishes the two cases below.
    /// Seeds the generator from a non-negative integer, as `random.seed(x)`
    /// would, and discards any cached normal.
    pub fn seed(self: *Self, s: u64) void {
        // `seeded` runs this at comptime, where there is nothing to log to.
        if (!@inComptime()) log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.gauss_next = null;
        if (s >> 32 != 0) {
            self.initByArray(2, .{ @truncate(s), @truncate(s >> 32) });
        } else {
            self.initByArray(1, .{@truncate(s)});
        }
    }

    /// Returns a generator already seeded with `s`, resolved at compile time so
    /// the seeded state is a constant in the binary rather than work at startup.
    pub fn seeded(comptime s: u64) Self {
        return comptime blk: {
            @setEvalBranchQuota(100_000);
            var r: Self = .{};
            r.seed(s);
            break :blk r;
        };
    }

    /// Advances `w` consecutive words of the state at once, starting at `kk`
    /// and taking the far operand from `far_idx`.
    inline fn twistBlock(self: *Self, kk: usize, far_idx: usize) void {
        // Every lane loads before any lane stores, so the overlap between the
        // words written here and the `nxt` words read here is harmless.
        const cur: V = self.mt[kk..][0..w].*;
        const nxt: V = self.mt[kk + 1 ..][0..w].*;
        const far: V = self.mt[far_idx..][0..w].*;
        const y = (cur & @as(V, @splat(upper_mask))) | (nxt & @as(V, @splat(lower_mask)));
        // `matrix_a & -(y & 1)` is the branchless form of the odd-y test.
        const mag = @as(V, @splat(matrix_a)) & (@as(V, @splat(0)) -% (y & @as(V, @splat(1))));
        self.mt[kk..][0..w].* = far ^ (y >> @as(Shift, @splat(1))) ^ mag;
    }

    /// Advances a single word of the state.
    inline fn twistWord(self: *Self, kk: usize, far: u32) void {
        const y = (self.mt[kk] & upper_mask) | (self.mt[(kk + 1) % n] & lower_mask);
        self.mt[kk] = far ^ (y >> 1) ^ (if (y & 1 != 0) matrix_a else 0);
    }

    /// Advances the entire state to the next block of 624 words.
    fn twist(self: *Self) void {
        @setRuntimeSafety(false);
        var kk: usize = 0;
        // Phase 1: `mt[kk + 1]` and `mt[kk + m]` both still hold old values.
        while (kk + w <= n - m) : (kk += w) self.twistBlock(kk, kk + m);
        while (kk < n - m) : (kk += 1) self.twistWord(kk, self.mt[kk + m]);
        // Phase 2: the far operand was written above, `n - m` words back.
        while (kk + w <= n - 1) : (kk += w) self.twistBlock(kk, kk + m - n);
        while (kk < n - 1) : (kk += 1) self.twistWord(kk, self.mt[kk + m - n]);
        // Phase 3: the last word wraps around to `mt[0]`.
        self.twistWord(n - 1, self.mt[m - 1]);
    }

    // Words temper independently, so doing all 624 up front yields exactly what
    // tempering one word per draw would.
    /// Produces the tempered output for every word of the refilled state.
    fn temperAll(self: *Self) void {
        @setRuntimeSafety(false);
        var i: usize = 0;
        while (i + w <= n) : (i += w) {
            var y: V = self.mt[i..][0..w].*;
            y ^= y >> @as(Shift, @splat(11));
            y ^= (y << @as(Shift, @splat(7))) & @as(V, @splat(0x9d2c5680));
            y ^= (y << @as(Shift, @splat(15))) & @as(V, @splat(0xefc60000));
            y ^= y >> @as(Shift, @splat(18));
            self.out[i..][0..w].* = y;
        }
        while (i < n) : (i += 1) {
            var y = self.mt[i];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680;
            y ^= (y << 15) & 0xefc60000;
            y ^= y >> 18;
            self.out[i] = y;
        }
    }

    /// Returns the next 32-bit value of the stream, refilling the state when
    /// the current block runs dry.
    inline fn genrand(self: *Self) u32 {
        @setRuntimeSafety(false);
        if (self.mti >= n) {
            self.twist();
            self.temperAll();
            self.mti = 0;
        }
        const y = self.out[self.mti];
        self.mti += 1;
        return y;
    }

    // 53 bits of mantissa, assembled from two 32-bit draws.
    /// Returns the next value of the stream as a float in `[0, 1)`, as
    /// `random.random()` would.
    inline fn random(self: *Self) f64 {
        const a: f64 = @floatFromInt(self.genrand() >> 5);
        const b: f64 = @floatFromInt(self.genrand() >> 6);
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    /// Returns the next `k` random bits, as `random.getrandbits(k)` would.
    inline fn getrandbits(self: *Self, k: u6) u32 {
        if (k == 0) return 0;
        std.debug.assert(k <= 32);
        return self.genrand() >> @intCast(32 - @as(u32, k));
    }

    // Rejection sampling rather than a modulo, both to avoid the bias and
    // because CPython's draw-for-draw behaviour has to match.
    /// Returns a uniformly random value below `bound`, as CPython's
    /// `_randbelow` would.
    fn randbelow(self: *Self, bound: u32) u32 {
        if (bound == 0) return 0;
        const k: u6 = @intCast(32 - @clz(bound)); // bound.bit_length()
        while (true) {
            const v = self.getrandbits(k);
            if (v < bound) return v;
        }
    }

    /// Reorders `xs` into a uniformly random permutation, as `random.shuffle`
    /// would.
    pub fn shuffle(self: *Self, comptime T: type, xs: []T) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (xs.len < 2) return;
        var i: usize = xs.len - 1;
        while (i >= 1) : (i -= 1) {
            const j = self.randbelow(@intCast(i + 1));
            std.mem.swap(T, &xs[i], &xs[j]);
        }
    }

    /// Returns a normally distributed value with mean `mu` and standard
    /// deviation `sigma`, as `random.gauss(mu, sigma)` would.
    pub fn gauss(self: *Self, mu: f64, sigma: f64) f64 {
        const two_pi: f64 = 2.0 * @as(f64, std.math.pi);
        if (self.gauss_next) |z| {
            self.gauss_next = null;
            return mu + z * sigma;
        }
        const x2pi = self.random() * two_pi;
        const g2rad = @sqrt(-2.0 * @log(1.0 - self.random()));
        const z = @cos(x2pi) * g2rad;
        self.gauss_next = @sin(x2pi) * g2rad;
        return mu + z * sigma;
    }

    /// Returns an index into `weights`, chosen with probability proportional to
    /// its weight — what `random.choices(range(n), weights=w)[0]` returns.
    /// `cum` is caller-provided scratch, the same length as `weights`.
    pub fn choices(self: *Self, weights: []const f64, cum: []f64) usize {
        // Sequential on purpose: a vectorized sum reassociates the additions
        // and would drift from CPython in the last bits.
        var acc: f64 = weights[0];
        cum[0] = acc;
        for (weights[1..], 1..) |wt, i| {
            acc += wt;
            cum[i] = acc;
        }
        const total = cum[weights.len - 1] + 0.0;
        const x = self.random() * total;
        var lo: usize = 0;
        var hi: usize = weights.len - 1; // bisect_right's `hi`, per CPython
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (x < cum[mid]) hi = mid else lo = mid + 1;
        }
        return lo;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests
//
// Parity tests. The CPython 3.12 vectors are the same ones port/gpt_random.zig
// carries; the cross-checks below then pin this file to it draw for draw.

test "random.random matches CPython after seed(42)" {
    var r: Random = .{};
    r.seed(42);
    try std.testing.expectEqual(@as(f64, 0.6394267984578837), r.random());
    try std.testing.expectEqual(@as(f64, 0.025010755222666936), r.random());
    try std.testing.expectEqual(@as(f64, 0.27502931836911926), r.random());
}

test "random.getrandbits matches CPython after seed(42)" {
    var r: Random = .{};
    r.seed(42);
    const expected = [_]u32{ 20952, 3648, 819, 24299, 9012 };
    for (expected) |e| try std.testing.expectEqual(e, r.getrandbits(15));
}

test "random.gauss matches CPython after seed(42)" {
    var r: Random = .{};
    r.seed(42);
    // Six draws, so that the cached second normal is exercised three times.
    const expected = [_]f64{
        -0.011527226366234268, -0.013832288026521545, -0.008905268925412997,
        0.05615869800790905,   -0.010207062702630967, -0.1197882731472766,
    };
    for (expected) |e| try std.testing.expectEqual(e, r.gauss(0, 0.08));
}

test "random.shuffle matches CPython after seed(42)" {
    var r: Random = .{};
    r.seed(42);
    var xs: [10]u32 = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    r.shuffle(u32, &xs);
    try std.testing.expectEqual([10]u32{ 7, 3, 2, 8, 5, 6, 9, 4, 0, 1 }, xs);
}

test "random.choices matches CPython after seed(42)" {
    var r: Random = .{};
    r.seed(42);
    const weights = [_]f64{ 0.1, 0.2, 0.3, 0.25, 0.15 };
    var cum: [5]f64 = undefined;
    const expected = [_]usize{ 3, 0, 1, 1, 3, 3, 4, 0 };
    for (expected) |e| try std.testing.expectEqual(e, r.choices(&weights, &cum));
}

test "comptime seeding equals runtime seeding" {
    var baked = Random.seeded(42);
    var live: Random = .{};
    live.seed(42);
    for (0..2048) |_| try std.testing.expectEqual(live.gauss(0, 1), baked.gauss(0, 1));
}

test "seeding is identical to port/gpt_random.zig" {
    // seed(42) alone never reaches the two-word key or the edges of the fold,
    // so sweep both key lengths and the boundaries between them.
    const Ref = @import("port/gpt_random.zig").Random;
    var fast: Random = .{};
    var ref: Ref = .{};
    const edges = [_]u64{ 0, 1, 42, 19650218, 0xffff_ffff, 0x1_0000_0000, 0x1_0000_0001, std.math.maxInt(u64) };
    var x: u64 = 0x9E3779B97F4A7C15;
    for (0..edges.len + 2000) |i| {
        const s = if (i < edges.len) edges[i] else blk: {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            break :blk if (i % 2 == 0) x else x >> 32; // two-word keys, then one-word
        };
        fast.seed(s);
        ref.seed(s);
        try std.testing.expectEqualSlices(u32, &ref.mt, &fast.mt);
        for (0..4) |_| try std.testing.expectEqual(ref.gauss(0, 1), fast.gauss(0, 1));
    }
}

test "stream is identical to port/gpt_random.zig" {
    const Ref = @import("port/gpt_random.zig").Random;
    var fast: Random = .{};
    var ref: Ref = .{};
    fast.seed(42);
    ref.seed(42);
    // Well past the refill boundary, so many twists are compared, not one.
    for (0..20_000) |_| try std.testing.expectEqual(ref.gauss(0, 0.08), fast.gauss(0, 0.08));

    fast.seed(42);
    ref.seed(42);
    var a: [64]u32 = undefined;
    var b: [64]u32 = undefined;
    for (&a, 0..) |*e, i| e.* = @intCast(i);
    b = a;
    fast.shuffle(u32, &a);
    ref.shuffle(u32, &b);
    try std.testing.expectEqual(b, a);

    const weights = [_]f64{ 0.1, 0.2, 0.3, 0.25, 0.15 };
    var cum_a: [5]f64 = undefined;
    var cum_b: [5]f64 = undefined;
    for (0..1000) |_| try std.testing.expectEqual(ref.choices(&weights, &cum_b), fast.choices(&weights, &cum_a));
}
