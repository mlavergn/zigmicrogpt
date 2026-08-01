// ---------------------------------------------------------------------------
// CPython's random module, vectorized
// ---------------------------------------------------------------------------
//
// A drop-in replacement for random.zig with an identical public API and a
// bit-identical output stream. Every optimization here is required to be
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
//   * Seeding can run at comptime, baking the initial state into the binary.
//   * The stream itself is strictly sequential and cannot be threaded: word i
//     depends on the state left by word i-1, and `gauss`/`shuffle`/`choices`
//     must consume draws in CPython's exact order. Threads would also lose to
//     their own launch overhead here — a refill is 624 words, microseconds of
//     work. SIMD is the parallelism that fits this problem.
//   * `choices` sums weights sequentially on purpose. Vectorizing a float sum
//     reassociates it, which changes the last bits and breaks parity.

const std = @import("std");

pub const Random = struct {
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
    fn initGenrand(r: *Random, s: u32) void {
        r.mt[0] = s;
        for (1..n) |i| {
            const prev = r.mt[i - 1];
            r.mt[i] = 1812433253 *% (prev ^ (prev >> 30)) +% @as(u32, @intCast(i));
        }
        r.mti = n;
    }

    /// Returns the next position of the seeding cursor, folding back to the
    /// start once the end of the state is reached.
    fn wrap(r: *Random, i: usize) usize {
        if (i + 1 < n) return i + 1;
        r.mt[0] = r.mt[n - 1];
        return 1;
    }

    /// Seeds the state from a key of arbitrary length. This is the seeding
    /// CPython uses for integer seeds, so it is the one that matters here.
    fn initByArray(r: *Random, key: []const u32) void {
        r.initGenrand(19650218);
        var i: usize = 1;
        var j: usize = 0;
        for (0..@max(n, key.len)) |_| {
            const prev = r.mt[i - 1];
            r.mt[i] = (r.mt[i] ^ ((prev ^ (prev >> 30)) *% 1664525)) +% key[j] +% @as(u32, @intCast(j));
            i = r.wrap(i);
            j = (j + 1) % key.len;
        }
        for (0..n - 1) |_| {
            const prev = r.mt[i - 1];
            r.mt[i] = (r.mt[i] ^ ((prev ^ (prev >> 30)) *% 1566083941)) -% @as(u32, @intCast(i));
            i = r.wrap(i);
        }
        r.mt[0] = 0x80000000;
    }

    // CPython feeds the absolute value to `init_by_array` as little-endian
    // 32-bit words; the key length is what distinguishes the two cases below.
    /// Seeds the generator from a non-negative integer, as `random.seed(x)`
    /// would, and discards any cached normal.
    pub fn seed(r: *Random, s: u64) void {
        var key: [2]u32 = .{ @truncate(s), @truncate(s >> 32) };
        const len: usize = if (s >> 32 != 0) 2 else 1;
        r.gauss_next = null;
        r.initByArray(key[0..len]);
    }

    /// Returns a generator already seeded with `s`, resolved at compile time so
    /// the seeded state is a constant in the binary rather than work at startup.
    pub fn seeded(comptime s: u64) Random {
        return comptime blk: {
            @setEvalBranchQuota(100_000);
            var r: Random = .{};
            r.seed(s);
            break :blk r;
        };
    }

    /// Advances `w` consecutive words of the state at once, starting at `kk`
    /// and taking the far operand from `far_idx`.
    inline fn twistBlock(r: *Random, kk: usize, far_idx: usize) void {
        // Every lane loads before any lane stores, so the overlap between the
        // words written here and the `nxt` words read here is harmless.
        const cur: V = r.mt[kk..][0..w].*;
        const nxt: V = r.mt[kk + 1 ..][0..w].*;
        const far: V = r.mt[far_idx..][0..w].*;
        const y = (cur & @as(V, @splat(upper_mask))) | (nxt & @as(V, @splat(lower_mask)));
        // `matrix_a & -(y & 1)` is the branchless form of the odd-y test.
        const mag = @as(V, @splat(matrix_a)) & (@as(V, @splat(0)) -% (y & @as(V, @splat(1))));
        r.mt[kk..][0..w].* = far ^ (y >> @as(Shift, @splat(1))) ^ mag;
    }

    /// Advances a single word of the state.
    inline fn twistWord(r: *Random, kk: usize, far: u32) void {
        const y = (r.mt[kk] & upper_mask) | (r.mt[(kk + 1) % n] & lower_mask);
        r.mt[kk] = far ^ (y >> 1) ^ (if (y & 1 != 0) matrix_a else 0);
    }

    /// Advances the entire state to the next block of 624 words.
    fn twist(r: *Random) void {
        @setRuntimeSafety(false);
        var kk: usize = 0;
        // Phase 1: `mt[kk + 1]` and `mt[kk + m]` both still hold old values.
        while (kk + w <= n - m) : (kk += w) r.twistBlock(kk, kk + m);
        while (kk < n - m) : (kk += 1) r.twistWord(kk, r.mt[kk + m]);
        // Phase 2: the far operand was written above, `n - m` words back.
        while (kk + w <= n - 1) : (kk += w) r.twistBlock(kk, kk + m - n);
        while (kk < n - 1) : (kk += 1) r.twistWord(kk, r.mt[kk + m - n]);
        // Phase 3: the last word wraps around to `mt[0]`.
        r.twistWord(n - 1, r.mt[m - 1]);
    }

    // Words temper independently, so doing all 624 up front yields exactly what
    // tempering one word per draw would.
    /// Produces the tempered output for every word of the refilled state.
    fn temperAll(r: *Random) void {
        @setRuntimeSafety(false);
        var i: usize = 0;
        while (i + w <= n) : (i += w) {
            var y: V = r.mt[i..][0..w].*;
            y ^= y >> @as(Shift, @splat(11));
            y ^= (y << @as(Shift, @splat(7))) & @as(V, @splat(0x9d2c5680));
            y ^= (y << @as(Shift, @splat(15))) & @as(V, @splat(0xefc60000));
            y ^= y >> @as(Shift, @splat(18));
            r.out[i..][0..w].* = y;
        }
        while (i < n) : (i += 1) {
            var y = r.mt[i];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680;
            y ^= (y << 15) & 0xefc60000;
            y ^= y >> 18;
            r.out[i] = y;
        }
    }

    /// Returns the next 32-bit value of the stream, refilling the state when
    /// the current block runs dry.
    inline fn genrand(r: *Random) u32 {
        @setRuntimeSafety(false);
        if (r.mti >= n) {
            r.twist();
            r.temperAll();
            r.mti = 0;
        }
        const y = r.out[r.mti];
        r.mti += 1;
        return y;
    }

    // 53 bits of mantissa, assembled from two 32-bit draws.
    /// Returns the next value of the stream as a float in `[0, 1)`, as
    /// `random.random()` would.
    inline fn random(r: *Random) f64 {
        const a: f64 = @floatFromInt(r.genrand() >> 5);
        const b: f64 = @floatFromInt(r.genrand() >> 6);
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    /// Returns the next `k` random bits, as `random.getrandbits(k)` would.
    inline fn getrandbits(r: *Random, k: u6) u32 {
        if (k == 0) return 0;
        std.debug.assert(k <= 32);
        return r.genrand() >> @intCast(32 - @as(u32, k));
    }

    // Rejection sampling rather than a modulo, both to avoid the bias and
    // because CPython's draw-for-draw behaviour has to match.
    /// Returns a uniformly random value below `bound`, as CPython's
    /// `_randbelow` would.
    fn randbelow(r: *Random, bound: u32) u32 {
        if (bound == 0) return 0;
        const k: u6 = @intCast(32 - @clz(bound)); // bound.bit_length()
        while (true) {
            const v = r.getrandbits(k);
            if (v < bound) return v;
        }
    }

    /// Reorders `xs` into a uniformly random permutation, as `random.shuffle`
    /// would.
    pub fn shuffle(r: *Random, comptime T: type, xs: []T) void {
        if (xs.len < 2) return;
        var i: usize = xs.len - 1;
        while (i >= 1) : (i -= 1) {
            const j = r.randbelow(@intCast(i + 1));
            std.mem.swap(T, &xs[i], &xs[j]);
        }
    }

    /// Returns a normally distributed value with mean `mu` and standard
    /// deviation `sigma`, as `random.gauss(mu, sigma)` would.
    pub fn gauss(r: *Random, mu: f64, sigma: f64) f64 {
        const two_pi: f64 = 2.0 * @as(f64, std.math.pi);
        if (r.gauss_next) |z| {
            r.gauss_next = null;
            return mu + z * sigma;
        }
        const x2pi = r.random() * two_pi;
        const g2rad = @sqrt(-2.0 * @log(1.0 - r.random()));
        const z = @cos(x2pi) * g2rad;
        r.gauss_next = @sin(x2pi) * g2rad;
        return mu + z * sigma;
    }

    /// Returns an index into `weights`, chosen with probability proportional to
    /// its weight — what `random.choices(range(n), weights=w)[0]` returns.
    /// `cum` is caller-provided scratch, the same length as `weights`.
    pub fn choices(r: *Random, weights: []const f64, cum: []f64) usize {
        // Sequential on purpose: a vectorized sum reassociates the additions
        // and would drift from CPython in the last bits.
        var acc: f64 = weights[0];
        cum[0] = acc;
        for (weights[1..], 1..) |wt, i| {
            acc += wt;
            cum[i] = acc;
        }
        const total = cum[weights.len - 1] + 0.0;
        const x = r.random() * total;
        var lo: usize = 0;
        var hi: usize = weights.len - 1; // bisect_right's `hi`, per CPython
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (x < cum[mid]) hi = mid else lo = mid + 1;
        }
        return lo;
    }
};

// ---------------------------------------------------------------------------
// Parity tests. The CPython 3.12 vectors are the same ones random.zig carries;
// the cross-check below then pins this file to random.zig draw for draw.
// ---------------------------------------------------------------------------

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

test "stream is identical to random.zig" {
    const Ref = @import("random_port.zig").Random;
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
