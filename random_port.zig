// ---------------------------------------------------------------------------
// Let there be order among chaos: CPython's random module, reimplemented
// ---------------------------------------------------------------------------

const std = @import("std");

/// MT19937 plus the exact algorithms CPython layers on top of it. Without this
/// the two programs would diverge at the very first weight initialization.
pub const Random = struct {
    const n = 624;
    const m = 397;
    const matrix_a: u32 = 0x9908b0df;
    const upper_mask: u32 = 0x80000000;
    const lower_mask: u32 = 0x7fffffff;

    mt: [n]u32 = undefined,
    mti: usize = n + 1,
    /// `random.gauss` generates two normals at a time and caches the spare.
    gauss_next: ?f64 = null,

    fn initGenrand(r: *Random, s: u32) void {
        r.mt[0] = s;
        for (1..n) |i| {
            const prev = r.mt[i - 1];
            r.mt[i] = 1812433253 *% (prev ^ (prev >> 30)) +% @as(u32, @intCast(i));
        }
        r.mti = n;
    }

    /// Advance the seeding cursor, folding the state over at the end.
    fn wrap(r: *Random, i: usize) usize {
        if (i + 1 < n) return i + 1;
        r.mt[0] = r.mt[n - 1];
        return 1;
    }

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

    /// `random.seed(x)` for a non-negative integer: CPython feeds the absolute
    /// value to `init_by_array` as little-endian 32-bit words.
    pub fn seed(r: *Random, s: u64) void {
        var key: [2]u32 = .{ @truncate(s), @truncate(s >> 32) };
        const len: usize = if (s >> 32 != 0) 2 else 1;
        r.gauss_next = null;
        r.initByArray(key[0..len]);
    }

    fn genrand(r: *Random) u32 {
        if (r.mti >= n) {
            // The reference implementation splits this into three cases to keep
            // the indices in range; folding them back by hand keeps the walk in
            // one loop without paying for a division.
            for (0..n) |kk| {
                const next = if (kk + 1 < n) kk + 1 else 0;
                const far = if (kk + m < n) kk + m else kk + m - n;
                const y = (r.mt[kk] & upper_mask) | (r.mt[next] & lower_mask);
                r.mt[kk] = r.mt[far] ^ (y >> 1) ^ (if (y & 1 != 0) matrix_a else 0);
            }
            r.mti = 0;
        }
        var y = r.mt[r.mti];
        r.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        return y;
    }

    /// `random.random()`: 53 bits of mantissa out of two 32-bit draws.
    fn random(r: *Random) f64 {
        const a: f64 = @floatFromInt(r.genrand() >> 5);
        const b: f64 = @floatFromInt(r.genrand() >> 6);
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    fn getrandbits(r: *Random, k: u6) u32 {
        if (k == 0) return 0;
        std.debug.assert(k <= 32);
        return r.genrand() >> @intCast(32 - @as(u32, k));
    }

    /// `Random._randbelow_with_getrandbits`: rejection sampling, no modulo bias.
    fn randbelow(r: *Random, bound: u32) u32 {
        if (bound == 0) return 0;
        const k: u6 = @intCast(32 - @clz(bound)); // bound.bit_length()
        while (true) {
            const v = r.getrandbits(k);
            if (v < bound) return v;
        }
    }

    pub fn shuffle(r: *Random, comptime T: type, xs: []T) void {
        if (xs.len < 2) return;
        var i: usize = xs.len - 1;
        while (i >= 1) : (i -= 1) {
            const j = r.randbelow(@intCast(i + 1));
            std.mem.swap(T, &xs[i], &xs[j]);
        }
    }

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

    /// `random.choices(range(n), weights=w)[0]`: accumulate, then bisect_right.
    /// `cum` is caller-provided scratch, the same length as `weights`.
    pub fn choices(r: *Random, weights: []const f64, cum: []f64) usize {
        var acc: f64 = weights[0];
        cum[0] = acc;
        for (weights[1..], 1..) |w, i| {
            acc += w;
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
// Parity tests. Every expected value below was produced by CPython 3.12 and is
// reproduced here bit-for-bit; if one of these fails, the port has drifted off
// microgpt.py's random stream and no amount of matching arithmetic will save it.
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
