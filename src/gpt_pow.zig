// ---------------------------------------------------------------------------
// Let there be powers, rounded the way CPython rounds them
// ---------------------------------------------------------------------------

const std = @import("std");
const log = std.log.scoped(.microgptzig_pow);

/// The implementations `Pow.pow` can be built on.
const Impl = enum {
    /// Correctly rounded, which is what CPython's `float.__pow__` aims for.
    parity,
    /// Zig's `std.math.pow`. Up to hundreds of ulps off, so the weights drift
    /// from microgpt.py's in the low bits; here to measure what parity costs.
    std,
};

/// A double-double: the unevaluated sum `hi + lo`, with `lo` under half an ulp
/// of `hi`, carrying about 106 bits of significand.
const Dd = struct {
    const Self = @This();
    hi: f64,
    lo: f64,

    /// Returns `a + b` exactly, as a normalized pair. Requires `|a| >= |b|`.
    fn fastTwoSum(a: f64, b: f64) Self {
        const s = a + b;
        return .{ .hi = s, .lo = b - (s - a) };
    }

    /// Returns the square root of `x`, refined by one Newton step whose
    /// residual `x - s*s` the fused multiply-add computes exactly.
    fn sqrt(x: f64) Self {
        const s = @sqrt(x);
        return fastTwoSum(s, @mulAdd(f64, -s, s, x) / (2 * s));
    }

    /// Returns `self * other`. The fused multiply-add recovers the rounding
    /// error of `self.hi * other.hi` exactly.
    fn mul(self: *const Self, other: Self) Self {
        const p = self.hi * other.hi;
        const e = @mulAdd(f64, self.hi, other.hi, -p) + (self.hi * other.lo + self.lo * other.hi);
        return fastTwoSum(p, e);
    }

    /// Returns `1 / self`, refined by one Newton step from `1 / self.hi`.
    fn recip(self: *const Self) Self {
        const q = 1 / self.hi;
        // r = 1 - q*self. `q * self.hi` is near 1, so `1 - p` is exact.
        const p = q * self.hi;
        const r = (1 - p) - @mulAdd(f64, q, self.hi, -p) - q * self.lo;
        return fastTwoSum(q, q * r);
    }
};

/// The only power function the port uses, in place of `std.math.pow`.
pub const Pow = struct {
    const Self = @This();

    /// Which implementation `pow` uses. Comptime, so the other one is not even
    /// compiled and the switch adds nothing to a benchmark.
    pub const impl: Impl = .parity;

    /// Returns `x` raised to `y`, the port's stand-in for Python's `x ** y`.
    pub fn pow(x: f64, y: f64) f64 {
        return switch (impl) {
            .parity => parity(x, y),
            .std => std.math.pow(f64, x, y),
        };
    }

    // parity: CPython hands `x ** y` to the C library's pow, which is accurate
    // to within an ulp; macOS's still misrounds about 0.13% of inputs, even
    // pow(x, 2) against x * x. std.math.pow is far worse: it computes x**-0.5
    // as 1 / @sqrt(x), rounding twice (2**-0.5 lands one ulp low), and integer
    // powers by repeated squaring in plain f64, whose error doubles with every
    // squaring (0.9**1000 is 209 ulps off). This does the same squaring in
    // double-double, about 106 bits, and rounds once at the end. That only
    // works for the exponents the program uses, multiples of one half, which
    // is all it needs to.
    /// Returns `x` raised to `y`, correctly rounded, for `y` a multiple of 0.5.
    pub fn parity(x: f64, y: f64) f64 {
        // A single IEEE operation is already correctly rounded.
        if (y == 0) return 1;
        if (y == 1) return x;
        if (y == 2) return x * x;
        if (y == 0.5) return @sqrt(x);
        if (y == -1) return 1 / x;

        // x**(k + 1/2) is sqrt(x)**(2k + 1), so every case is an integer power.
        const twice = 2 * y;
        std.debug.assert(twice == @trunc(twice));
        const odd = @mod(twice, 2) != 0;
        var base: Dd = if (odd) .sqrt(x) else .{ .hi = x, .lo = 0 };
        var n: u64 = @intFromFloat(@abs(if (odd) twice else y));

        // Square-and-multiply, starting the product at the lowest set bit
        // rather than at 1, so no step is a multiplication by one. `n` is at
        // least 1.
        while (n & 1 == 0) : (n >>= 1) base = base.mul(base);
        var acc = base;
        n >>= 1;
        while (n != 0) : (n >>= 1) {
            base = base.mul(base);
            if (n & 1 == 1) acc = acc.mul(base);
        }
        if (y < 0) acc = acc.recip();

        const r = acc.hi + acc.lo;
        // Outside the normal range a double-double loses its extra bits, or
        // overflows into nan. The program never gets here; std handles the
        // edges.
        if (!std.math.isNormal(r)) {
            @branchHint(.unlikely);
            log.debug("{s}:{d} :: {s} :: {d}**{d} is outside the normal range, deferring to std", .{ @src().file, @src().line, @src().fn_name, x, y });
            return std.math.pow(f64, x, y);
        }
        return r;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "powers match CPython's float.__pow__" {
    // The reason this file does not just call std.math.pow: std.math.pow(f64, 2, -0.5)
    // returns 0.7071067811865475, one ulp below what CPython prints.
    try std.testing.expectEqual(@as(f64, 0.7071067811865476), Pow.pow(2.0, -0.5));
    try std.testing.expectEqual(@as(f64, 0.6141249999999999), Pow.pow(0.85, 3.0));

    // Every exponent shape the program actually reaches, against CPython 3.12.
    try std.testing.expectEqual(@as(f64, 1.0), Pow.pow(7.3, 0.0)); // n == 0
    try std.testing.expectEqual(@as(f64, 0.136986301369863), Pow.pow(7.3, -1.0));
    try std.testing.expectEqual(@as(f64, 0.018765246762994934), Pow.pow(7.3, -2.0));
    try std.testing.expectEqual(@as(f64, 0.3701166050988026), Pow.pow(7.3, -0.5));
    try std.testing.expectEqual(@as(f64, 0.050700904808055156), Pow.pow(7.3, -1.5));
    try std.testing.expectEqual(@as(f64, 2.701851217221259), Pow.pow(7.3, 0.5));
    try std.testing.expectEqual(@as(f64, 53.29), Pow.pow(7.3, 2.0));
    try std.testing.expectEqual(@as(f64, 0.0625), Pow.pow(-0.25, 2.0)); // negative base
    try std.testing.expectEqual(@as(f64, 0.0), Pow.pow(0.0, 2.0));
    // Adam's bias correction, at the first and last step of a 1000-step run.
    try std.testing.expectEqual(@as(f64, 0.9), Pow.pow(0.9, 1.0));
    try std.testing.expectEqual(@as(f64, 1.7478712517226947e-46), Pow.pow(0.9, 1000.0));
    try std.testing.expectEqual(@as(f64, 4.317124741065786e-5), Pow.pow(0.99, 1000.0));
}
