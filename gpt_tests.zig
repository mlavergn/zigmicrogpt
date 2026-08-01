// ---------------------------------------------------------------------------
// Parity tests. Every expected value below was produced by CPython 3.12 and is
// reproduced here bit-for-bit; if one of these fails, the port has drifted off
// microgpt.py's random stream and no amount of matching arithmetic will save it.
// ---------------------------------------------------------------------------

const std = @import("std");
const microgpt = @import("gpt_main.zig");
const Tape = microgpt.Tape;
const Value = microgpt.Value;

// The random module's own parity tests live beside the code they cover; this
// pulls them into the same test binary.
test {
    _ = @import("gpt_random.zig");
}

test "powers match CPython's float.__pow__" {
    // The reason this file does not just call std.math.pow: std.math.pow(f64, 2, -0.5)
    // returns 0.7071067811865475, one ulp below what CPython prints.
    try std.testing.expectEqual(@as(f64, 0.7071067811865476), std.math.pow(f64, 2.0, -0.5));
    try std.testing.expectEqual(@as(f64, 0.6141249999999999), std.math.pow(f64, 0.85, 3.0));

    // Every exponent shape the program actually reaches, against CPython 3.12.
    try std.testing.expectEqual(@as(f64, 1.0), std.math.pow(f64, 7.3, 0.0)); // n == 0
    try std.testing.expectEqual(@as(f64, 0.136986301369863), std.math.pow(f64, 7.3, -1.0));
    try std.testing.expectEqual(@as(f64, 0.018765246762994934), std.math.pow(f64, 7.3, -2.0));
    try std.testing.expectEqual(@as(f64, 0.3701166050988026), std.math.pow(f64, 7.3, -0.5));
    try std.testing.expectEqual(@as(f64, 0.050700904808055156), std.math.pow(f64, 7.3, -1.5));
    try std.testing.expectEqual(@as(f64, 2.701851217221259), std.math.pow(f64, 7.3, 0.5));
    try std.testing.expectEqual(@as(f64, 53.29), std.math.pow(f64, 7.3, 2.0));
    try std.testing.expectEqual(@as(f64, 0.0625), std.math.pow(f64, -0.25, 2.0)); // negative base
    try std.testing.expectEqual(@as(f64, 0.0), std.math.pow(f64, 0.0, 2.0));
    // Adam's bias correction, at the first and last step of a 1000-step run.
    try std.testing.expectEqual(@as(f64, 0.9), std.math.pow(f64, 0.9, 1.0));
    try std.testing.expectEqual(@as(f64, 1.7478712517226947e-46), std.math.pow(f64, 0.9, 1000.0));
    try std.testing.expectEqual(@as(f64, 4.317124741065786e-5), std.math.pow(f64, 0.99, 1000.0));
}

test "autograd reproduces the chain rule" {
    var tape: Tape = .{ .gpa = std.testing.allocator };
    defer tape.deinit();
    // f(a, b) = (a * b + a).relu(), at a = 3, b = -2 -> relu(-3) = 0, grads 0
    // f(a, b) = (a * b + a).relu(), at a = 3, b = 2  -> relu(9) = 9
    const a = try tape.leaf(3.0);
    const b = try tape.leaf(2.0);
    const f = try tape.relu(try tape.add(try tape.mul(a, b), a));
    try tape.backward(f);
    try std.testing.expectEqual(@as(f64, 9.0), tape.data(f));
    try std.testing.expectEqual(@as(f64, 3.0), tape.nodes.items[a].grad); // b + 1
    try std.testing.expectEqual(@as(f64, 3.0), tape.nodes.items[b].grad); // a
}

test "softmax sums to one and rmsnorm normalizes" {
    var tape: Tape = .{ .gpa = std.testing.allocator };
    defer tape.deinit();
    var xs: [4]Value = undefined;
    for (&xs, [_]f64{ 1.0, 2.0, 3.0, 4.0 }) |*x, d| x.* = try tape.leaf(d);

    var probs: [4]Value = undefined;
    try microgpt.softmax(&tape, &probs, &xs);
    var total: f64 = 0;
    for (probs) |p| total += tape.data(p);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), total, 1e-12);

    try microgpt.rmsnorm(&tape, &xs);
    var ms: f64 = 0;
    for (xs) |x| ms += tape.data(x) * tape.data(x);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), ms / xs.len, 1e-5);
}
