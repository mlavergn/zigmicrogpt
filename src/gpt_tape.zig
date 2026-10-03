// ---------------------------------------------------------------------------
// Autograd: recursively apply the chain rule
// ---------------------------------------------------------------------------

const std = @import("std");
const log = std.log.scoped(.microgptzig_tape);
const mod = @import("module.zig");

/// Marks an absent operand, so a node needs no separate child count.
const no_child: u32 = std.math.maxInt(u32);

/// A node in the computation graph, identified by its index on the `Tape`.
/// Exactly 40 bytes, and the hot arrays are walked in full twice per step.
const Node = struct {
    const Self = @This();
    /// scalar value of this node calculated during the forward pass
    data: f64,
    /// derivative of the loss w.r.t. this node, calculated in the backward pass
    grad: f64,
    /// local derivative of this node w.r.t. each of its children
    local_grads: [2]f64,
    /// children of this node in the computation graph, `no_child` where absent
    children: [2]u32,
};

/// One entry of the explicit DFS stack: a node whose first child is being
/// walked, and the second child still to visit after it (`no_child` if none).
const Frame = struct {
    const Self = @This();
    node: u32,
    pending: u32,
};

/// The computation graph: every node built during a forward pass, and the
/// gradients that flow back through them. Nodes are referred to by index,
/// so the graph owns its own storage and nothing points into it.
pub const Tape = struct {
    const Self = @This();
    allocator: std.mem.Allocator,
    nodes: std.ArrayList(Node) = .empty,
    /// Scratch for `backward`, all sized once by `reserve`.
    visited: []u64 = &.{},
    topo: []u32 = &.{},
    stack: []Frame = &.{},
    /// Parameters occupy nodes `0..n_params`; everything above is per-step.
    n_params: usize = 0,

    /// Creates an empty tape that allocates through `allocator`.
    pub fn init(allocator: std.mem.Allocator) Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        return .{ .allocator = allocator };
    }

    /// Releases every allocation owned by the tape, in reverse order of
    /// `reserve`.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.allocator.free(self.stack);
        self.allocator.free(self.topo);
        self.allocator.free(self.visited);
        self.nodes.deinit(self.allocator);
    }

    /// Returns the forward-pass value of a node.
    pub fn data(self: *const Self, v: u32) f64 {
        return self.nodes.items[v].data;
    }

    /// Discard the graph built during a step, keeping the parameters.
    pub fn rewind(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.nodes.shrinkRetainingCapacity(self.n_params);
    }

    /// Reserves room for the parameters plus `extra` nodes. `rewind` only ever
    /// shrinks the length, so one call before training covers every step and
    /// the hot path never reaches the allocator again.
    pub fn reserve(self: *Self, extra: usize) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const cap = self.n_params + extra;
        try self.nodes.ensureTotalCapacity(self.allocator, cap);
        self.visited = try self.allocator.alloc(u64, (cap + 63) / 64);
        self.topo = try self.allocator.alloc(u32, cap);
        self.stack = try self.allocator.alloc(Frame, cap);
    }

    /// Records a new node on the tape and returns the index that identifies it.
    /// Infallible by construction: `reserve` has already secured the capacity.
    fn push(self: *Self, d: f64, ch: [2]u32, lg: [2]f64) u32 {
        std.debug.assert(self.nodes.items.len < self.nodes.capacity);
        const idx: u32 = @intCast(self.nodes.items.len);
        self.nodes.appendAssumeCapacity(.{
            .data = d,
            .grad = 0,
            .local_grads = lg,
            .children = ch,
        });
        return idx;
    }

    /// Creates an input node: a value the graph depends on but never derives,
    /// such as a model parameter.
    pub fn leaf(self: *Self, d: f64) !u32 {
        const idx: u32 = @intCast(self.nodes.items.len);
        // The only fallible tape op: parameters are created during setup, before
        // `reserve` has run.
        try self.nodes.append(self.allocator, .{
            .data = d,
            .grad = 0,
            .local_grads = .{ 0, 0 },
            .children = .{ no_child, no_child },
        });
        return idx;
    }

    /// Creates the node `a + b`.
    pub fn add(self: *Self, a: u32, b: u32) u32 {
        return self.push(self.data(a) + self.data(b), .{ a, b }, .{ 1, 1 });
    }

    /// Creates the node `a * b`.
    pub fn mul(self: *Self, a: u32, b: u32) u32 {
        return self.push(self.data(a) * self.data(b), .{ a, b }, .{ self.data(b), self.data(a) });
    }

    // parity: where an operand is a bare number, Python wraps it in a childless
    // `u32` and builds a two-child node. That wrapper's gradient is never
    // read, so a one-child node carries identical data and identical gradients.
    /// Creates the node `a + k`, for a constant `k`.
    pub fn addK(self: *Self, a: u32, k: f64) u32 {
        return self.push(self.data(a) + k, .{ a, no_child }, .{ 1, 0 });
    }

    /// Creates the node `a * k`, for a constant `k`.
    pub fn mulK(self: *Self, a: u32, k: f64) u32 {
        return self.push(self.data(a) * k, .{ a, no_child }, .{ k, 0 });
    }

    /// Creates the node `a` raised to the constant power `k`.
    pub fn powK(self: *Self, a: u32, k: f64) u32 {
        const d = self.data(a);
        return self.push(mod.Pow.pow(d, k), .{ a, no_child }, .{ k * mod.Pow.pow(d, k - 1), 0 });
    }

    /// Creates the node holding the natural logarithm of `a`.
    pub fn logv(self: *Self, a: u32) u32 {
        const d = self.data(a);
        return self.push(@log(d), .{ a, no_child }, .{ 1 / d, 0 });
    }

    /// Creates the node holding `e` raised to the power of `a`.
    pub fn expv(self: *Self, a: u32) u32 {
        const d = self.data(a);
        return self.push(@exp(d), .{ a, no_child }, .{ @exp(d), 0 });
    }

    /// Creates the node holding `a` with its negative range flattened to zero,
    /// the network's nonlinearity.
    pub fn relu(self: *Self, a: u32) u32 {
        const d = self.data(a);
        return self.push(@max(0, d), .{ a, no_child }, .{ if (d > 0) @as(f64, 1) else 0, 0 });
    }

    // parity: Python's builtin `sum` seeds its accumulator with the integer 0,
    // so the first term of every sum is really `term + 0`. That node is part of
    // the graph; dropping it would shift the gradient accumulation order.
    /// Creates the accumulator a running sum over nodes starts from.
    pub fn sumInit(self: *Self, first: u32) u32 {
        return self.addK(first, 0.0);
    }

    /// Whether `backward` must walk `c`: a real child, and not a parameter.
    /// Parameters are leaves, so they have nothing to pass gradient on to.
    inline fn interior(self: *const Self, c: u32) bool {
        return c != no_child and c >= self.n_params;
    }

    // The topological sort is an explicit-stack transcription of Python's
    // recursive `build_topo`, and yields the identical post-order. That order is
    // load-bearing: `+=` on floats is not associative, so visiting a node's
    // parents in a different sequence shifts the gradient in the last bits.
    //
    // A plain reverse scan of the tape is a *valid* reverse-topological order —
    // children are always pushed before their parent — but it is not *this*
    // order, and it measurably diverges by step 8. The sort stays.
    //
    // What is safe to optimize is the bookkeeping: the scratch buffers are
    // preallocated and `visited` is a bitset, so no allocator call or large
    // memset happens per step. The walk also skips parameters, which are
    // leaves and so cannot affect the order of anything else, and reads each
    // node's children once, carrying the second in its stack frame. Together
    // that more than halves the sort, which had been half of a training run.
    /// Fills in the derivative of `root` with respect to every node it was
    /// built from, leaving each result in that node's `grad`.
    pub fn backward(self: *Self, root: u32) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const nodes = self.nodes.items;
        @memset(self.visited[0 .. (nodes.len + 63) / 64], 0);
        var topo_len: usize = 0;
        var sp: usize = 0;

        var cur = root;
        walk: while (true) {
            // Descend: mark `cur`, then take its first unvisited child, as the
            // recursion would, keeping the second for when that one returns.
            self.setVisited(cur);
            const ch = nodes[cur].children;
            if (self.interior(ch[0]) and !self.isVisited(ch[0])) {
                self.stack[sp] = .{ .node = cur, .pending = ch[1] };
                sp += 1;
                cur = ch[0];
                continue;
            }
            if (self.interior(ch[1]) and !self.isVisited(ch[1])) {
                self.stack[sp] = .{ .node = cur, .pending = no_child };
                sp += 1;
                cur = ch[1];
                continue;
            }
            self.topo[topo_len] = cur;
            topo_len += 1;
            // Ascend: each parent either still has its second child to visit,
            // or is finished and joins the order.
            while (sp > 0) {
                sp -= 1;
                const f = self.stack[sp];
                if (self.interior(f.pending) and !self.isVisited(f.pending)) {
                    self.stack[sp] = .{ .node = f.node, .pending = no_child };
                    sp += 1;
                    cur = f.pending;
                    continue :walk;
                }
                self.topo[topo_len] = f.node;
                topo_len += 1;
            }
            break;
        }

        nodes[root].grad = 1;
        var i = topo_len;
        while (i > 0) {
            i -= 1;
            const node = nodes[self.topo[i]];
            if (node.children[0] == no_child) continue;
            nodes[node.children[0]].grad += node.local_grads[0] * node.grad;
            if (node.children[1] == no_child) continue;
            nodes[node.children[1]].grad += node.local_grads[1] * node.grad;
        }
    }

    inline fn isVisited(self: *const Self, i: u32) bool {
        return self.visited[i >> 6] & (@as(u64, 1) << @intCast(i & 63)) != 0;
    }

    inline fn setVisited(self: *Self, i: u32) void {
        self.visited[i >> 6] |= @as(u64, 1) << @intCast(i & 63);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "autograd reproduces the chain rule" {
    var tape = Tape.init(std.testing.allocator);
    defer tape.deinit();
    try tape.reserve(16);
    // f(a, b) = (a * b + a).relu(), at a = 3, b = -2 -> relu(-3) = 0, grads 0
    // f(a, b) = (a * b + a).relu(), at a = 3, b = 2  -> relu(9) = 9
    const a = try tape.leaf(3.0);
    const b = try tape.leaf(2.0);
    const f = tape.relu(tape.add(tape.mul(a, b), a));
    tape.backward(f);
    try std.testing.expectEqual(@as(f64, 9.0), tape.data(f));
    try std.testing.expectEqual(@as(f64, 3.0), tape.nodes.items[a].grad); // b + 1
    try std.testing.expectEqual(@as(f64, 3.0), tape.nodes.items[b].grad); // a
}

/// microgpt.py's `build_topo`, transcribed as literally as Zig allows: the
/// reference `backward`'s walk must reproduce, order and all.
fn buildTopo(nodes: anytype, visited: []bool, topo: *std.ArrayList(u32), v: u32) !void {
    if (visited[v]) return;
    visited[v] = true;
    for (nodes[v].children) |child| {
        if (child != std.math.maxInt(u32)) try buildTopo(nodes, visited, topo, child);
    }
    try topo.append(std.testing.allocator, v);
}

test "backward sums gradients in build_topo's order" {
    // A random graph with shared nodes and wide fan-in, so the order in which
    // a node's gradient is summed shows up in its last bits.
    const allocator = std.testing.allocator;
    var tape = Tape.init(allocator);
    defer tape.deinit();
    const n_params = 16;
    const n_ops = 800;
    var x: u64 = 0x9E3779B97F4A7C15;
    for (0..n_params) |_| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        _ = try tape.leaf(@as(f64, @floatFromInt(x >> 11)) * 0x1p-53 * 2 - 1);
    }
    tape.n_params = n_params;
    try tape.reserve(2 * n_ops + 1);
    for (0..n_ops) |_| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        const len: u32 = @intCast(tape.nodes.items.len);
        // One operand from the last few nodes, so chains grow deep; the other
        // from anywhere, so nodes are shared far apart.
        const a: u32 = len - 1 - @as(u32, @intCast(x % @min(len, 8)));
        const b: u32 = @intCast((x >> 16) % len);
        _ = switch ((x >> 40) % 5) {
            0, 1 => tape.add(a, b),
            2 => tape.mul(a, b),
            3 => tape.mulK(a, 0.5),
            else => tape.relu(a),
        };
    }
    // The root sums every operation, so everything is reachable.
    var root = tape.sumInit(n_params);
    for (n_params + 1..n_params + n_ops) |i| root = tape.add(root, @intCast(i));

    tape.backward(root);
    const nodes = tape.nodes.items;
    const got = try allocator.alloc(f64, nodes.len);
    defer allocator.free(got);
    for (nodes, got) |n, *g| g.* = n.grad;

    // The same gradients, from the reference walk and microgpt.py's loop.
    for (nodes) |*n| n.grad = 0;
    const visited = try allocator.alloc(bool, nodes.len);
    defer allocator.free(visited);
    @memset(visited, false);
    var topo: std.ArrayList(u32) = .empty;
    defer topo.deinit(allocator);
    try buildTopo(nodes, visited, &topo, root);
    nodes[root].grad = 1;
    var i = topo.items.len;
    while (i > 0) {
        i -= 1;
        const v = nodes[topo.items[i]];
        for (v.children, v.local_grads) |child, local_grad| {
            if (child != std.math.maxInt(u32)) nodes[child].grad += local_grad * v.grad;
        }
    }
    for (nodes, got) |n, g| try std.testing.expectEqual(@as(u64, @bitCast(n.grad)), @as(u64, @bitCast(g)));
}
