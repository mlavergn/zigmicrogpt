// The most atomic way to train and run inference for a GPT in pure, dependency-free Zig.
// This file is the complete algorithm.
// Everything else is just efficiency.
// A Zig port of microgpt.py from @karpathy.
// @mlavergn

const std = @import("std");
const Random = @import("gpt_random.zig").Random;
const train = @import("gpt_train.zig");
const inference = @import("gpt_inference.zig");
const Tokenizer = @import("gpt_tokenizer.zig").Tokenizer;

// ---------------------------------------------------------------------------
// Autograd: recursively apply the chain rule
// ---------------------------------------------------------------------------

/// Marks an absent operand, so a node needs no separate child count.
const no_child: u32 = std.math.maxInt(u32);

/// A node in the computation graph, identified by its index on the `Tape`.
/// Exactly 40 bytes, and the hot arrays are walked in full twice per step.
const Node = struct {
    /// scalar value of this node calculated during the forward pass
    data: f64,
    /// derivative of the loss w.r.t. this node, calculated in the backward pass
    grad: f64,
    /// local derivative of this node w.r.t. each of its children
    local_grads: [2]f64,
    /// children of this node in the computation graph, `no_child` where absent
    children: [2]u32,
};

/// How many operands a node actually has.
inline fn childCount(node: Node) u8 {
    if (node.children[0] == no_child) return 0;
    if (node.children[1] == no_child) return 1;
    return 2;
}

/// One entry of the explicit DFS stack.
const Frame = struct { node: u32, next_child: u8 };

/// The computation graph: every node built during a forward pass, and the
/// gradients that flow back through them. Nodes are referred to by index,
/// so the graph owns its own storage and nothing points into it.
pub const Value = struct {
    gpa: std.mem.Allocator,
    nodes: std.ArrayList(Node) = .empty,
    /// Scratch for `backward`, all sized once by `reserve`.
    visited: []u64 = &.{},
    topo: []u32 = &.{},
    stack: []Frame = &.{},
    /// Parameters occupy nodes `0..n_params`; everything above is per-step.
    n_params: usize = 0,

    /// Releases every allocation owned by the tape.
    pub fn deinit(t: *Value) void {
        t.nodes.deinit(t.gpa);
        t.gpa.free(t.visited);
        t.gpa.free(t.topo);
        t.gpa.free(t.stack);
    }

    /// Returns the forward-pass value of a node.
    pub fn data(t: *const Value, v: u32) f64 {
        return t.nodes.items[v].data;
    }

    /// Discard the graph built during a step, keeping the parameters.
    pub fn rewind(t: *Value) void {
        t.nodes.shrinkRetainingCapacity(t.n_params);
    }

    /// Reserves room for the parameters plus `extra` nodes. `rewind` only ever
    /// shrinks the length, so one call before training covers every step and
    /// the hot path never reaches the allocator again.
    fn reserve(t: *Value, extra: usize) !void {
        const cap = t.n_params + extra;
        try t.nodes.ensureTotalCapacity(t.gpa, cap);
        t.visited = try t.gpa.alloc(u64, (cap + 63) / 64);
        t.topo = try t.gpa.alloc(u32, cap);
        t.stack = try t.gpa.alloc(Frame, cap);
    }

    /// Records a new node on the tape and returns the index that identifies it.
    /// Infallible by construction: `reserve` has already secured the capacity.
    fn push(t: *Value, d: f64, ch: [2]u32, lg: [2]f64) u32 {
        std.debug.assert(t.nodes.items.len < t.nodes.capacity);
        const idx: u32 = @intCast(t.nodes.items.len);
        t.nodes.appendAssumeCapacity(.{
            .data = d,
            .grad = 0,
            .local_grads = lg,
            .children = ch,
        });
        return idx;
    }

    /// Creates an input node: a value the graph depends on but never derives,
    /// such as a model parameter.
    pub fn leaf(t: *Value, d: f64) !u32 {
        const idx: u32 = @intCast(t.nodes.items.len);
        // The only fallible tape op: parameters are created during setup, before
        // `reserve` has run.
        try t.nodes.append(t.gpa, .{
            .data = d,
            .grad = 0,
            .local_grads = .{ 0, 0 },
            .children = .{ no_child, no_child },
        });
        return idx;
    }

    /// Creates the node `a + b`.
    pub fn add(t: *Value, a: u32, b: u32) u32 {
        return t.push(t.data(a) + t.data(b), .{ a, b }, .{ 1, 1 });
    }

    /// Creates the node `a * b`.
    pub fn mul(t: *Value, a: u32, b: u32) u32 {
        return t.push(t.data(a) * t.data(b), .{ a, b }, .{ t.data(b), t.data(a) });
    }

    // parity: where an operand is a bare number, Python wraps it in a childless
    // `u32` and builds a two-child node. That wrapper's gradient is never
    // read, so a one-child node carries identical data and identical gradients.
    /// Creates the node `a + k`, for a constant `k`.
    fn addK(t: *Value, a: u32, k: f64) u32 {
        return t.push(t.data(a) + k, .{ a, no_child }, .{ 1, 0 });
    }

    /// Creates the node `a * k`, for a constant `k`.
    pub fn mulK(t: *Value, a: u32, k: f64) u32 {
        return t.push(t.data(a) * k, .{ a, no_child }, .{ k, 0 });
    }

    /// Creates the node `a` raised to the constant power `k`.
    fn powK(t: *Value, a: u32, k: f64) u32 {
        const d = t.data(a);
        return t.push(std.math.pow(f64, d, k), .{ a, no_child }, .{ k * std.math.pow(f64, d, k - 1), 0 });
    }

    /// Creates the node holding the natural logarithm of `a`.
    pub fn logv(t: *Value, a: u32) u32 {
        const d = t.data(a);
        return t.push(@log(d), .{ a, no_child }, .{ 1 / d, 0 });
    }

    /// Creates the node holding `e` raised to the power of `a`.
    fn expv(t: *Value, a: u32) u32 {
        const d = t.data(a);
        return t.push(@exp(d), .{ a, no_child }, .{ @exp(d), 0 });
    }

    /// Creates the node holding `a` with its negative range flattened to zero,
    /// the network's nonlinearity.
    pub fn relu(t: *Value, a: u32) u32 {
        const d = t.data(a);
        return t.push(@max(0, d), .{ a, no_child }, .{ if (d > 0) @as(f64, 1) else 0, 0 });
    }

    // parity: Python's builtin `sum` seeds its accumulator with the integer 0,
    // so the first term of every sum is really `term + 0`. That node is part of
    // the graph; dropping it would shift the gradient accumulation order.
    /// Creates the accumulator a running sum over nodes starts from.
    pub fn sumInit(t: *Value, first: u32) u32 {
        return t.addK(first, 0.0);
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
    // memset happens per step.
    /// Fills in the derivative of `root` with respect to every node it was
    /// built from, leaving each result in that node's `grad`.
    pub fn backward(t: *Value, root: u32) void {
        const nodes = t.nodes.items;
        @memset(t.visited[0 .. (nodes.len + 63) / 64], 0);
        var topo_len: usize = 0;
        var sp: usize = 0;

        t.setVisited(root);
        t.stack[sp] = .{ .node = root, .next_child = 0 };
        sp += 1;
        while (sp > 0) {
            const top = sp - 1;
            const node_idx = t.stack[top].node;
            const next_child = t.stack[top].next_child;
            const node = nodes[node_idx];
            if (next_child < childCount(node)) {
                t.stack[top].next_child = next_child + 1;
                const child = node.children[next_child];
                if (!t.isVisited(child)) {
                    t.setVisited(child);
                    t.stack[sp] = .{ .node = child, .next_child = 0 };
                    sp += 1;
                }
            } else {
                sp -= 1;
                t.topo[topo_len] = node_idx;
                topo_len += 1;
            }
        }

        nodes[root].grad = 1;
        var i = topo_len;
        while (i > 0) {
            i -= 1;
            const node = nodes[t.topo[i]];
            if (node.children[0] == no_child) continue;
            nodes[node.children[0]].grad += node.local_grads[0] * node.grad;
            if (node.children[1] == no_child) continue;
            nodes[node.children[1]].grad += node.local_grads[1] * node.grad;
        }
    }

    inline fn isVisited(t: *const Value, i: u32) bool {
        return t.visited[i >> 6] & (@as(u64, 1) << @intCast(i & 63)) != 0;
    }

    inline fn setVisited(t: *Value, i: u32) void {
        t.visited[i >> 6] |= @as(u64, 1) << @intCast(i & 63);
    }
};

// ---------------------------------------------------------------------------
// The parameters, to store the knowledge of the model
// ---------------------------------------------------------------------------

pub const n_layer = 1; // depth of the transformer neural network (number of layers)
pub const n_embd = 16; // width of the network (embedding dimension)
pub const block_size = 16; // maximum context length of the attention window (note: the longest name is 15 characters)
pub const n_head = 4; // number of attention heads
const head_dim = n_embd / n_head; // derived dimension of each head

/// An upper bound on the nodes one step (or one sample) appends, derived from
/// the architecture constants. Every count below is the worst case: attention
/// is charged for a full `block_size` window at every position.
fn stepNodeBound(vocab_size: usize) usize {
    const dot = struct {
        fn n(len: usize) usize {
            return 2 * len; // one mul per term, one add per term after the first
        }
    }.n;
    const norm = 3 * n_embd + 3; // dot + mulK + addK + powK + rescale
    const per_layer = norm // pre-attention norm
        + 4 * (n_embd * dot(n_embd)) // q, k, v and the output projection
        + n_head * block_size * (dot(head_dim) + 1) // attention logits
        + n_head * 5 * block_size // attention softmax
        + n_head * head_dim * dot(block_size) // attention output
        + norm // pre-MLP norm
        + 2 * (4 * n_embd * dot(n_embd)) // both MLP projections
        + 4 * n_embd // relu
        + 2 * n_embd; // both residual adds
    const per_pos = n_embd + norm + n_layer * per_layer + vocab_size * dot(n_embd) // lm_head
    + 5 * vocab_size + 3; // loss softmax, log, negate, accumulate
    return block_size * per_pos + 8;
}

/// A parameter matrix: `n_out` rows of `n_in` values, row-major.
pub const Matrix = struct {
    n_out: usize,
    n_in: usize,
    w: []u32,

    /// Returns the weights of output row `i`.
    fn row(m: Matrix, i: usize) []const u32 {
        return m.w[i * m.n_in ..][0..m.n_in];
    }
};

/// Creates an `n_out` by `n_in` parameter matrix with randomly initialized
/// weights, each one a leaf of the computation graph.
fn matrix(t: *Value, rnd: *Random, gpa: std.mem.Allocator, n_out: usize, n_in: usize) !Matrix {
    const w = try gpa.alloc(u32, n_out * n_in);
    for (w) |*p| p.* = try t.leaf(rnd.gauss(0, 0.08));
    return .{ .n_out = n_out, .n_in = n_in, .w = w };
}

/// The weights of one transformer layer: an attention block and an MLP block.
pub const Layer = struct {
    attn_wq: Matrix,
    attn_wk: Matrix,
    attn_wv: Matrix,
    attn_wo: Matrix,
    mlp_fc1: Matrix,
    mlp_fc2: Matrix,
};

/// Every weight the model has: the embeddings, the layers, and the head that
/// turns the final activations back into scores over the vocabulary.
pub const Model = struct {
    wte: Matrix,
    wpe: Matrix,
    lm_head: Matrix,
    layers: [n_layer]Layer,

    // The creation order below is the insertion order of Python's `state_dict`,
    // which is what fixes the order the weights are drawn from the RNG.
    /// Creates an untrained model sized for `vocab_size` tokens.
    fn init(t: *Value, rnd: *Random, gpa: std.mem.Allocator, vocab_size: usize) !Model {
        var self: Model = .{
            .wte = try matrix(t, rnd, gpa, vocab_size, n_embd),
            .wpe = try matrix(t, rnd, gpa, block_size, n_embd),
            .lm_head = try matrix(t, rnd, gpa, vocab_size, n_embd),
            .layers = undefined,
        };
        for (&self.layers) |*l| {
            l.* = .{
                .attn_wq = try matrix(t, rnd, gpa, n_embd, n_embd),
                .attn_wk = try matrix(t, rnd, gpa, n_embd, n_embd),
                .attn_wv = try matrix(t, rnd, gpa, n_embd, n_embd),
                .attn_wo = try matrix(t, rnd, gpa, n_embd, n_embd),
                .mlp_fc1 = try matrix(t, rnd, gpa, 4 * n_embd, n_embd),
                .mlp_fc2 = try matrix(t, rnd, gpa, n_embd, 4 * n_embd),
            };
        }
        return self;
    }
};

// ---------------------------------------------------------------------------
// The model architecture: tokens and parameters in, logits over what comes next
// Follow GPT-2, blessed among the GPTs, with minor differences:
// layernorm -> rmsnorm, no biases, GeLU -> ReLU
// ---------------------------------------------------------------------------

/// Projects the vector `x` through the weight matrix `w`, writing one output
/// per row of `w` into `out`.
fn linear(t: *Value, out: []u32, x: []const u32, w: Matrix) void {
    std.debug.assert(out.len == w.n_out and x.len == w.n_in);
    for (out, 0..) |*o, i| {
        const wo = w.row(i);
        var acc = t.sumInit(t.mul(wo[0], x[0]));
        for (wo[1..], x[1..]) |wi, xi| acc = t.add(acc, t.mul(wi, xi));
        o.* = acc;
    }
}

/// Turns `logits` into a probability distribution over the same positions,
/// written to `out`: every entry positive, and the whole summing to one.
pub fn softmax(t: *Value, out: []u32, logits: []const u32) void {
    std.debug.assert(out.len == logits.len);
    var max_val = t.data(logits[0]);
    for (logits[1..]) |l| {
        const d = t.data(l);
        if (d > max_val) max_val = d; // `max` keeps the first maximal element
    }
    for (out, logits) |*e, l| e.* = t.expv(t.addK(l, -max_val));
    var total = t.sumInit(out[0]);
    for (out[1..]) |e| total = t.add(total, e);
    // parity: `e / total` is `e * total**-1`, and the comprehension re-evaluates
    // `total**-1` once per element. A single shared reciprocal would fold the
    // gradient contributions in a different order.
    for (out) |*e| e.* = t.mul(e.*, t.powK(total, -1));
}

/// Rescales `x` in place to a root-mean-square magnitude of one, keeping the
/// activations flowing through the network at a stable size.
pub fn rmsnorm(t: *Value, x: []u32) void {
    var acc = t.sumInit(t.mul(x[0], x[0]));
    for (x[1..]) |xi| acc = t.add(acc, t.mul(xi, xi));
    const ms = t.mulK(acc, 1.0 / @as(f64, @floatFromInt(x.len)));
    const scale = t.powK(t.addK(ms, 1e-5), -0.5);
    for (x) |*xi| xi.* = t.mul(xi.*, scale);
}

/// The KV cache doubles as the causal mask: position `p` only ever sees the
/// keys and values appended by positions `0..p`.
pub const Cache = struct {
    keys: [n_layer][block_size][n_embd]u32 = undefined,
    values: [n_layer][block_size][n_embd]u32 = undefined,
    len: usize = 0,
};

/// Forwards a single token at position `pos_id` through the model, writing one
/// score per vocabulary entry into `logits` — how strongly the model expects
/// each token to come next, given this token and everything already in `cache`.
/// The token's own keys and values are appended to `cache` for later positions.
pub fn gpt(t: *Value, model: Model, logits: []u32, token_id: usize, pos_id: usize, cache: *Cache) void {
    var x: [n_embd]u32 = undefined;
    const tok_emb = model.wte.row(token_id); // token embedding
    const pos_emb = model.wpe.row(pos_id); // position embedding
    for (&x, tok_emb, pos_emb) |*xi, tok, p| xi.* = t.add(tok, p); // joint token and position embedding
    rmsnorm(t, &x); // note: not redundant due to backward pass via the residual connection

    const pos = cache.len;
    cache.len += 1;
    // `/ head_dim**0.5` is `* (head_dim**0.5)**-1`
    const attn_scale = std.math.pow(f64, std.math.pow(f64, @as(f64, head_dim), 0.5), -1);

    for (model.layers, 0..) |layer, li| {
        // 1) Multi-head Attention block
        var x_residual = x;
        rmsnorm(t, &x);
        var q: [n_embd]u32 = undefined;
        linear(t, &q, &x, layer.attn_wq);
        linear(t, &cache.keys[li][pos], &x, layer.attn_wk);
        linear(t, &cache.values[li][pos], &x, layer.attn_wv);

        var x_attn: [n_embd]u32 = undefined;
        for (0..n_head) |h| {
            const hs = h * head_dim;
            const q_h = q[hs..][0..head_dim];

            var attn_logits: [block_size]u32 = undefined;
            for (0..cache.len) |ti| {
                const k_h = cache.keys[li][ti][hs..][0..head_dim];
                var acc = t.sumInit(t.mul(q_h[0], k_h[0]));
                for (q_h[1..], k_h[1..]) |qj, kj| acc = t.add(acc, t.mul(qj, kj));
                attn_logits[ti] = t.mulK(acc, attn_scale);
            }
            var attn_weights: [block_size]u32 = undefined;
            softmax(t, attn_weights[0..cache.len], attn_logits[0..cache.len]);

            for (0..head_dim) |j| {
                var acc = t.sumInit(t.mul(attn_weights[0], cache.values[li][0][hs + j]));
                for (1..cache.len) |ti| {
                    acc = t.add(acc, t.mul(attn_weights[ti], cache.values[li][ti][hs + j]));
                }
                x_attn[hs + j] = acc;
            }
        }
        linear(t, &x, &x_attn, layer.attn_wo);
        for (&x, x_residual) |*xi, res| xi.* = t.add(xi.*, res);

        // 2) MLP block
        x_residual = x;
        rmsnorm(t, &x);
        var h1: [4 * n_embd]u32 = undefined;
        linear(t, &h1, &x, layer.mlp_fc1);
        for (&h1) |*hi| hi.* = t.relu(hi.*);
        linear(t, &x, &h1, layer.mlp_fc2);
        for (&x, x_residual) |*xi, res| xi.* = t.add(xi.*, res);
    }

    linear(t, logits, &x, model.lm_head);
}

// ---------------------------------------------------------------------------

/// Trains a fresh model on the names corpus and then samples new names from it,
/// reporting progress and the generated names on stdout.
pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: std.Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    var rng: Random = .{};
    rng.seed(42); // fixed seed, so runs are reproducible

    // The dataset `docs`: a list of documents (e.g. a list of names). Unlike the
    // Python original this does not download the corpus; `..` is tried so that
    // the program works from the repo root or from here.
    const raw = std.Io.Dir.cwd().readFileAlloc(io, "input.txt", arena, .unlimited) catch
        std.Io.Dir.cwd().readFileAlloc(io, "../input.txt", arena, .unlimited) catch {
        std.debug.print(
            \\could not read input.txt (tried ./ and ../)
            \\fetch it with `make data` in the microgpt directory
            \\
        , .{});
        return error.MissingDataset;
    };

    var docs: std.ArrayList([]const u8) = .empty;
    defer docs.deinit(gpa);
    var lines = std.mem.splitScalar(u8, raw, '\n');
    while (lines.next()) |line| {
        const doc = std.mem.trim(u8, line, &std.ascii.whitespace);
        if (doc.len > 0) try docs.append(gpa, doc);
    }
    rng.shuffle([]const u8, docs.items);
    try out.print("num docs: {d}\n", .{docs.items.len});

    const tok = Tokenizer.init(docs.items);
    const vocab_size = tok.vocab_size;
    try out.print("vocab size: {d}\n", .{vocab_size});

    var tape: Value = .{ .gpa = gpa };
    defer tape.deinit();
    const model = try Model.init(&tape, &rng, arena, vocab_size);
    tape.n_params = tape.nodes.items.len;
    const n_params = tape.n_params;
    // One allocation for the whole run: `rewind` keeps the capacity, so no step
    // ever touches the allocator again.
    try tape.reserve(stepNodeBound(vocab_size));
    try out.print("num params: {d}\n", .{n_params});
    try out.flush();

    const logits = try arena.alloc(u32, vocab_size);
    const probs = try arena.alloc(u32, vocab_size);

    try train.run(&tape, model, out, arena, docs.items, &tok, logits, probs);
    try inference.run(&tape, model, out, arena, &rng, &tok, logits, probs);
    try out.flush();
}
