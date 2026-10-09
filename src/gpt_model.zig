// ---------------------------------------------------------------------------
// The model: tokens and parameters in, logits over what comes next. Follow
// GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm,
// no biases, GeLU -> ReLU
// ---------------------------------------------------------------------------

const std = @import("std");
const log = std.log.scoped(.microgptzig_model);
const mod = @import("module.zig");

/// A parameter matrix: `n_out` rows of `n_in` values, row-major.
const Matrix = struct {
    const Self = @This();
    /// No weights yet, so a partly built `Model` can still be freed.
    const empty: Self = .{ .n_out = 0, .n_in = 0, .w = &.{} };
    n_out: usize,
    n_in: usize,
    w: []u32,

    /// Returns the weights of output row `i`.
    fn row(self: *const Self, i: usize) []const u32 {
        return self.w[i * self.n_in ..][0..self.n_in];
    }
};

/// The weights of one transformer layer: an attention block and an MLP block.
const Layer = struct {
    const Self = @This();
    attn_wq: Matrix = .empty,
    attn_wk: Matrix = .empty,
    attn_wv: Matrix = .empty,
    attn_wo: Matrix = .empty,
    mlp_fc1: Matrix = .empty,
    mlp_fc2: Matrix = .empty,
};

/// The KV cache doubles as the causal mask: position `p` only ever sees the
/// keys and values appended by positions `0..p`.
pub const Cache = struct {
    const Self = @This();
    keys: [Model.n_layer][Model.block_size][Model.n_embd]u32 = undefined,
    values: [Model.n_layer][Model.block_size][Model.n_embd]u32 = undefined,
    len: usize = 0,
};

/// Every weight the model has: the embeddings, the layers, and the head that
/// turns the final activations back into scores over the vocabulary.
pub const Model = struct {
    const Self = @This();

    // The parameters, to store the knowledge of the model
    pub const n_layer = 1; // depth of the transformer neural network (number of layers)
    pub const n_embd = 16; // width of the network (embedding dimension)
    pub const block_size = 16; // maximum context length of the attention window (note: the longest name is 15 characters)
    pub const n_head = 4; // number of attention heads
    const head_dim = n_embd / n_head; // derived dimension of each head

    allocator: std.mem.Allocator,
    wte: Matrix = .empty,
    wpe: Matrix = .empty,
    lm_head: Matrix = .empty,
    layers: [n_layer]Layer = @splat(.{}),

    // The creation order below is the insertion order of Python's `state_dict`,
    // which is what fixes the order the weights are drawn from the RNG.
    /// Creates an untrained model sized for `vocab_size` tokens. Its weights
    /// are drawn from `rng` and recorded on `tape` as leaves.
    pub fn init(allocator: std.mem.Allocator, tape: *mod.Tape, rng: *mod.Random, vocab_size: usize) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var self: Self = .{ .allocator = allocator };
        errdefer self.deinit();
        self.wte = try self.matrix(tape, rng, vocab_size, n_embd);
        self.wpe = try self.matrix(tape, rng, block_size, n_embd);
        self.lm_head = try self.matrix(tape, rng, vocab_size, n_embd);
        for (&self.layers) |*l| {
            l.attn_wq = try self.matrix(tape, rng, n_embd, n_embd);
            l.attn_wk = try self.matrix(tape, rng, n_embd, n_embd);
            l.attn_wv = try self.matrix(tape, rng, n_embd, n_embd);
            l.attn_wo = try self.matrix(tape, rng, n_embd, n_embd);
            l.mlp_fc1 = try self.matrix(tape, rng, 4 * n_embd, n_embd);
            l.mlp_fc2 = try self.matrix(tape, rng, n_embd, 4 * n_embd);
        }
        return self;
    }

    /// Releases every weight matrix, in reverse order of `init`.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var li = self.layers.len;
        while (li > 0) {
            li -= 1;
            const l = &self.layers[li];
            self.allocator.free(l.mlp_fc2.w);
            self.allocator.free(l.mlp_fc1.w);
            self.allocator.free(l.attn_wo.w);
            self.allocator.free(l.attn_wv.w);
            self.allocator.free(l.attn_wk.w);
            self.allocator.free(l.attn_wq.w);
        }
        self.allocator.free(self.lm_head.w);
        self.allocator.free(self.wpe.w);
        self.allocator.free(self.wte.w);
    }

    /// An upper bound on the nodes one step (or one sample) appends, derived
    /// from the architecture constants. Every count below is the worst case:
    /// attention is charged for a full `block_size` window at every position.
    pub fn stepNodeBound(vocab_size: usize) usize {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
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

    /// Forwards a single token at position `pos_id` through the model, writing
    /// one score per vocabulary entry into `logits`: how strongly the model
    /// expects each token to come next, given this token and everything
    /// already in `cache`. The token's own keys and values are appended to
    /// `cache` for later positions.
    pub fn gpt(self: *const Self, tape: *mod.Tape, logits: []u32, token_id: usize, pos_id: usize, cache: *Cache) void {
        var x: [n_embd]u32 = undefined;
        const tok_emb = self.wte.row(token_id); // token embedding
        const pos_emb = self.wpe.row(pos_id); // position embedding
        for (&x, tok_emb, pos_emb) |*xi, tok, p| xi.* = tape.add(tok, p); // joint token and position embedding
        rmsnorm(tape, &x); // note: not redundant due to backward pass via the residual connection

        const pos = cache.len;
        cache.len += 1;
        // `/ head_dim**0.5` is `* (head_dim**0.5)**-1`
        const attn_scale = mod.Pow.pow(mod.Pow.pow(@as(f64, head_dim), 0.5), -1);

        for (&self.layers, 0..) |*layer, li| {
            // 1) Multi-head Attention block
            var x_residual = x;
            rmsnorm(tape, &x);
            var q: [n_embd]u32 = undefined;
            linear(tape, &q, &x, &layer.attn_wq);
            linear(tape, &cache.keys[li][pos], &x, &layer.attn_wk);
            linear(tape, &cache.values[li][pos], &x, &layer.attn_wv);

            var x_attn: [n_embd]u32 = undefined;
            for (0..n_head) |h| {
                const hs = h * head_dim;
                const q_h = q[hs..][0..head_dim];

                var attn_logits: [block_size]u32 = undefined;
                for (0..cache.len) |ti| {
                    const k_h = cache.keys[li][ti][hs..][0..head_dim];
                    var acc = tape.sumInit(tape.mul(q_h[0], k_h[0]));
                    for (q_h[1..], k_h[1..]) |qj, kj| acc = tape.add(acc, tape.mul(qj, kj));
                    attn_logits[ti] = tape.mulK(acc, attn_scale);
                }
                var attn_weights: [block_size]u32 = undefined;
                softmax(tape, attn_weights[0..cache.len], attn_logits[0..cache.len]);

                for (0..head_dim) |j| {
                    var acc = tape.sumInit(tape.mul(attn_weights[0], cache.values[li][0][hs + j]));
                    for (1..cache.len) |ti| {
                        acc = tape.add(acc, tape.mul(attn_weights[ti], cache.values[li][ti][hs + j]));
                    }
                    x_attn[hs + j] = acc;
                }
            }
            linear(tape, &x, &x_attn, &layer.attn_wo);
            for (&x, x_residual) |*xi, res| xi.* = tape.add(xi.*, res);

            // 2) MLP block
            x_residual = x;
            rmsnorm(tape, &x);
            var h1: [4 * n_embd]u32 = undefined;
            linear(tape, &h1, &x, &layer.mlp_fc1);
            for (&h1) |*hi| hi.* = tape.relu(hi.*);
            linear(tape, &x, &h1, &layer.mlp_fc2);
            for (&x, x_residual) |*xi, res| xi.* = tape.add(xi.*, res);
        }

        linear(tape, logits, &x, &self.lm_head);
    }

    /// Turns `logits` into a probability distribution over the same positions,
    /// written to `out`: every entry positive, and the whole summing to one.
    pub fn softmax(tape: *mod.Tape, out: []u32, logits: []const u32) void {
        std.debug.assert(out.len == logits.len);
        var max_val = tape.data(logits[0]);
        for (logits[1..]) |l| {
            const d = tape.data(l);
            if (d > max_val) max_val = d; // `max` keeps the first maximal element
        }
        for (out, logits) |*e, l| e.* = tape.expv(tape.addK(l, -max_val));
        var total = tape.sumInit(out[0]);
        for (out[1..]) |e| total = tape.add(total, e);
        // parity: `e / total` is `e * total**-1`, and the comprehension
        // re-evaluates `total**-1` once per element. A single shared reciprocal
        // would fold the gradient contributions in a different order.
        for (out) |*e| e.* = tape.mul(e.*, tape.powK(total, -1));
    }

    /// Rescales `x` in place to a root-mean-square magnitude of one, keeping
    /// the activations flowing through the network at a stable size.
    pub fn rmsnorm(tape: *mod.Tape, x: []u32) void {
        var acc = tape.sumInit(tape.mul(x[0], x[0]));
        for (x[1..]) |xi| acc = tape.add(acc, tape.mul(xi, xi));
        const ms = tape.mulK(acc, 1.0 / @as(f64, @floatFromInt(x.len)));
        const scale = tape.powK(tape.addK(ms, 1e-5), -0.5);
        for (x) |*xi| xi.* = tape.mul(xi.*, scale);
    }

    /// Creates an `n_out` by `n_in` parameter matrix with randomly initialized
    /// weights, each one a leaf of the computation graph.
    fn matrix(self: *Self, tape: *mod.Tape, rng: *mod.Random, n_out: usize, n_in: usize) !Matrix {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const w = try self.allocator.alloc(u32, n_out * n_in);
        errdefer self.allocator.free(w);
        for (w) |*p| p.* = try tape.leaf(rng.gauss(0, 0.08));
        return .{ .n_out = n_out, .n_in = n_in, .w = w };
    }

    /// Projects the vector `x` through the weight matrix `w`, writing one
    /// output per row of `w` into `out`.
    fn linear(tape: *mod.Tape, out: []u32, x: []const u32, w: *const Matrix) void {
        std.debug.assert(out.len == w.n_out and x.len == w.n_in);
        for (out, 0..) |*o, i| {
            const wo = w.row(i);
            var acc = tape.sumInit(tape.mul(wo[0], x[0]));
            for (wo[1..], x[1..]) |wi, xi| acc = tape.add(acc, tape.mul(wi, xi));
            o.* = acc;
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "softmax sums to one and rmsnorm normalizes" {
    var tape = mod.Tape.init(std.testing.allocator);
    defer tape.deinit();
    try tape.reserve(64);
    var xs: [4]u32 = undefined;
    for (&xs, [_]f64{ 1.0, 2.0, 3.0, 4.0 }) |*x, d| x.* = try tape.leaf(d);

    var probs: [4]u32 = undefined;
    Model.softmax(&tape, &probs, &xs);
    var total: f64 = 0;
    for (probs) |p| total += tape.data(p);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), total, 1e-12);

    Model.rmsnorm(&tape, &xs);
    var ms: f64 = 0;
    for (xs) |x| ms += tape.data(x) * tape.data(x);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), ms / xs.len, 1e-5);
}
