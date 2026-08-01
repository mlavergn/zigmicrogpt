// The most atomic way to train and run inference for a GPT in pure, dependency-free Zig.
// This file is the complete algorithm.
// Everything else is just efficiency.
// A Zig port of microgpt.py from @karpathy.
// @mlavergn

const std = @import("std");
const Random = @import("random_port.zig").Random;

// Let there be Autograd to recursively apply the chain rule
const Node = struct {
    data: f64, // scalar value of this node calculated during the forward pass
    grad: f64, // derivative of the loss w.r.t. this node, calculated in the backward pass
    local_grads: [2]f64, // local derivative of this node w.r.t. each of its children
    children: [2]u32, // children of this node in the computation graph
    n_children: u8,
};

pub const Value = struct {
    gpa: std.mem.Allocator,
    nodes: std.ArrayList(Node) = .empty,
    visited: std.ArrayList(bool) = .empty,
    topo: std.ArrayList(u32) = .empty,
    stack: std.ArrayList(struct { node: u32, next_child: u8 }) = .empty,
    n_params: usize = 0,

    pub fn deinit(t: *Value) void {
        t.nodes.deinit(t.gpa);
        t.visited.deinit(t.gpa);
        t.topo.deinit(t.gpa);
        t.stack.deinit(t.gpa);
    }

    pub fn data(t: *const Value, v: u32) f64 {
        return t.nodes.items[v].data;
    }

    fn rewind(t: *Value) void {
        t.nodes.shrinkRetainingCapacity(t.n_params);
    }

    fn push(t: *Value, d: f64, ch: [2]u32, lg: [2]f64, nch: u8) !u32 {
        const idx: u32 = @intCast(t.nodes.items.len);
        try t.nodes.append(t.gpa, .{ .data = d, .grad = 0, .local_grads = lg, .children = ch, .n_children = nch });
        return idx;
    }

    pub fn leaf(t: *Value, d: f64) !u32 {
        return t.push(d, .{ 0, 0 }, .{ 0, 0 }, 0);
    }

    pub fn add(t: *Value, a: u32, b: u32) !u32 {
        return t.push(t.data(a) + t.data(b), .{ a, b }, .{ 1, 1 }, 2);
    }

    pub fn mul(t: *Value, a: u32, b: u32) !u32 {
        return t.push(t.data(a) * t.data(b), .{ a, b }, .{ t.data(b), t.data(a) }, 2);
    }

    fn addK(t: *Value, a: u32, k: f64) !u32 {
        return t.push(t.data(a) + k, .{ a, 0 }, .{ 1, 0 }, 1);
    }

    fn mulK(t: *Value, a: u32, k: f64) !u32 {
        return t.push(t.data(a) * k, .{ a, 0 }, .{ k, 0 }, 1);
    }

    fn powK(t: *Value, a: u32, k: f64) !u32 {
        const d = t.data(a);
        return t.push(std.math.pow(f64, d, k), .{ a, 0 }, .{ k * std.math.pow(f64, d, k - 1), 0 }, 1);
    }

    fn logv(t: *Value, a: u32) !u32 {
        const d = t.data(a);
        return t.push(@log(d), .{ a, 0 }, .{ 1 / d, 0 }, 1);
    }

    fn expv(t: *Value, a: u32) !u32 {
        const d = t.data(a);
        return t.push(@exp(d), .{ a, 0 }, .{ @exp(d), 0 }, 1);
    }

    pub fn relu(t: *Value, a: u32) !u32 {
        const d = t.data(a);
        return t.push(@max(0, d), .{ a, 0 }, .{ if (d > 0) @as(f64, 1) else 0, 0 }, 1);
    }

    fn sumInit(t: *Value, first: u32) !u32 {
        return t.addK(first, 0.0);
    }

    pub fn backward(t: *Value, root: u32) !void {
        try t.visited.resize(t.gpa, t.nodes.items.len);
        @memset(t.visited.items, false);
        t.topo.clearRetainingCapacity();
        t.stack.clearRetainingCapacity();

        t.visited.items[root] = true;
        try t.stack.append(t.gpa, .{ .node = root, .next_child = 0 });
        while (t.stack.items.len > 0) {
            const top = t.stack.items.len - 1;
            const node_idx = t.stack.items[top].node;
            const next_child = t.stack.items[top].next_child;
            const node = t.nodes.items[node_idx];
            if (next_child < node.n_children) {
                t.stack.items[top].next_child = next_child + 1;
                const child = node.children[next_child];
                if (!t.visited.items[child]) {
                    t.visited.items[child] = true;
                    try t.stack.append(t.gpa, .{ .node = child, .next_child = 0 });
                }
            } else {
                _ = t.stack.pop();
                try t.topo.append(t.gpa, node_idx);
            }
        }

        t.nodes.items[root].grad = 1;
        var i = t.topo.items.len;
        while (i > 0) {
            i -= 1;
            const node = t.nodes.items[t.topo.items[i]];
            var c: u8 = 0;
            while (c < node.n_children) : (c += 1) {
                t.nodes.items[node.children[c]].grad += node.local_grads[c] * node.grad;
            }
        }
    }
};

// Initialize the parameters, to store the knowledge of the model
const n_layer = 1; // depth of the transformer neural network (number of layers)
const n_embd = 16; // width of the network (embedding dimension)
const block_size = 16; // maximum context length of the attention window (note: the longest name is 15 characters)
const n_head = 4; // number of attention heads
const head_dim = n_embd / n_head; // derived dimension of each head
const max_vocab = 256;

const Matrix = struct {
    n_out: usize,
    n_in: usize,
    w: []u32,

    fn row(m: Matrix, i: usize) []const u32 {
        return m.w[i * m.n_in ..][0..m.n_in];
    }
};

fn matrix(t: *Value, r: *Random, gpa: std.mem.Allocator, n_out: usize, n_in: usize) !Matrix {
    const w = try gpa.alloc(u32, n_out * n_in);
    for (w) |*p| p.* = try t.leaf(r.gauss(0, 0.08));
    return .{ .n_out = n_out, .n_in = n_in, .w = w };
}

const Layer = struct {
    attn_wq: Matrix,
    attn_wk: Matrix,
    attn_wv: Matrix,
    attn_wo: Matrix,
    mlp_fc1: Matrix,
    mlp_fc2: Matrix,
};

const Cache = struct {
    keys: [n_layer][block_size][n_embd]u32 = undefined,
    values: [n_layer][block_size][n_embd]u32 = undefined,
    len: usize = 0,
};

const Model = struct {
    wte: Matrix,
    wpe: Matrix,
    lm_head: Matrix,
    layers: [n_layer]Layer,

    fn init(t: *Value, r: *Random, gpa: std.mem.Allocator, vocab_size: usize) !Model {
        var self: Model = .{
            .wte = try matrix(t, r, gpa, vocab_size, n_embd),
            .wpe = try matrix(t, r, gpa, block_size, n_embd),
            .lm_head = try matrix(t, r, gpa, vocab_size, n_embd),
            .layers = undefined,
        };
        for (&self.layers) |*l| {
            l.* = .{
                .attn_wq = try matrix(t, r, gpa, n_embd, n_embd),
                .attn_wk = try matrix(t, r, gpa, n_embd, n_embd),
                .attn_wv = try matrix(t, r, gpa, n_embd, n_embd),
                .attn_wo = try matrix(t, r, gpa, n_embd, n_embd),
                .mlp_fc1 = try matrix(t, r, gpa, 4 * n_embd, n_embd),
                .mlp_fc2 = try matrix(t, r, gpa, n_embd, 4 * n_embd),
            };
        }
        return self;
    }
};

// The model architecture: tokens and parameters in, logits over what comes next
// Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
fn linear(t: *Value, out: []u32, x: []const u32, w: Matrix) !void {
    std.debug.assert(out.len == w.n_out and x.len == w.n_in);
    for (out, 0..) |*o, i| {
        const wo = w.row(i);
        var acc = try t.sumInit(try t.mul(wo[0], x[0]));
        for (wo[1..], x[1..]) |wi, xi| acc = try t.add(acc, try t.mul(wi, xi));
        o.* = acc;
    }
}

pub fn softmax(t: *Value, out: []u32, logits: []const u32) !void {
    std.debug.assert(out.len == logits.len);
    var max_val = t.data(logits[0]);
    for (logits[1..]) |l| max_val = @max(max_val, t.data(l));
    for (out, logits) |*e, l| e.* = try t.expv(try t.addK(l, -max_val));
    var total = try t.sumInit(out[0]);
    for (out[1..]) |e| total = try t.add(total, e);
    for (out) |*e| e.* = try t.mul(e.*, try t.powK(total, -1));
}

pub fn rmsnorm(t: *Value, x: []u32) !void {
    var acc = try t.sumInit(try t.mul(x[0], x[0]));
    for (x[1..]) |xi| acc = try t.add(acc, try t.mul(xi, xi));
    const ms = try t.mulK(acc, 1.0 / @as(f64, @floatFromInt(x.len)));
    const scale = try t.powK(try t.addK(ms, 1e-5), -0.5);
    for (x) |*xi| xi.* = try t.mul(xi.*, scale);
}

fn gpt(t: *Value, model: Model, logits: []u32, token_id: usize, pos_id: usize, cache: *Cache) !void {
    var x: [n_embd]u32 = undefined;
    const tok_emb = model.wte.row(token_id); // token embedding
    const pos_emb = model.wpe.row(pos_id); // position embedding
    for (&x, tok_emb, pos_emb) |*xi, tok, p| xi.* = try t.add(tok, p); // joint token and position embedding
    try rmsnorm(t, &x); // note: not redundant due to backward pass via the residual connection

    const pos = cache.len;
    cache.len += 1;
    const attn_scale = std.math.pow(f64, std.math.pow(f64, @as(f64, head_dim), 0.5), -1);

    for (model.layers, 0..) |layer, li| {
        // 1) Multi-head Attention block
        var x_residual = x;
        try rmsnorm(t, &x);
        var q: [n_embd]u32 = undefined;
        try linear(t, &q, &x, layer.attn_wq);
        try linear(t, &cache.keys[li][pos], &x, layer.attn_wk);
        try linear(t, &cache.values[li][pos], &x, layer.attn_wv);

        var x_attn: [n_embd]u32 = undefined;
        for (0..n_head) |h| {
            const hs = h * head_dim;
            const q_h = q[hs..][0..head_dim];

            var attn_logits: [block_size]u32 = undefined;
            for (0..cache.len) |ti| {
                const k_h = cache.keys[li][ti][hs..][0..head_dim];
                var acc = try t.sumInit(try t.mul(q_h[0], k_h[0]));
                for (q_h[1..], k_h[1..]) |qj, kj| acc = try t.add(acc, try t.mul(qj, kj));
                attn_logits[ti] = try t.mulK(acc, attn_scale);
            }
            var attn_weights: [block_size]u32 = undefined;
            try softmax(t, attn_weights[0..cache.len], attn_logits[0..cache.len]);

            for (0..head_dim) |j| {
                var acc = try t.sumInit(try t.mul(attn_weights[0], cache.values[li][0][hs + j]));
                for (1..cache.len) |ti| {
                    acc = try t.add(acc, try t.mul(attn_weights[ti], cache.values[li][ti][hs + j]));
                }
                x_attn[hs + j] = acc;
            }
        }
        try linear(t, &x, &x_attn, layer.attn_wo);
        for (&x, x_residual) |*xi, res| xi.* = try t.add(xi.*, res);

        // 2) MLP block
        x_residual = x;
        try rmsnorm(t, &x);
        var h1: [4 * n_embd]u32 = undefined;
        try linear(t, &h1, &x, layer.mlp_fc1);
        for (&h1) |*hi| hi.* = try t.relu(hi.*);
        try linear(t, &x, &h1, layer.mlp_fc2);
        for (&x, x_residual) |*xi, res| xi.* = try t.add(xi.*, res);
    }

    try linear(t, logits, &x, model.lm_head);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: std.Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    var rng: Random = .{};
    rng.seed(42); // Let there be order among chaos

    // Let there be a Dataset `docs`: a list of documents (e.g. a list of names)
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

    // Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
    var seen = [_]bool{false} ** max_vocab;
    for (docs.items) |doc| for (doc) |ch| {
        seen[ch] = true;
    };
    var uchars: [max_vocab]u8 = undefined;
    var token_of: [max_vocab]u8 = undefined;
    var n_uchars: usize = 0;
    for (seen, 0..) |present, ch| if (present) {
        uchars[n_uchars] = @intCast(ch); // unique characters, sorted, become token ids 0..n-1
        token_of[ch] = @intCast(n_uchars);
        n_uchars += 1;
    };
    const bos = n_uchars; // token id for a special Beginning of Sequence (BOS) token
    const vocab_size = n_uchars + 1; // total number of unique tokens, +1 is for BOS
    std.debug.assert(vocab_size <= max_vocab);
    try out.print("vocab size: {d}\n", .{vocab_size});

    var tape: Value = .{ .gpa = gpa };
    defer tape.deinit();
    const model = try Model.init(&tape, &rng, arena, vocab_size);
    tape.n_params = tape.nodes.items.len;
    const n_params = tape.n_params;
    try out.print("num params: {d}\n", .{n_params});
    try out.flush();

    // Let there be Adam, the blessed optimizer and its buffers
    const learning_rate, const beta1, const beta2, const eps_adam = .{ 0.01, 0.85, 0.99, 1e-8 };
    const m = try arena.alloc(f64, n_params); // first moment buffer
    const v = try arena.alloc(f64, n_params); // second moment buffer
    @memset(m, 0);
    @memset(v, 0);

    const logits, const probs = .{ try arena.alloc(u32, vocab_size), try arena.alloc(u32, vocab_size) };

    // Repeat in sequence
    const num_steps = 1000; // number of training steps
    for (0..num_steps) |step| {
        tape.rewind();

        // Take a single document, tokenize it, surround it with the BOS special token on both sides
        const doc = docs.items[step % docs.items.len];
        var tokens: [block_size + 2]usize = undefined;
        tokens[0] = bos;
        var n_tokens: usize = 1;
        for (doc) |ch| {
            if (n_tokens + 1 >= tokens.len) break;
            tokens[n_tokens] = token_of[ch];
            n_tokens += 1;
        }
        tokens[n_tokens] = bos;
        n_tokens += 1;
        const n = @min(block_size, n_tokens - 1);

        // Forward the token sequence through the model, building up the computation graph all the way to the loss
        var cache: Cache = .{};
        var loss: u32 = undefined;
        for (0..n) |pos_id| {
            const token_id = tokens[pos_id];
            const target_id = tokens[pos_id + 1];
            try gpt(&tape, model, logits, token_id, pos_id, &cache);
            try softmax(&tape, probs, logits);
            const loss_t = try tape.mulK(try tape.logv(probs[target_id]), -1);
            loss = if (pos_id == 0) try tape.sumInit(loss_t) else try tape.add(loss, loss_t);
        }
        // final average loss over the document sequence. May yours be low.
        loss = try tape.mulK(loss, 1.0 / @as(f64, @floatFromInt(n)));

        // Backward the loss, calculating the gradients with respect to all model parameters
        try tape.backward(loss);

        // Adam optimizer update: update the model parameters based on the corresponding gradients
        const lr_t = learning_rate * (1 - @as(f64, @floatFromInt(step)) / num_steps); // linear learning rate decay
        const bias1 = 1 - std.math.pow(f64, beta1, @floatFromInt(step + 1));
        const bias2 = 1 - std.math.pow(f64, beta2, @floatFromInt(step + 1));
        for (tape.nodes.items[0..n_params], m, v) |*p, *mi, *vi| {
            mi.* = beta1 * mi.* + (1 - beta1) * p.grad;
            vi.* = beta2 * vi.* + (1 - beta2) * std.math.pow(f64, p.grad, 2);
            const m_hat = mi.* / bias1;
            const v_hat = vi.* / bias2;
            p.data -= lr_t * m_hat / (std.math.pow(f64, v_hat, 0.5) + eps_adam);
            p.grad = 0;
        }

        try out.print("step {d:>4} / {d:>4} | loss {d:.4}\r", .{ step + 1, num_steps, tape.data(loss) });
        try out.flush();
    }

    // Inference: may the model babble back to us
    const temperature = 0.5; // in (0, 1], control the "creativity" of generated text, low to high
    const inv_temperature = std.math.pow(f64, temperature, -1);
    try out.print("\n--- inference (new, hallucinated names) ---\n", .{});
    const weights = try arena.alloc(f64, vocab_size);
    const cum = try arena.alloc(f64, vocab_size);
    for (0..20) |sample_idx| {
        tape.rewind();
        var cache: Cache = .{};
        var token_id: usize = bos;
        var sample: [block_size]u8 = undefined;
        var sample_len: usize = 0;
        for (0..block_size) |pos_id| {
            try gpt(&tape, model, logits, token_id, pos_id, &cache);
            for (logits) |*l| l.* = try tape.mulK(l.*, inv_temperature);
            try softmax(&tape, probs, logits);
            for (weights, probs) |*w, p| w.* = tape.data(p);
            token_id = rng.choices(weights, cum);
            if (token_id == bos) break;
            sample[sample_len] = uchars[token_id];
            sample_len += 1;
        }
        try out.print("sample {d:>2}: {s}\n", .{ sample_idx + 1, sample[0..sample_len] });
    }
    try out.flush();
}
