// ---------------------------------------------------------------------------
// Training: one document per step, forwarded to a loss, backpropagated, and
// folded into the parameters by Adam.
// ---------------------------------------------------------------------------

const std = @import("std");
const log = std.log.scoped(.microgptzig_trainer);
const mod = @import("module.zig");

/// Trains a model's parameters in place on a list of documents.
pub const Trainer = struct {
    const Self = @This();

    pub const num_steps = 1000; // number of training steps

    // The Adam optimizer's hyperparameters
    const learning_rate = 0.01;
    const beta1 = 0.85;
    const beta2 = 0.99;
    const eps_adam = 1e-8;

    allocator: std.mem.Allocator,
    tape: *mod.Tape,
    model: *const mod.Model,
    tokenizer: *const mod.Tokenizer,
    // Buffers
    m: []f64 = &.{}, // first moment buffer
    v: []f64 = &.{}, // second moment buffer
    logits: []u32 = &.{},
    probs: []u32 = &.{},

    /// Creates a trainer for `model`, whose parameters are the leaves of `tape`.
    pub fn init(allocator: std.mem.Allocator, tape: *mod.Tape, model: *const mod.Model, tokenizer: *const mod.Tokenizer) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var self: Self = .{ .allocator = allocator, .tape = tape, .model = model, .tokenizer = tokenizer };
        errdefer self.deinit();
        self.m = try allocator.alloc(f64, tape.n_params);
        @memset(self.m, 0);
        self.v = try allocator.alloc(f64, tape.n_params);
        @memset(self.v, 0);
        self.logits = try allocator.alloc(u32, tokenizer.vocab_size);
        self.probs = try allocator.alloc(u32, tokenizer.vocab_size);
        return self;
    }

    /// Releases the optimizer state and scratch, in reverse order of `init`.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.allocator.free(self.probs);
        self.allocator.free(self.logits);
        self.allocator.free(self.v);
        self.allocator.free(self.m);
    }

    /// Trains for `num_steps` steps, one document of `docs` per step, reporting
    /// the loss to `out` as it goes.
    pub fn run(self: *Self, out: *std.Io.Writer, docs: []const []const u8) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const tape = self.tape;
        const n_params = tape.n_params;

        // Repeat in sequence
        for (0..num_steps) |step| {
            tape.rewind();

            // Take a single document, tokenize it, surround it with the BOS
            // special token on both sides
            const doc = docs[step % docs.len];
            var tokens: [mod.Model.block_size + 2]usize = undefined;
            const n_tokens = self.tokenizer.encode(doc, &tokens);
            const n = @min(mod.Model.block_size, n_tokens - 1);

            // Forward the token sequence through the model, building up the
            // computation graph all the way to the loss
            var cache: mod.Cache = .{};
            var loss: u32 = undefined;
            for (0..n) |pos_id| {
                const token_id = tokens[pos_id];
                const target_id = tokens[pos_id + 1];
                self.model.gpt(tape, self.logits, token_id, pos_id, &cache);
                mod.Model.softmax(tape, self.probs, self.logits);
                const loss_t = tape.mulK(tape.logv(self.probs[target_id]), -1);
                loss = if (pos_id == 0) tape.sumInit(loss_t) else tape.add(loss, loss_t);
            }
            // final average loss over the document sequence. May yours be low.
            loss = tape.mulK(loss, 1.0 / @as(f64, @floatFromInt(n)));

            // Backward the loss, calculating the gradients with respect to all
            // model parameters
            tape.backward(loss);

            // Adam optimizer update: update the model parameters based on the
            // corresponding gradients
            const lr_t = learning_rate * (1 - @as(f64, @floatFromInt(step)) / num_steps); // linear learning rate decay
            const bias1 = 1 - mod.Pow.pow(beta1, @floatFromInt(step + 1));
            const bias2 = 1 - mod.Pow.pow(beta2, @floatFromInt(step + 1));
            for (tape.nodes.items[0..n_params], self.m, self.v) |*p, *mi, *vi| {
                mi.* = beta1 * mi.* + (1 - beta1) * p.grad;
                vi.* = beta2 * vi.* + (1 - beta2) * mod.Pow.pow(p.grad, 2);
                const m_hat = mi.* / bias1;
                const v_hat = vi.* / bias2;
                p.data -= lr_t * m_hat / (mod.Pow.pow(v_hat, 0.5) + eps_adam);
                p.grad = 0;
            }

            try out.print("step {d:>4} / {d:>4} | loss {d:.4}\r", .{ step + 1, num_steps, tape.data(loss) });
            try out.flush();
        }
    }
};
