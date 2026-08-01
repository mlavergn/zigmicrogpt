// Training: one document per step, forwarded to a loss, backpropagated, and
// folded into the parameters by Adam.

const std = @import("std");
const module = @import("gpt_main.zig");

const Value = module.Value;
const Model = module.Model;
const Cache = module.Cache;
const block_size = module.block_size;
const Tokenizer = @import("gpt_tokenizer.zig").Tokenizer;

pub const num_steps = 1000; // number of training steps

// The Adam optimizer's hyperparameters
const learning_rate = 0.01;
const beta1 = 0.85;
const beta2 = 0.99;
const eps_adam = 1e-8;

/// Trains `model` in place for `num_steps` steps, reporting the loss as it goes.
/// `logits` and `probs` are caller-owned scratch, one slot per vocabulary entry.
pub fn run(tape: *Value, model: Model, out: *std.Io.Writer, arena: std.mem.Allocator, docs: []const []const u8, tok: *const Tokenizer, logits: []u32, probs: []u32) !void {
    const n_params = tape.n_params;
    const m = try arena.alloc(f64, n_params); // first moment buffer
    const v = try arena.alloc(f64, n_params); // second moment buffer
    @memset(m, 0);
    @memset(v, 0);

    // Repeat in sequence
    for (0..num_steps) |step| {
        tape.rewind();

        // Take a single document, tokenize it, surround it with the BOS special
        // token on both sides
        const doc = docs[step % docs.len];
        var tokens: [block_size + 2]usize = undefined;
        const n_tokens = tok.encode(doc, &tokens);
        const n = @min(block_size, n_tokens - 1);

        // Forward the token sequence through the model, building up the
        // computation graph all the way to the loss
        var cache: Cache = .{};
        var loss: u32 = undefined;
        for (0..n) |pos_id| {
            const token_id = tokens[pos_id];
            const target_id = tokens[pos_id + 1];
            module.gpt(tape, model, logits, token_id, pos_id, &cache);
            module.softmax(tape, probs, logits);
            const loss_t = tape.mulK(tape.logv(probs[target_id]), -1);
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
}
