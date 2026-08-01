// Inference: may the model babble back to us. One token is sampled at a time
// and fed back in, until the model emits BOS or the context window fills.

const std = @import("std");
const module = @import("gpt_main.zig");
const Random = @import("gpt_random.zig").Random;

const Value = module.Value;
const Model = module.Model;
const Cache = module.Cache;
const block_size = module.block_size;
const Tokenizer = @import("gpt_tokenizer.zig").Tokenizer;

pub const num_samples = 20; // how many names to hallucinate
const temperature = 0.5; // in (0, 1], control the "creativity" of generated text, low to high

/// Samples `num_samples` names from a trained `model` and prints them.
/// `logits` and `probs` are caller-owned scratch, one slot per vocabulary entry.
pub fn run(tape: *Value, model: Model, out: *std.Io.Writer, arena: std.mem.Allocator, rng: *Random, tok: *const Tokenizer, logits: []u32, probs: []u32) !void {
    const inv_temperature = std.math.pow(f64, temperature, -1);
    try out.print("\n--- inference (new, hallucinated names) ---\n", .{});
    const weights = try arena.alloc(f64, tok.vocab_size);
    const cum = try arena.alloc(f64, tok.vocab_size);

    for (0..num_samples) |sample_idx| {
        tape.rewind();
        var cache: Cache = .{};
        var token_id: usize = tok.bos;
        var sample: [block_size]u8 = undefined;
        var sample_len: usize = 0;
        for (0..block_size) |pos_id| {
            module.gpt(tape, model, logits, token_id, pos_id, &cache);
            for (logits) |*l| l.* = tape.mulK(l.*, inv_temperature);
            module.softmax(tape, probs, logits);
            for (weights, probs) |*w, p| w.* = tape.data(p);
            token_id = rng.choices(weights, cum);
            if (token_id == tok.bos) break;
            sample[sample_len] = tok.decode(token_id);
            sample_len += 1;
        }
        try out.print("sample {d:>2}: {s}\n", .{ sample_idx + 1, sample[0..sample_len] });
    }
    try out.flush();
}
