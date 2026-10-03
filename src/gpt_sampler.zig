// ---------------------------------------------------------------------------
// Inference: may the model babble back to us. One token is sampled at a time
// and fed back in, until the model emits BOS or the context window fills.
// ---------------------------------------------------------------------------

const std = @import("std");
const log = std.log.scoped(.microgptzig_sampler);
const mod = @import("module.zig");

/// Generates new documents, one token at a time, from a trained model.
pub const Sampler = struct {
    const Self = @This();

    pub const num_samples = 20; // how many names to hallucinate
    const temperature = 0.5; // in (0, 1], control the "creativity" of generated text, low to high

    allocator: std.mem.Allocator,
    tape: *mod.Tape,
    model: *const mod.Model,
    tokenizer: *const mod.Tokenizer,
    rng: *mod.Random,
    // Buffers
    logits: []u32 = &.{},
    probs: []u32 = &.{},
    weights: []f64 = &.{},
    cum: []f64 = &.{},

    /// Creates a sampler for a trained `model`, drawing tokens from `rng`.
    pub fn init(allocator: std.mem.Allocator, tape: *mod.Tape, model: *const mod.Model, tokenizer: *const mod.Tokenizer, rng: *mod.Random) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var self: Self = .{ .allocator = allocator, .tape = tape, .model = model, .tokenizer = tokenizer, .rng = rng };
        errdefer self.deinit();
        self.logits = try allocator.alloc(u32, tokenizer.vocab_size);
        self.probs = try allocator.alloc(u32, tokenizer.vocab_size);
        self.weights = try allocator.alloc(f64, tokenizer.vocab_size);
        self.cum = try allocator.alloc(f64, tokenizer.vocab_size);
        return self;
    }

    /// Releases the scratch buffers, in reverse order of `init`.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.allocator.free(self.cum);
        self.allocator.free(self.weights);
        self.allocator.free(self.probs);
        self.allocator.free(self.logits);
    }

    /// Samples `num_samples` names and prints them to `out`.
    pub fn run(self: *Self, out: *std.Io.Writer) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const tape = self.tape;
        const tokenizer = self.tokenizer;
        const inv_temperature = mod.Pow.pow(temperature, -1);
        try out.print("\n--- inference (new, hallucinated names) ---\n", .{});

        for (0..num_samples) |sample_idx| {
            tape.rewind();
            var cache: mod.Cache = .{};
            var token_id: usize = tokenizer.bos;
            var sample: [mod.Model.block_size]u8 = undefined;
            var sample_len: usize = 0;
            for (0..mod.Model.block_size) |pos_id| {
                self.model.gpt(tape, self.logits, token_id, pos_id, &cache);
                for (self.logits) |*l| l.* = tape.mulK(l.*, inv_temperature);
                mod.Model.softmax(tape, self.probs, self.logits);
                for (self.weights, self.probs) |*w, p| w.* = tape.data(p);
                token_id = self.rng.choices(self.weights, self.cum);
                if (token_id == tokenizer.bos) break;
                sample[sample_len] = tokenizer.decode(token_id);
                sample_len += 1;
            }
            try out.print("sample {d:>2}: {s}\n", .{ sample_idx + 1, sample[0..sample_len] });
        }
        try out.flush();
    }
};
