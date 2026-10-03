// The tokenizer, translating strings to sequences of integers ("tokens") and
// back. The corpus is ASCII, so one byte is one character.

const std = @import("std");
const log = std.log.scoped(.microgptzig_tokenizer);

/// Maps the corpus's characters to token ids and back, with one extra id for BOS.
pub const Tokenizer = struct {
    const Self = @This();
    /// A token id per byte, so the vocabulary cannot exceed this.
    const max_vocab = 256;
    /// Token id to character. The unique characters of the corpus, sorted,
    /// become token ids `0..n_uchars`.
    uchars: [max_vocab]u8 = undefined,
    /// Character to token id, the inverse of `uchars`.
    token_of: [max_vocab]u8 = [_]u8{0} ** max_vocab,
    /// How many distinct characters the corpus uses.
    n_uchars: usize = 0,
    /// Token id for a special Beginning of Sequence (BOS) token.
    bos: usize = 0,
    /// Total number of unique tokens, +1 is for BOS.
    vocab_size: usize = 0,

    /// Builds the vocabulary from every character that appears in `docs`.
    pub fn init(docs: []const []const u8) Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var self: Self = .{};
        var seen = [_]bool{false} ** max_vocab;
        for (docs) |doc| for (doc) |ch| {
            seen[ch] = true;
        };
        for (seen, 0..) |present, ch| {
            if (!present) continue;
            self.uchars[self.n_uchars] = @intCast(ch);
            self.token_of[ch] = @intCast(self.n_uchars);
            self.n_uchars += 1;
        }
        self.bos = self.n_uchars;
        self.vocab_size = self.n_uchars + 1;
        std.debug.assert(self.vocab_size <= max_vocab);
        return self;
    }

    /// Releases nothing: the tables live inside the struct.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        _ = self;
    }

    /// Writes `doc` into `tokens` surrounded by BOS on both sides, stopping
    /// early if `tokens` runs out of room. Returns how many slots were written.
    pub fn encode(self: *const Self, doc: []const u8, tokens: []usize) usize {
        var n: usize = 0;
        tokens[n] = self.bos;
        n += 1;
        for (doc) |ch| {
            // Only the first `tokens.len - 2` characters are ever trained on, so
            // anything past that can be dropped rather than stored.
            if (n + 1 >= tokens.len) break;
            tokens[n] = self.token_of[ch];
            n += 1;
        }
        tokens[n] = self.bos;
        n += 1;
        return n;
    }

    /// The character a token stands for.
    pub fn decode(self: *const Self, token_id: usize) u8 {
        return self.uchars[token_id];
    }
};
