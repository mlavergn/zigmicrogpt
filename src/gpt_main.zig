// ---------------------------------------------------------------------------
// microgpt, optimized: a Zig port of microgpt.py from @karpathy that trains on
// the names in input.txt, then samples new ones. The entry point only; the
// work lives in the types re-exported by module.zig. For the whole algorithm in
// one file, read port/gpt_main.zig.
// @mlavergn
// ---------------------------------------------------------------------------

const std = @import("std");
const log = std.log.scoped(.microgptzig_main);
const mod = @import("module.zig");

/// `.info` compiles every function-entry trace out; `.debug` turns them on.
pub const std_options: std.Options = .{ .log_level = .info };

/// Trains a fresh model on the names corpus and then samples new names from it,
/// reporting progress and the generated names on stdout.
pub fn main(init: std.process.Init) !void {
    log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
    const allocator = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: std.Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    var rng: mod.Random = .{};
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
    defer docs.deinit(allocator);
    var lines = std.mem.splitScalar(u8, raw, '\n');
    while (lines.next()) |line| {
        const doc = std.mem.trim(u8, line, &std.ascii.whitespace);
        if (doc.len > 0) try docs.append(allocator, doc);
    }
    rng.shuffle([]const u8, docs.items);
    try out.print("num docs: {d}\n", .{docs.items.len});

    var tokenizer = mod.Tokenizer.init(docs.items);
    defer tokenizer.deinit();
    const vocab_size = tokenizer.vocab_size;
    try out.print("vocab size: {d}\n", .{vocab_size});

    var tape = mod.Tape.init(allocator);
    defer tape.deinit();
    var model = try mod.Model.init(allocator, &tape, &rng, vocab_size);
    defer model.deinit();
    tape.n_params = tape.nodes.items.len;
    // One allocation for the whole run: `rewind` keeps the capacity, so no step
    // ever touches the allocator again.
    try tape.reserve(mod.Model.stepNodeBound(vocab_size));
    try out.print("num params: {d}\n", .{tape.n_params});
    try out.flush();

    var trainer = try mod.Trainer.init(allocator, &tape, &model, &tokenizer);
    defer trainer.deinit();
    try trainer.run(out, docs.items);

    var sampler = try mod.Sampler.init(allocator, &tape, &model, &tokenizer, &rng);
    defer sampler.deinit();
    try sampler.run(out);
    try out.flush();
}
