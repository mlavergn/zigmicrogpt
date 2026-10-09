const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .cpu_model = .native,
        },
    });
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .fast,
    });

    const exe = b.addExecutable(.{
        .name = "microgpt",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/gpt_main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    const run_step = b.step("run", "Train the model and sample from it");
    run_step.dependOn(&run_cmd.step);

    const exe_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/module.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);
    const test_step = b.step("test", "Check parity with CPython's random module");
    test_step.dependOn(&run_exe_tests.step);

    // The straight port in src/port/ is two standalone programs sharing
    // gpt_pow.zig with the optimized build. That file sits outside their
    // directory, so it reaches them as a named module, not a relative import.
    const pow_module = b.createModule(.{ .root_source_file = b.path("src/gpt_pow.zig") });
    const port_files = [_][]const u8{ "gpt_main", "gpt_docs" };
    for (port_files) |name| {
        const port = b.addExecutable(.{
            .name = b.fmt("port_{s}", .{name}),
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("src/port/{s}.zig", .{name})),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "gpt_pow", .module = pow_module }},
            }),
        });
        // Nothing else builds the ports, so the test step compiles both.
        test_step.dependOn(&port.step);
        if (std.mem.eql(u8, name, "gpt_main")) {
            // Installed only on request, so `zig build` still ships just the optimized CLI.
            const install_port = b.addInstallArtifact(port, .{});
            b.step("port-build", "Build the straight port into zig-out/bin").dependOn(&install_port.step);
            const run_port = b.addRunArtifact(port);
            run_port.setCwd(b.path(".")); // reads input.txt from its CWD
            b.step("port", "Train and sample with the straight port").dependOn(&run_port.step);
        }
    }

    // Benchmarks always measure ReleaseFast, whatever -Drelease says, so the
    // CLIs they time are their own artifacts rather than the installed one:
    // the optimized build, and the straight port it is compared against.
    const bench_cli = b.addExecutable(.{
        .name = "microgpt-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/gpt_main.zig"),
            .target = target,
            .optimize = .fast,
        }),
    });
    const bench_port = b.addExecutable(.{
        .name = "port_gpt_main-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/port/gpt_main.zig"),
            .target = target,
            .optimize = .fast,
            .imports = &.{.{ .name = "gpt_pow", .module = pow_module }},
        }),
    });
    const bench_exe = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/gpt_bench.zig"),
            .target = target,
            .optimize = .fast,
        }),
    });
    const run_bench = b.addRunArtifact(bench_exe);
    run_bench.addArtifactArg2(bench_cli, .{});
    run_bench.addArtifactArg2(bench_port, .{});
    run_bench.setCwd(b.path(".")); // the CLI reads input.txt from its CWD
    run_bench.has_side_effects = true; // timings are never cached
    const bench_step = b.step("bench", "Time each component against its reference, then both CLIs");
    bench_step.dependOn(&run_bench.step);
}
