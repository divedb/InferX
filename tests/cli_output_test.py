"""Process-level CLI contracts: exit codes, streams, help, and terminal detection."""

import os
import re
import subprocess
import sys
import tempfile
import unittest

INFERX = sys.argv.pop(1)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def environment(**overrides):
    env = os.environ.copy()
    env.pop("NO_COLOR", None)
    env["TERM"] = "xterm-256color"
    env.update(overrides)
    return env


class CliOutputTest(unittest.TestCase):
    def run_cli(self, *args, env=None):
        return subprocess.run(
            [INFERX, *args], capture_output=True, text=True,
            env=environment(**(env or {})), timeout=20,
        )

    def assert_error(self, args, *messages):
        result = self.run_cli(*args)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout, "")
        self.assertTrue(result.stderr.startswith("error:"), result.stderr)
        self.assertNotRegex(result.stderr, ANSI)
        for message in messages:
            self.assertIn(message, result.stderr)
        return result

    def test_root_help_and_no_arguments(self):
        for args in [(), ("--help",), ("-h",)]:
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stderr, "")
                for text in ["InferX", "Usage:\n  inferx <command> [options]",
                             "Commands:", "Options:", "Examples:", "version",
                             "serve", "bench", "diagnostic"]:
                    self.assertIn(text, result.stdout)
                self.assertNotIn("<help>", result.stdout)
                self.assertNotIn("<version>", result.stdout)
                self.assertNotRegex(result.stdout, ANSI)

    def test_nested_help_paths(self):
        for path in ["serve", "bench", "bench latency", "bench throughput",
                     "bench workload", "diagnostic replay-logits", "launch render"]:
            with self.subTest(path=path):
                result = self.run_cli(*path.split(), "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, "")
                self.assertIn("Usage:\n  inferx " + path, result.stdout)
                self.assertIn("-h, --help", result.stdout)
                self.assertNotIn("--color", result.stdout)

    def test_option_groups_and_defaults(self):
        text = self.run_cli("serve", "--help").stdout
        for value in ["Engine:", "Sampling:", "--model <model>", "--port <port>",
                      "[default: 8000]", "--cuda-graphs"]:
            self.assertIn(value, text)
        self.assertNotIn("--cuda-graphs <", text)

    def test_vllm_parity_options_in_help(self):
        text = self.run_cli("bench", "latency", "--help").stdout
        for value in ["--input-len", "--num-iters", "--num-iters-warmup",
                      "[default: 30]", "[default: 10]", "--output-json",
                      "--disable-log-stats", "--dtype", "--seed",
                      "--max-model-len", "--kv-cache-memory-bytes",
                      "--enable-chunked-prefill", "--no-enable-chunked-prefill",
                      "--cudagraph-capture-sizes",
                      "--max-cudagraph-capture-size"]:
            self.assertIn(value, text)
        for path in [["bench", "throughput"], ["bench", "workload"]]:
            text = self.run_cli(*path, "--help").stdout
            for value in ["--output-json", "--disable-log-stats", "--dtype"]:
                self.assertIn(value, text)
        # --seed moved to the engine group; it must not appear twice.
        text = self.run_cli("bench", "throughput", "--help").stdout
        self.assertEqual(text.count("--seed"), 1)

    def test_vllm_parity_option_validation(self):
        self.assert_error(["serve", "--dtype", "fp8"], "invalid value 'fp8'", "--dtype")
        self.assert_error(["bench", "latency", "--cudagraph-capture-sizes", "0"],
                          "invalid value", "--cudagraph-capture-sizes")
        self.assert_error(["serve", "--max-model-len", "-1"], "invalid value")
        self.assert_error(["serve", "--enable-chunked-prefill",
                           "--no-enable-chunked-prefill"],
                          "--enable-chunked-prefill", "--no-enable-chunked-prefill")

    def test_version(self):
        result = self.run_cli("version")
        self.assertEqual(result.returncode, 0)
        self.assertRegex(result.stdout, r"^\d+\.\d+\.\d+")
        self.assertEqual(result.stderr, "")
        # The former flags are gone.
        for args in [("--version",), ("-v",), ("serve", "--version")]:
            self.assert_error(list(args), "unknown option")

    def test_missing_subcommands(self):
        for group in ["bench", "launch", "diagnostic"]:
            self.assert_error([group], "missing required command <command>",
                              "Available commands:", f"inferx {group} --help")

    def test_unknown_commands_and_suggestions(self):
        self.assert_error(["serv"], "unknown command 'serv'", "Did you mean 'serve'?")
        self.assert_error(["bench", "latncy"], "unknown command 'latncy'",
                          "Did you mean 'latency'?", "inferx bench --help")
        result = self.assert_error(["completely-unrelated"], "unknown command")
        self.assertNotIn("Did you mean", result.stderr)

    def test_unknown_options(self):
        for args in [["serve", "--modle", "foo"], ["serve", "--modle=foo"]]:
            self.assert_error(args, "unknown option '--modle'", "Did you mean '--model'?",
                              "inferx serve --help")
        self.assert_error(["bench", "latency", "--batch-szie", "4"],
                          "unknown option '--batch-szie'", "Did you mean '--batch-size'?",
                          "inferx bench latency --help")
        self.assert_error(["serve", "-z"], "unknown option '-z'")

    def test_required_options(self):
        self.assert_error(["diagnostic", "replay-logits"],
                          "missing required option '--fixture'", "<fixture>",
                          "inferx diagnostic replay-logits --help")
        self.assert_error(["diagnostic", "replay-logits", "--fixture", "fixture.json"],
                          "missing required option '--output'")

    def test_missing_values(self):
        for option in ["--model", "--port"]:
            self.assert_error(["serve", option], "missing value", f"for '{option}'")
        self.assert_error(["serve", "--model", "--port", "8000"], "missing value", "--model")

    def test_invalid_values(self):
        for value in ["0", "70000", "abc"]:
            self.assert_error(["serve", "--port", value], "invalid value", "--port")
        self.assert_error(["serve", "--device", "tpu"], "invalid value 'tpu'", "--device", "cuda")
        # An option without a validator exercises CLI11's ConversionError.
        self.assert_error(["bench", "workload", "--repeats", "oops"], "invalid value")

    def test_runtime_errors(self):
        result = self.assert_error(["collect-env"], "not implemented")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("UNIMPLEMENTED:", result.stderr)

    def test_plain_and_colored_layouts_match(self):
        for args in [["--help"], ["serve", "--help"], ["bench"], ["serve", "--modle"]]:
            plain = self.run_cli(*args, env={"NO_COLOR": "1"})
            color = self.run_cli(*args, env={"CLICOLOR_FORCE": "1"})
            self.assertEqual(plain.returncode, color.returncode)
            self.assertRegex(color.stdout + color.stderr, ANSI)
            self.assertEqual(ANSI.sub("", color.stdout), plain.stdout)
            self.assertEqual(ANSI.sub("", color.stderr), plain.stderr)

    def test_color_environment_control(self):
        # CLICOLOR_FORCE colors redirected output; NO_COLOR disables it and
        # wins over CLICOLOR_FORCE.
        result = self.run_cli("--help", env={"NO_COLOR": "1"})
        self.assertNotRegex(result.stdout, ANSI)
        result = self.run_cli("bench", "latency", "--help", env={"CLICOLOR_FORCE": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, ANSI)
        result = self.run_cli("--help", env={"CLICOLOR_FORCE": "1", "NO_COLOR": "1"})
        self.assertNotRegex(result.stdout, ANSI)

    def test_semantic_styles_in_colored_output(self):
        # Placeholders render yellow, option names and commands cyan.
        result = self.run_cli("serve", "--model", "--port", "8000",
                              env={"CLICOLOR_FORCE": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("\x1b[33m<model>\x1b[0m", result.stderr)
        self.assertIn("\x1b[36m--model\x1b[0m", result.stderr)
        result = self.run_cli("bench", "latency", "--help", env={"CLICOLOR_FORCE": "1"})
        self.assertEqual(result.returncode, 0)
        self.assertIn("\x1b[33m<batch-size>\x1b[0m", result.stdout)
        self.assertIn("\x1b[1mUsage:\x1b[0m", result.stdout)
        self.assertIn("\x1b[36minferx bench latency\x1b[0m", result.stdout)
        # The same invocations without forcing stay plain when piped.
        self.assertNotIn("\x1b[33m<model>\x1b[0m",
                         self.run_cli("serve", "--model", "--port", "8000").stderr)

    def test_redirect_to_file(self):
        for env, colored in [({}, False), ({"NO_COLOR": "1"}, False),
                             ({"CLICOLOR_FORCE": "1"}, True)]:
            with tempfile.TemporaryFile(mode="w+") as output:
                result = subprocess.run([INFERX, "--help"],
                                        stdout=output, stderr=subprocess.PIPE,
                                        env=environment(**env), timeout=20)
                output.seek(0)
                self.assertEqual(result.returncode, 0)
                self.assertEqual(bool(ANSI.search(output.read())), colored)

    @unittest.skipUnless(os.name == "posix", "requires a pseudo-terminal")
    def test_interactive_streams_independently(self):
        import pty
        # Keep output small enough to fit a PTY buffer before reading it.
        for tty_stream, args, env, colored in [
            ("stdout", ["bench", "--help"], {}, True),
            ("stdout", ["bench", "--help"], {"NO_COLOR": "1"}, False),
            ("stdout", ["bench", "--help"], {"NO_COLOR": ""}, True),
            ("stdout", ["bench", "--help"], {"TERM": "dumb"}, False),
            ("stdout", ["bench", "--help"], {"TERM": "dumb", "CLICOLOR_FORCE": "1"}, True),
            ("stderr", ["bench"], {}, True),
            ("stderr", ["bench"], {"NO_COLOR": "1"}, False),
            ("stdout", ["bench"], {}, False),  # stderr is a pipe
        ]:
            with self.subTest(stream=tty_stream, args=args, env=env):
                master, slave = pty.openpty()
                try:
                    result = subprocess.run(
                        [INFERX, *args], env=environment(**env), timeout=20,
                        stdout=slave if tty_stream == "stdout" else subprocess.PIPE,
                        stderr=slave if tty_stream == "stderr" else subprocess.PIPE,
                    )
                    os.close(slave)
                    slave = None
                    data = bytearray()
                    while True:
                        try:
                            chunk = os.read(master, 4096)
                        except OSError:
                            break
                        if not chunk:
                            break
                        data.extend(chunk)
                    text = data.decode() + (result.stdout or b"").decode() + (result.stderr or b"").decode()
                    self.assertEqual(bool(ANSI.search(text)), colored, text)
                finally:
                    os.close(master)
                    if slave is not None:
                        os.close(slave)


if __name__ == "__main__":
    unittest.main()
