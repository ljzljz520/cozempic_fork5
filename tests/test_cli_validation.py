"""Tests for CLI argument validation (BMAD R4-12)."""

from __future__ import annotations

import argparse
import io
import contextlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from cozempic.cli import _prescan_argv, _positive_float, build_parser, _digest_session


class TestPositiveFloatArgparseHelper:
    """Direct unit tests for cli._positive_float argparse type= helper.

    NaN/inf bypass: float("nan") and float("inf") both pass `f <= 0` (False
    in IEEE 754), so the validator silently returns nan/inf.
    RED at base: _positive_float("nan") returns nan instead of raising.
    RED at base: _positive_float("inf") returns inf instead of raising.
    These are bug-capture tests — must FAIL against the unmodified source.
    """

    def test_positive_float_rejects_nan(self):
        """'nan' → float('nan') → bypasses <= 0 → silently returned.
        RED at base: no exception raised.
        Also asserts the specific 'finite' discriminating word — ensures the NaN/inf
        guard fires (not the generic type-guard or positivity check)."""
        try:
            result = _positive_float("nan")
            raise AssertionError(
                f"Expected ArgumentTypeError but _positive_float returned {result!r}"
            )
        except argparse.ArgumentTypeError as e:
            assert "finite" in str(e), (
                f"expected 'finite' in error message, got: {e!r}"
            )

    def test_positive_float_rejects_inf(self):
        """'inf' → float('inf') → bypasses <= 0 → silently returned.
        RED at base: no exception raised.
        Also asserts the specific 'finite' discriminating word."""
        try:
            result = _positive_float("inf")
            raise AssertionError(
                f"Expected ArgumentTypeError but _positive_float returned {result!r}"
            )
        except argparse.ArgumentTypeError as e:
            assert "finite" in str(e), (
                f"expected 'finite' in error message, got: {e!r}"
            )

    def test_positive_float_rejects_negative_inf(self):
        """-inf is already caught by `f <= 0` at base (-inf <= 0 is True).
        GREEN at base — regression guard, not a RED/bug-capture test."""
        try:
            result = _positive_float("-inf")
            raise AssertionError(
                f"Expected ArgumentTypeError but _positive_float returned {result!r}"
            )
        except argparse.ArgumentTypeError:
            pass  # expected

    def test_positive_float_accepts_valid(self):
        """Positive finite float must still be accepted after the fix."""
        assert _positive_float("50.5") == 50.5

    def test_positive_float_accepts_int_string(self):
        """'50' (int string) is valid — user may write --threshold 50."""
        assert _positive_float("50") == 50.0


class TestPrescanArgvValidation:
    def test_invalid_context_window_ignored(self):
        """Non-numeric --context-window is ignored with a warning."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COZEMPIC_CONTEXT_WINDOW", None)
            _prescan_argv(["treat", "current", "--context-window", "abc"])
            assert "COZEMPIC_CONTEXT_WINDOW" not in os.environ

    def test_negative_context_window_ignored(self):
        """Negative --context-window is ignored."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COZEMPIC_CONTEXT_WINDOW", None)
            _prescan_argv(["treat", "current", "--context-window", "-500"])
            assert "COZEMPIC_CONTEXT_WINDOW" not in os.environ

    def test_zero_context_window_ignored(self):
        """Zero --context-window is ignored."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COZEMPIC_CONTEXT_WINDOW", None)
            _prescan_argv(["treat", "current", "--context-window", "0"])
            assert "COZEMPIC_CONTEXT_WINDOW" not in os.environ

    def test_valid_context_window_set(self):
        """Valid positive --context-window is accepted."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COZEMPIC_CONTEXT_WINDOW", None)
            _prescan_argv(["treat", "current", "--context-window", "1000000"])
            assert os.environ["COZEMPIC_CONTEXT_WINDOW"] == "1000000"
            os.environ.pop("COZEMPIC_CONTEXT_WINDOW", None)

    def test_invalid_system_overhead_tokens_ignored(self):
        """Non-numeric --system-overhead-tokens is ignored."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COZEMPIC_SYSTEM_OVERHEAD_TOKENS", None)
            _prescan_argv(["treat", "current", "--system-overhead-tokens", "xyz"])
            assert "COZEMPIC_SYSTEM_OVERHEAD_TOKENS" not in os.environ

    def test_valid_system_overhead_tokens_set(self):
        """Valid positive --system-overhead-tokens is accepted."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COZEMPIC_SYSTEM_OVERHEAD_TOKENS", None)
            _prescan_argv(["treat", "current", "--system-overhead-tokens", "25000"])
            assert os.environ["COZEMPIC_SYSTEM_OVERHEAD_TOKENS"] == "25000"
            os.environ.pop("COZEMPIC_SYSTEM_OVERHEAD_TOKENS", None)

    def test_invalid_context_window_equals_form_ignored(self):
        """--context-window=abc (equals form) is ignored."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COZEMPIC_CONTEXT_WINDOW", None)
            _prescan_argv(["treat", "current", "--context-window=notanumber"])
            assert "COZEMPIC_CONTEXT_WINDOW" not in os.environ


class TestGuardArgparseValidation:
    """`cozempic guard` numeric flags must reject nonsensical values at
    parse time rather than silently triggering a reload storm or crashing
    deep in `time.sleep(interval)` on the first cycle."""

    def _parse(self, argv):
        parser = build_parser()
        return parser.parse_args(argv)

    def _assert_argparse_rejects(self, argv, match_stderr=None):
        import io
        import contextlib
        parser = build_parser()
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            try:
                parser.parse_args(argv)
            except SystemExit as e:
                assert e.code == 2, f"expected exit code 2, got {e.code}"
                if match_stderr:
                    assert match_stderr in buf.getvalue(), (
                        f"expected {match_stderr!r} in stderr, got: {buf.getvalue()!r}"
                    )
                return
        raise AssertionError(f"argparse accepted {argv!r} but should have rejected")

    def test_threshold_rejects_zero(self):
        self._assert_argparse_rejects(
            ["guard", "--threshold", "0"], match_stderr="positive"
        )

    def test_threshold_rejects_negative(self):
        self._assert_argparse_rejects(
            ["guard", "--threshold", "-1"], match_stderr="positive"
        )

    def test_threshold_rejects_non_numeric(self):
        self._assert_argparse_rejects(["guard", "--threshold", "abc"])

    def test_threshold_accepts_valid_float(self):
        args = self._parse(["guard", "--threshold", "50.5"])
        assert args.threshold == 50.5

    def test_threshold_accepts_int(self):
        """User may write `--threshold 50` (int) — accept."""
        args = self._parse(["guard", "--threshold", "50"])
        assert args.threshold == 50.0

    def test_soft_threshold_rejects_zero(self):
        self._assert_argparse_rejects(["guard", "--soft-threshold", "0"])

    def test_soft_threshold_rejects_negative(self):
        self._assert_argparse_rejects(["guard", "--soft-threshold", "-1"])

    def test_interval_rejects_zero(self):
        """interval=0 → spin loop (guard cycles with no pause)."""
        self._assert_argparse_rejects(
            ["guard", "--interval", "0"], match_stderr="positive"
        )

    def test_interval_rejects_negative(self):
        """interval=-1 → ValueError from time.sleep(-1) mid-daemon."""
        self._assert_argparse_rejects(["guard", "--interval", "-1"])

    def test_interval_accepts_valid(self):
        args = self._parse(["guard", "--interval", "30"])
        assert args.interval == 30

    def test_threshold_tokens_rejects_zero(self):
        self._assert_argparse_rejects(["guard", "--threshold-tokens", "0"])

    def test_threshold_tokens_rejects_negative(self):
        self._assert_argparse_rejects(["guard", "--threshold-tokens", "-100"])

    def test_soft_threshold_tokens_rejects_zero(self):
        self._assert_argparse_rejects(["guard", "--soft-threshold-tokens", "0"])

    def test_soft_threshold_tokens_rejects_negative(self):
        self._assert_argparse_rejects(["guard", "--soft-threshold-tokens", "-1"])

    def test_remind_interval_rejects_zero(self):
        """Separate `--interval` on `cozempic remind` — same validation."""
        self._assert_argparse_rejects(["remind", "--interval", "0"])

    def test_remind_interval_rejects_negative(self):
        self._assert_argparse_rejects(["remind", "--interval", "-5"])


def _assert_raises_like(exc_type, substr):
    """Context manager: assert exc_type is raised and substr appears in str(e).

    Module-level helper shared by TestStartGuardOrderingValidation,
    TestStartGuardNanInfValidation, and TestReloadSelfDaemonNanInfValidation.
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        try:
            yield
        except exc_type as e:
            assert substr in str(e), f"expected {substr!r} in {e!r}"
            return
        raise AssertionError(f"expected {exc_type.__name__}, nothing raised")

    return _ctx()


class TestStartGuardOrderingValidation:
    """After argparse, soft thresholds may be resolved from defaults
    (60% of threshold). The soft < hard invariant must hold at that point
    too — checked inside start_guard, not just at argparse."""

    def setup_method(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp(prefix="_cozempic_ordering_test_")

    def teardown_method(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _call_start_guard(self, **kwargs):
        from cozempic.guard import start_guard
        return start_guard(**kwargs)

    def test_soft_mb_equal_hard_mb_rejected(self):
        from cozempic._validation import ConfigError
        with _assert_raises_like(ConfigError, "strictly less"):
            self._call_start_guard(
                threshold_mb=50.0,
                soft_threshold_mb=50.0,
                cwd=self._tmpdir,
            )

    def test_soft_mb_greater_than_hard_mb_rejected(self):
        from cozempic._validation import ConfigError
        with _assert_raises_like(ConfigError, "strictly less"):
            self._call_start_guard(
                threshold_mb=50.0,
                soft_threshold_mb=100.0,
                cwd=self._tmpdir,
            )

    def test_soft_tokens_greater_than_threshold_tokens_rejected(self):
        from cozempic._validation import ConfigError
        with _assert_raises_like(ConfigError, "strictly less"):
            self._call_start_guard(
                threshold_mb=50.0,
                threshold_tokens=10_000,
                soft_threshold_tokens=20_000,
                cwd=self._tmpdir,
            )


class TestStartGuardNanInfValidation:
    """NaN/inf bypass in guard.py belt-and-braces validators (H-1 fold).

    start_guard and start_guard_daemon both validate thresholds before any
    subprocess spawn, so direct-Python callers passing NaN/inf are the audience.
    Argparse (now fixed via P0-B) only covers the CLI string path; these tests
    cover the direct-Python-caller contract documented in start_guard's docstring.

    RED at base: NaN/inf bypass `<= 0` (IEEE 754 semantics), no ConfigError raised.
    These are bug-capture tests — must FAIL against the unmodified source.

    Isolation (per isolate-subprocess-tests-by-design memory):
      start_guard:        validation fires before any I/O (before find_current_session).
                          A tempdir is still created and cleaned up in teardown to
                          defend against future code reorderings.
      start_guard_daemon: ALL I/O primitives between function entry and Popen are mocked
                          to prevent any /tmp/cozempic_guard_* file creation at base:
                            _cleanup_legacy_pid      (legacy pid cleanup)
                            _reload_sentinel_active  (sentinel file read)
                            find_current_session     (jsonl directory scan)
                            _is_guard_running_for_session (pid file read)
                            _guard_tmp_root          (redirects pid/log path construction)
                            cozempic.spawn_lock.DaemonSpawnClaim (O_CREAT|O_EXCL on .pid)
                            subprocess.Popen         (daemon spawn)
                          Teardown removes the tempdir unconditionally.
                          Acceptance: BEFORE/AFTER ls /tmp/cozempic_guard_* count equal
                          when daemon tests run against unpatched (base) source.
    """

    def setup_method(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp(prefix="_cozempic_guard_nan_test_")

    def teardown_method(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_start_guard_threshold_mb_nan_raises_config_error(self):
        """start_guard(threshold_mb=float('nan')) must raise ConfigError with 'finite'.
        RED at base: nan <= 0 is False, validation bypassed; ValueError raised downstream
        at int(nan * 1024 * 1024) instead of ConfigError at the validation block."""
        from cozempic.guard import start_guard
        from cozempic._validation import ConfigError
        with _assert_raises_like(ConfigError, "finite"):
            start_guard(threshold_mb=float("nan"), cwd=self._tmpdir)

    def test_start_guard_threshold_mb_inf_raises_config_error(self):
        """start_guard(threshold_mb=float('inf')) must raise ConfigError with 'finite'.
        RED at base: inf <= 0 is False; OverflowError raised downstream at int(inf)
        instead of ConfigError at the validation block."""
        from cozempic.guard import start_guard
        from cozempic._validation import ConfigError
        with _assert_raises_like(ConfigError, "finite"):
            start_guard(threshold_mb=float("inf"), cwd=self._tmpdir)

    def test_start_guard_threshold_mb_huge_int_raises_config_error(self):
        """start_guard(threshold_mb=10**400) must raise ConfigError with 'finite'.

        RED at base: isinstance gate skips ints entirely; 10**400 passes the float
        guard, then `round(threshold_mb * 0.6, 1)` (the soft-threshold default) does
        an int*float multiply that raises a bare OverflowError downstream instead of
        ConfigError at the validation block. Same class of bug as P0-F
        (coerce_positive_float). Fix: widen the gate to isinstance(_v, (int, float))
        so huge ints enter the try/except OverflowError path."""
        from cozempic.guard import start_guard
        from cozempic._validation import ConfigError
        with _assert_raises_like(ConfigError, "finite"):
            start_guard(threshold_mb=10**400, cwd=self._tmpdir)

    def test_start_guard_daemon_threshold_mb_nan_raises_config_error(self):
        """start_guard_daemon(threshold_mb=float('nan')) must raise ConfigError before spawn.
        RED at base: nan <= 0 is False, validation silently passes; execution reaches
        DaemonSpawnClaim which creates /tmp/cozempic_guard_*.pid via O_CREAT|O_EXCL,
        and open(log_file) creates /tmp/cozempic_guard_*.log — two real /tmp artifacts.

        All I/O primitives between function entry and Popen are mocked to guarantee
        zero /tmp/cozempic_guard_* files even when running against the unpatched source:
          - _cleanup_legacy_pid      (legacy pid cleanup)
          - _reload_sentinel_active  (sentinel file read — only fires when session_id set)
          - find_current_session     (jsonl directory scan — fires when no session_id)
          - _is_guard_running_for_session (pid file read)
          - _guard_tmp_root          (redirects pid/log path construction to self._tmpdir)
          - cozempic.spawn_lock.DaemonSpawnClaim (O_CREAT|O_EXCL on .pid — the leak source)
          - subprocess.Popen         (daemon spawn — must never be reached)
        """
        from cozempic.guard import start_guard_daemon
        from cozempic._validation import ConfigError
        with (
            patch("cozempic.guard._cleanup_legacy_pid"),
            patch("cozempic.guard._reload_sentinel_active", return_value=False),
            patch("cozempic.guard.find_current_session", return_value=None),
            patch("cozempic.guard._is_guard_running_for_session", return_value=None),
            patch("cozempic.guard._guard_tmp_root", return_value=Path(self._tmpdir)),
            patch("cozempic.spawn_lock.DaemonSpawnClaim"),
            patch("cozempic.guard.subprocess.Popen"),
        ):
            with _assert_raises_like(ConfigError, "finite"):
                start_guard_daemon(
                    threshold_mb=float("nan"), cwd=self._tmpdir
                )

    def test_start_guard_daemon_threshold_mb_inf_raises_config_error(self):
        """start_guard_daemon(threshold_mb=float('inf')) must raise ConfigError before spawn.
        RED at base: inf <= 0 is False; same leak pattern as nan (two /tmp artifacts).

        All I/O primitives mocked — see test_start_guard_daemon_threshold_mb_nan for details.
        """
        from cozempic.guard import start_guard_daemon
        from cozempic._validation import ConfigError
        with (
            patch("cozempic.guard._cleanup_legacy_pid"),
            patch("cozempic.guard._reload_sentinel_active", return_value=False),
            patch("cozempic.guard.find_current_session", return_value=None),
            patch("cozempic.guard._is_guard_running_for_session", return_value=None),
            patch("cozempic.guard._guard_tmp_root", return_value=Path(self._tmpdir)),
            patch("cozempic.spawn_lock.DaemonSpawnClaim"),
            patch("cozempic.guard.subprocess.Popen"),
        ):
            with _assert_raises_like(ConfigError, "finite"):
                start_guard_daemon(
                    threshold_mb=float("inf"), cwd=self._tmpdir
                )


class TestReloadSelfDaemonNanInfValidation:
    """NaN/inf bypass in reload_self_daemon — the critical third entry point (P0-D).

    reload_self_daemon SIGTERMs/SIGKILLs the old daemon THEN calls
    start_guard_daemon. Without a finite check BEFORE the kill, passing
    threshold_mb=float('nan') kills the live daemon and then raises
    ConfigError (from start_guard_daemon's validator), leaving the session
    completely unprotected.

    Regression guard: _validate_finite_thresholds() is called at the top of
    reload_self_daemon before any I/O, so a NaN/inf threshold_mb raises
    ConfigError before os.kill is reached — the live daemon is never orphaned.

    All I/O between entry and the kill is mocked per isolate-subprocess-tests-by-design.
    """

    def _make_mocks(self):
        """Return a context manager that mocks all I/O in reload_self_daemon
        between the function entry and the os.kill call (inclusive).

        Mocked:
          - _is_guard_running_for_session → 99999 (fake PID — daemon appears running)
          - _is_cozempic_guard_process     → True  (PID identity verified)
          - os.kill                         → recorded, must NOT be called on nan/inf
          - _wait_for_exit                  → True  (clean exit on SIGTERM)
          - _pid_file_points_to             → False (CAS: leave pid file alone)
          - _pid_file_for_session           → MagicMock (avoid real filesystem)
          - start_guard_daemon              → {"started": True, "pid": 88888}
        """
        import contextlib
        from unittest.mock import MagicMock, patch, call

        @contextlib.contextmanager
        def ctx():
            with (
                patch("cozempic.guard._is_guard_running_for_session", return_value=99999),
                patch("cozempic.guard._is_cozempic_guard_process", return_value=True),
                patch("cozempic.guard.os.kill") as mock_kill,
                patch("cozempic.guard._wait_for_exit", return_value=True),
                patch("cozempic.guard._pid_file_points_to", return_value=False),
                patch("cozempic.guard._pid_file_for_session", return_value=MagicMock()),
                patch("cozempic.guard.start_guard_daemon",
                      return_value={"started": True, "pid": 88888, "log_file": "/tmp/x.log"}),
            ):
                yield mock_kill

        return ctx()

    def test_reload_self_daemon_threshold_mb_nan_raises_before_kill(self):
        """reload_self_daemon(threshold_mb=nan) must raise ConfigError BEFORE os.kill.

        Regression guard: _validate_finite_thresholds() fires at entry, ConfigError
        is raised before os.kill — the live daemon (99999) is never orphaned."""
        from cozempic.guard import reload_self_daemon
        from cozempic._validation import ConfigError
        with self._make_mocks() as mock_kill:
            with _assert_raises_like(ConfigError, "finite"):
                reload_self_daemon(
                    threshold_mb=float("nan"),
                    session_id="00000000-0000-0000-0000-000000000001",
                )
        assert not mock_kill.called, (
            "os.kill was called before ConfigError — daemon was killed with an invalid config"
        )

    def test_reload_self_daemon_threshold_mb_inf_raises_before_kill(self):
        """reload_self_daemon(threshold_mb=inf) must raise ConfigError BEFORE os.kill.

        Regression guard: same orphan-daemon scenario with inf — ConfigError
        fires before os.kill, daemon survives."""
        from cozempic.guard import reload_self_daemon
        from cozempic._validation import ConfigError
        with self._make_mocks() as mock_kill:
            with _assert_raises_like(ConfigError, "finite"):
                reload_self_daemon(
                    threshold_mb=float("inf"),
                    session_id="00000000-0000-0000-0000-000000000001",
                )
        assert not mock_kill.called, (
            "os.kill was called before ConfigError — daemon was killed with an invalid config"
        )


class TestReloadSessionFlag:
    """Reload must accept --session as an escape hatch when auto-detect fails."""

    def test_reload_accepts_session_flag(self):
        """reload --session <uuid> must parse without error."""
        parser = build_parser()
        args = parser.parse_args(["reload", "--session", "9a1256d9-639f-44ca-aada-dc61bf5c3986"])
        assert args.command == "reload"
        assert args.session == "9a1256d9-639f-44ca-aada-dc61bf5c3986"

    def test_reload_accepts_session_and_rx(self):
        """reload --session <id> -rx aggressive must parse together."""
        parser = build_parser()
        args = parser.parse_args(["reload", "--session", "abc123", "-rx", "aggressive"])
        assert args.session == "abc123"
        assert args.rx == "aggressive"

    def test_reload_rejects_positional_session(self):
        """Positional session ID is NOT accepted — flag only, to keep the API explicit."""
        parser = build_parser()
        try:
            parser.parse_args(["reload", "-rx", "aggressive", "9a1256d9-639f-44ca-aada-dc61bf5c3986"])
            assert False, "Expected SystemExit"
        except SystemExit:
            pass  # argparse rejects unknown positional — expected

    def test_reload_session_optional(self):
        """reload without --session still parses (uses auto-detect)."""
        parser = build_parser()
        args = parser.parse_args(["reload", "-rx", "standard"])
        assert args.session is None
        assert args.rx == "standard"


# ---------------------------------------------------------------------------
# F15: ValueError dispatch in main() — cozempic.cli lines 1708-1719
# ---------------------------------------------------------------------------

def _run_main_with_argv(argv):
    """Run main() with stubbed-out side-effects (updater, init hooks).

    Returns (stdout_str, stderr_str).  Raises SystemExit transparently.
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    with patch("sys.argv", ["cozempic"] + argv), \
         patch("cozempic.cli._maybe_global_init"), \
         patch("cozempic.cli._maybe_auto_init"), \
         patch("cozempic.updater.ping_install_if_new"), \
         patch("cozempic.updater.maybe_auto_update"), \
         patch("sys.stdout", stdout_buf), \
         patch("sys.stderr", stderr_buf):
        from cozempic.cli import main
        main()

    return stdout_buf.getvalue(), stderr_buf.getvalue()


class TestValueErrorDispatch:
    """main() try/except ValueError must:
      - guard / reload + ValueError → exit 2 + "Error:" on stderr
      - other command + ValueError → re-raise (not silently swallowed)
    """

    def test_guard_valueerror_exits_2_with_error_stderr(self):
        """guard command raising ValueError → SystemExit(2) + 'Error:' on stderr."""
        with patch("cozempic.cli.cmd_guard", side_effect=ValueError("bad session id")):
            try:
                _run_main_with_argv(["guard"])
                assert False, "expected SystemExit"
            except SystemExit as exc:
                assert exc.code == 2, f"expected code 2, got {exc.code}"

    def test_guard_valueerror_error_on_stderr(self):
        """guard ValueError message must appear on stderr prefixed with 'Error:'."""
        stderr_buf = io.StringIO()
        with patch("cozempic.cli.cmd_guard", side_effect=ValueError("bad session id")), \
             patch("sys.argv", ["cozempic", "guard"]), \
             patch("cozempic.cli._maybe_global_init"), \
             patch("cozempic.cli._maybe_auto_init"), \
             patch("cozempic.updater.ping_install_if_new"), \
             patch("cozempic.updater.maybe_auto_update"), \
             patch("sys.stderr", stderr_buf):
            from cozempic.cli import main
            try:
                main()
            except SystemExit:
                pass
        assert "Error:" in stderr_buf.getvalue(), (
            f"expected 'Error:' in stderr, got: {stderr_buf.getvalue()!r}"
        )

    def test_reload_valueerror_exits_2_with_error_stderr(self):
        """reload command raising ValueError → SystemExit(2)."""
        with patch("cozempic.cli.cmd_reload", side_effect=ValueError("malformed")):
            try:
                _run_main_with_argv(["reload"])
                assert False, "expected SystemExit"
            except SystemExit as exc:
                assert exc.code == 2, f"expected code 2, got {exc.code}"

    def test_non_guard_valueerror_propagates(self):
        """diagnose raising ValueError must propagate (not become exit 2)."""
        with patch("cozempic.cli.cmd_diagnose", side_effect=ValueError("unexpected")):
            try:
                _run_main_with_argv(["diagnose", "current"])
                assert False, "expected ValueError or SystemExit"
            except ValueError as exc:
                assert "unexpected" in str(exc)
            except SystemExit as exc:
                assert exc.code != 2, (
                    "non-guard/reload ValueError must NOT exit 2 — got SystemExit(2)"
                )

    def test_error_message_text_appears_in_stderr(self):
        """The ValueError message text must appear in stderr output."""
        stderr_buf = io.StringIO()
        with patch("cozempic.cli.cmd_guard", side_effect=ValueError("session_id malformed: @@")), \
             patch("sys.argv", ["cozempic", "guard"]), \
             patch("cozempic.cli._maybe_global_init"), \
             patch("cozempic.cli._maybe_auto_init"), \
             patch("cozempic.updater.ping_install_if_new"), \
             patch("cozempic.updater.maybe_auto_update"), \
             patch("sys.stderr", stderr_buf):
            from cozempic.cli import main
            try:
                main()
            except SystemExit:
                pass
        assert "session_id malformed: @@" in stderr_buf.getvalue(), (
            f"error text missing from stderr: {stderr_buf.getvalue()!r}"
        )


# ---------------------------------------------------------------------------
# CLI bug: _digest_session must resolve UUID via resolve_session (F4)
# ---------------------------------------------------------------------------

class TestDigestSessionResolution:
    """_digest_session(args) must call resolve_session() when args.session
    is provided, not return the string verbatim.

    Before fix: `return session_path, "", cwd` → passes UUID as a file path.
    After fix:  calls resolve_session(session_arg) → returns Path + stem ID.
    """

    def _make_args(self, session=None):
        args = MagicMock()
        args.session = session
        args.cwd = None
        return args

    def test_uuid_arg_resolves_to_path(self):
        """UUID string → resolve_session called, returned path is a Path object."""
        fake_path = Path("/fake/abc123.jsonl")
        # _digest_session does `from .session import ..., resolve_session` locally,
        # so patch the source module (cozempic.session) not the cli top-level binding.
        # The scope-conflict guard is covered separately in test_digest_isolation.
        with patch("cozempic.session.resolve_session", return_value=fake_path) as mock_rs, \
             patch("cozempic.digest.assert_session_in_scope"):
            path, session_id, cwd = _digest_session(self._make_args(session="abc123"))
        mock_rs.assert_called_once_with("abc123")
        assert path == fake_path

    def test_uuid_arg_session_id_from_stem(self):
        """session_id must be derived from path.stem (the UUID)."""
        fake_path = Path("/fake/abc123.jsonl")
        with patch("cozempic.session.resolve_session", return_value=fake_path), \
             patch("cozempic.digest.assert_session_in_scope"):
            path, session_id, cwd = _digest_session(self._make_args(session="abc123"))
        assert session_id == "abc123", f"expected 'abc123', got {session_id!r}"

    def test_path_arg_resolves_correctly(self):
        """Explicit file path → resolve_session returns it, stem extracted."""
        fake_path = Path("/real/path/uuid-val.jsonl")
        with patch("cozempic.session.resolve_session", return_value=fake_path), \
             patch("cozempic.digest.assert_session_in_scope"):
            path, session_id, cwd = _digest_session(
                self._make_args(session="/real/path/uuid-val.jsonl")
            )
        assert path == fake_path
        assert session_id == "uuid-val"

    def test_no_session_arg_uses_find_current(self):
        """args.session=None → find_current_session called; returns its path + id."""
        fake_sess = {"path": Path("/x/sess.jsonl"), "session_id": "sess"}
        # _digest_session does a local `from .session import find_current_session`,
        # so patch the source module (cozempic.session) not the cli binding.
        with patch("cozempic.session.find_current_session", return_value=fake_sess), \
             patch("cozempic.digest.assert_session_in_scope"):
            path, session_id, cwd = _digest_session(self._make_args(session=None))
        assert path == Path("/x/sess.jsonl")
        assert session_id == "sess"

    def test_no_session_arg_no_current_exits_1(self):
        """args.session=None + no current session → SystemExit(1)."""
        with patch("cozempic.session.find_current_session", return_value=None):
            try:
                _digest_session(self._make_args(session=None))
                assert False, "expected SystemExit(1)"
            except SystemExit as exc:
                assert exc.code == 1, f"expected exit 1, got {exc.code}"

    def test_current_literal_uses_cwd_find_current(self):
        """args.session='current' → uses cwd-based find_current_session (same as
        no-session path), NOT resolve_session('current') which uses process-detection.
        C3: keeps both paths consistent."""
        fake_sess = {"path": Path("/y/curr.jsonl"), "session_id": "curr"}
        with patch("cozempic.session.find_current_session", return_value=fake_sess) as mock_fc, \
             patch("cozempic.session.resolve_session") as mock_rs, \
             patch("cozempic.digest.assert_session_in_scope"):
            path, session_id, cwd = _digest_session(self._make_args(session="current"))
        # Must use find_current_session (cwd-based), NOT resolve_session
        mock_fc.assert_called_once()
        mock_rs.assert_not_called()
        assert path == Path("/y/curr.jsonl")
        assert session_id == "curr"

    def test_no_session_arg_calls_find_current_with_strict_true(self):
        """_digest_session calls find_current_session(cwd, strict=True) — not the
        non-strict default.

        digest flush/inject are write operations. Without strict=True, a user running
        `cozempic digest flush` in a project whose Strategy-3 lookup fails would
        silently inject rules into another project's session (Strategy-4 fallback).

        Spy asserts keyword `strict=True` is passed; no behavior gap if omitted.
        """
        fake_sess = {"path": Path("/z/strict.jsonl"), "session_id": "strict-sess"}
        with patch("cozempic.session.find_current_session", return_value=fake_sess) as mock_fc, \
             patch("cozempic.digest.assert_session_in_scope"):
            _digest_session(self._make_args(session=None))

        mock_fc.assert_called_once()
        _, kwargs = mock_fc.call_args
        assert kwargs.get("strict") is True, (
            f"find_current_session was called with strict={kwargs.get('strict')!r}, "
            "expected strict=True. Without this, digest writes can cross-contaminate "
            "when Strategy-3 fails and Strategy-4 picks a newer unrelated session."
        )
