"""Claude-config-dir × project isolation for the behavioral digest.

Verifies the isolation rules with TWO Claude config directories and TWO
projects:

  1. Extraction / injection / recovery are scoped to the active
     (CLAUDE_CONFIG_DIR, cwd) pair — rules never leak across projects or
     across config profiles ("不串用").
  2. Cross-project sharing works ONLY through an explicit share action and
     stays inside one config profile ("显式共享有效").
  3. Legacy global rules with unknown ownership are quarantined first and
     never auto-injected; they can be adopted explicitly.
  4. Clearing a scope removes cozempic artifacts only and preserves
     user-owned memory.
  5. A session transcript from another project/config combined with a
     mismatching working directory fails with a clear error (exit 2).

Note: digest symbols are accessed as ``digest.<name>`` at call time because
other tests in the suite ``importlib.reload(cozempic.digest)``; top-level
``from cozempic.digest import X`` bindings (especially exception classes)
would go stale.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cozempic import digest
from cozempic.helpers import msg_bytes
from cozempic.session import cwd_to_project_slug


# ---------------------------------------------------------------------------
# Message fixtures (same JSONL-tuple shape the parser consumes)
# ---------------------------------------------------------------------------

def make_user(text: str) -> tuple[int, dict, int]:
    msg = {"type": "user", "message": {"role": "user", "content": text}}
    return (0, msg, msg_bytes(msg))


RULE_A = "don't add Co-Authored-By lines to git commits"
RULE_B = "always run ruff format before writing a file"


# ---------------------------------------------------------------------------
# Two configs × two projects harness
# ---------------------------------------------------------------------------

class IsolationHarness:
    """Builds:

      <tmp>/cozempic/                        → patched digest.DIGEST_DIR
      <tmp>/cfg-a/projects/<slugA>/memory/   → config A project A
      <tmp>/cfg-a/projects/<slugB>/memory/   → config A project B
      <tmp>/cfg-b/projects/<slugA>/memory/   → config B project A
      <tmp>/cfg-b/projects/<slugB>/memory/   → config B project B

    (Memory dirs are pre-created because sync only writes into a memdir that
    Claude Code has already established.)
    """

    PA = "/work/alpha"
    PB = "/work/beta"

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.digest_dir = self.tmp / "cozempic"
        self.digest_dir.mkdir(parents=True)
        self.cfg_a = self.tmp / "cfg-a"
        self.cfg_b = self.tmp / "cfg-b"
        for cfg in (self.cfg_a, self.cfg_b):
            for proj in (self.PA, self.PB):
                (cfg / "projects" / cwd_to_project_slug(proj) / "memory").mkdir(
                    parents=True, exist_ok=True)

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @contextmanager
    def config(self, cfg: Path):
        env = {k: v for k, v in os.environ.items() if k != "CLAUDE_CONFIG_DIR"}
        env["CLAUDE_CONFIG_DIR"] = str(cfg)
        with patch.dict(os.environ, env), \
             patch("cozempic.digest.DIGEST_DIR", self.digest_dir):
            yield

    def mem_file(self, cfg: Path, proj: str) -> Path:
        return cfg / "projects" / cwd_to_project_slug(proj) / "memory" / "cozempic_digest.md"

    def promote(self, cfg: Path, proj: str, phrase: str) -> str:
        """Run two extraction passes so the correction promotes to ACTIVE;
        returns its rule ID in that project's own store."""
        with self.config(cfg):
            digest.update_digest([make_user(phrase)], project_dir=proj, session_id="s1")
            digest.update_digest([make_user(phrase)], project_dir=proj, session_id="s1")
            return digest.load_scoped(proj).own.active_rules()[0].id

    def sync(self, cfg: Path, proj: str) -> int:
        with self.config(cfg):
            return digest.recover_digest(proj)

    def injected_text(self, cfg: Path, proj: str) -> str | None:
        f = self.mem_file(cfg, proj)
        return f.read_text() if f.exists() else None


class TestScopeLayout(unittest.TestCase):

    def setUp(self):
        self.h = IsolationHarness()
        self.addCleanup(self.h.cleanup)

    def test_two_configs_get_distinct_keys(self):
        with self.h.config(self.h.cfg_a):
            key_a = digest.config_scope_key()
        with self.h.config(self.h.cfg_b):
            key_b = digest.config_scope_key()
        self.assertNotEqual(key_a, key_b)
        self.assertNotIn("/", key_a)  # slug-safe path component
        self.assertNotIn("/", key_b)

    def test_project_and_shared_paths_are_isolated(self):
        with self.h.config(self.h.cfg_a):
            pa = digest.scope_paths(IsolationHarness.PA)
            pb = digest.scope_paths(IsolationHarness.PB)
        # Same config root, distinct project files, one shared store.
        self.assertEqual(pa.root, pb.root)
        self.assertNotEqual(pa.own_json, pb.own_json)
        self.assertEqual(pa.shared_json, pb.shared_json)

        with self.h.config(self.h.cfg_b):
            pa_b = digest.scope_paths(IsolationHarness.PA)
        # Different config → completely different root (no cross-profile files).
        self.assertNotEqual(pa.root, pa_b.root)
        self.assertNotEqual(pa.shared_json, pa_b.shared_json)


class TestRulesDoNotLeak(unittest.TestCase):
    """CORE RULE: rules extracted in one (config × project) scope must never
    appear in another scope's store or injected memory."""

    def setUp(self):
        self.h = IsolationHarness()
        self.addCleanup(self.h.cleanup)

    def test_no_cross_project_or_cross_config_leak(self):
        h = self.h

        # Rule A lives in config A / project A; rule B in config A / project B.
        h.promote(h.cfg_a, h.PA, RULE_A)
        h.promote(h.cfg_a, h.PB, RULE_B)
        h.sync(h.cfg_a, h.PA)
        h.sync(h.cfg_a, h.PB)

        # Each injected file carries its OWN project's rule only.
        pa_text = h.injected_text(h.cfg_a, h.PA)
        pb_text = h.injected_text(h.cfg_a, h.PB)
        self.assertIn("Co-Authored-By", pa_text)
        self.assertNotIn("ruff format", pa_text)
        self.assertIn("ruff format", pb_text)
        self.assertNotIn("Co-Authored-By", pb_text)

        # Stores on disk are separate files; PB's own store never got rule A.
        with h.config(h.cfg_a):
            pa_store = json.loads(digest.scope_paths(h.PA).own_json.read_text())
            pb_store = json.loads(digest.scope_paths(h.PB).own_json.read_text())
        self.assertTrue(any("Co-Authored-By" in r["rule"] for r in pa_store["strategy_rules"]))
        self.assertFalse(any("Co-Authored-By" in r["rule"] for r in pb_store["strategy_rules"]))
        # Nothing was auto-shared.
        with h.config(h.cfg_a):
            shared_json = digest.scope_paths(h.PA).shared_json
        self.assertFalse(shared_json.exists())

        # Config B never saw either rule — no stores, no injected files.
        self.assertIsNone(h.injected_text(h.cfg_b, h.PA))
        self.assertIsNone(h.injected_text(h.cfg_b, h.PB))
        with h.config(h.cfg_b):
            scoped_b_pa = digest.load_scoped(h.PA)
        self.assertEqual(scoped_b_pa.own.strategy_rules, [])
        self.assertEqual(scoped_b_pa.shared.strategy_rules, [])

    def test_same_cwd_under_two_configs_is_independent(self):
        h = self.h
        h.promote(h.cfg_a, h.PA, RULE_A)
        h.sync(h.cfg_a, h.PA)
        self.assertIn("Co-Authored-By", h.injected_text(h.cfg_a, h.PA))
        # Config B / same path: empty scope, sync writes nothing and MUST NOT
        # read config A's rules.
        self.assertEqual(h.sync(h.cfg_b, h.PA), 0)
        self.assertIsNone(h.injected_text(h.cfg_b, h.PA))


class TestExplicitSharing(unittest.TestCase):
    """Cross-project channel exists ONLY via explicit share, and is itself
    bounded by the config profile."""

    def setUp(self):
        self.h = IsolationHarness()
        self.addCleanup(self.h.cleanup)

    def test_share_crosses_projects_but_not_configs(self):
        h = self.h
        rid = h.promote(h.cfg_a, h.PA, RULE_A)

        # Explicit share from project A into config A's shared store.
        with h.config(h.cfg_a):
            confirmed = digest.share_rules(h.PA, [rid])
        self.assertEqual(confirmed, [rid])

        # Project B (same config) now receives the rule via injection...
        self.assertEqual(h.sync(h.cfg_a, h.PB), 1)
        pb_text = h.injected_text(h.cfg_a, h.PB)
        self.assertIn("SHARED ACROSS PROJECTS", pb_text)
        self.assertIn("Co-Authored-By", pb_text)

        # ...but project B's OWN store stays empty (sharing ≠ copying ownership).
        with h.config(h.cfg_a):
            self.assertFalse(digest.scope_paths(h.PB).own_json.exists())

        # Project A still gets its own copy exactly once (no duplicated shared
        # render of the same rule).
        h.sync(h.cfg_a, h.PA)
        pa_text = h.injected_text(h.cfg_a, h.PA)
        self.assertEqual(pa_text.count("[R001|"), 1)

        # Config B projects receive NOTHING — sharing is per config.
        self.assertEqual(h.sync(h.cfg_b, h.PB), 0)
        self.assertIsNone(h.injected_text(h.cfg_b, h.PB))

    def test_unshare_revokes_injection_everywhere_in_config(self):
        h = self.h
        rid = h.promote(h.cfg_a, h.PA, RULE_A)
        with h.config(h.cfg_a):
            digest.share_rules(h.PA, [rid])
        self.assertEqual(h.sync(h.cfg_a, h.PB), 1)
        self.assertIsNotNone(h.injected_text(h.cfg_a, h.PB))

        # Revoke (can be done from any project of the same config) → the next
        # recovery removes the rule from project B's injected file. Project A's
        # own copy survives.
        with h.config(h.cfg_a):
            digest.unshare_rules(h.PB, [rid])
        self.assertEqual(h.sync(h.cfg_a, h.PB), 0)
        self.assertIsNone(h.injected_text(h.cfg_a, h.PB))
        h.sync(h.cfg_a, h.PA)
        self.assertIn("Co-Authored-By", h.injected_text(h.cfg_a, h.PA))

    def test_share_unknown_id_is_an_error(self):
        h = self.h
        with h.config(h.cfg_a):
            with self.assertRaises(KeyError):
                digest.share_rules(h.PA, ["R999"])


class TestLegacyQuarantine(unittest.TestCase):
    """Pre-isolation global store has no project ownership → isolate first,
    never auto-inject; explicit adopt claims it."""

    def setUp(self):
        self.h = IsolationHarness()
        self.addCleanup(self.h.cleanup)

    def _write_legacy_global(self):
        legacy = self.h.digest_dir / "behavioral-digest.json"
        rule = digest.DigestRule(
            id="R001",
            rule="Do not add Co-Authored-By lines to commits",
            evidence="don't add Co-Authored-By lines to commits",
            scope="git",
            priority="hard",
            status="active",
            occurrence_count=2,
            source_reliability=1.0,
            type_prior=0.8,
        )
        payload = {
            "version": "1",
            "project": "unknown-legacy",
            "updated": "2020-01-01T00:00:00+00:00",
            "session_id": "",
            "strategy_rules": [json.loads(json.dumps(rule.__dict__))],
            "failure_patterns": [],
        }
        legacy.write_text(json.dumps(payload))
        self.h.digest_dir.joinpath("behavioral-digest.md").write_text("# legacy")
        return legacy

    def test_legacy_rules_are_quarantined_and_never_injected(self):
        h = self.h
        legacy = self._write_legacy_global()

        with h.config(h.cfg_a):
            moved = digest.quarantine_legacy_global_store()
            scoped = digest.load_scoped(h.PA)
        self.assertTrue(moved)
        self.assertFalse(legacy.exists())
        q = self.h.digest_dir / "quarantine"
        self.assertTrue((q / "behavioral-digest.json").exists())
        self.assertTrue((q / ".migrated_from_global").exists())

        # Not injected into any project, not in own, not in shared.
        self.assertEqual(scoped.own.strategy_rules, [])
        self.assertEqual(scoped.shared.strategy_rules, [])
        self.assertEqual(h.sync(h.cfg_a, h.PA), 0)
        self.assertIsNone(h.injected_text(h.cfg_a, h.PA))

        # Idempotent: a second run does nothing.
        with h.config(h.cfg_a):
            self.assertFalse(digest.quarantine_legacy_global_store())

    def test_adopt_into_current_project(self):
        h = self.h
        self._write_legacy_global()
        with h.config(h.cfg_a):
            n = digest.adopt_quarantined(h.PA)
        self.assertEqual(n, 1)
        h.sync(h.cfg_a, h.PA)
        self.assertIn("Co-Authored-By", h.injected_text(h.cfg_a, h.PA))
        # Ownership resolved → quarantine cleared; other project unaffected.
        self.assertFalse((self.h.digest_dir / "quarantine" / "behavioral-digest.json").exists())
        self.assertIsNone(h.injected_text(h.cfg_a, h.PB))

    def test_adopt_into_shared_reaches_all_projects_of_config_only(self):
        h = self.h
        self._write_legacy_global()
        with h.config(h.cfg_a):
            n = digest.adopt_quarantined(h.PA, into_shared=True)
        self.assertEqual(n, 1)
        # Reaches every project in config A...
        self.assertEqual(h.sync(h.cfg_a, h.PB), 1)
        self.assertIn("SHARED ACROSS PROJECTS", h.injected_text(h.cfg_a, h.PB))
        # ...but not config B.
        self.assertEqual(h.sync(h.cfg_b, h.PB), 0)
        self.assertIsNone(h.injected_text(h.cfg_b, h.PB))

    def _write_legacy_store(self, rules):
        legacy = self.h.digest_dir / "behavioral-digest.json"
        payload = {
            "version": "1",
            "project": "unknown-legacy",
            "updated": "2020-01-01T00:00:00+00:00",
            "session_id": "",
            "strategy_rules": [
                json.loads(json.dumps(r.__dict__)) for r in rules
            ],
            "failure_patterns": [],
        }
        legacy.write_text(json.dumps(payload))
        self.h.digest_dir.joinpath("behavioral-digest.md").write_text("# legacy")
        return legacy

    @staticmethod
    def _admissible_rule(rule_id="R001"):
        return digest.DigestRule(
            id=rule_id,
            rule="Do not add Co-Authored-By lines to commits",
            evidence="don't add Co-Authored-By lines to commits",
            scope="git",
            priority="hard",
            status="active",
            occurrence_count=2,
            source_reliability=1.0,
            type_prior=0.8,
        )

    @staticmethod
    def _rejected_rule():
        # Composite score = 0.20 (recency only) < ADMISSION_THRESHOLD 0.55.
        return digest.DigestRule(
            id="R009",
            rule="Maybe consider thinking about things sometimes",
            evidence="perhaps try to maybe consider stuff sometime",
            scope="general",
            priority="low",
            status="pending",
            occurrence_count=0,
            source_reliability=0.0,
            type_prior=0.0,
        )

    def test_adopt_keeps_quarantine_when_gate_refuses_all_rules(self):
        h = self.h
        self._write_legacy_store([self._rejected_rule()])
        qjson = h.digest_dir / "quarantine" / "behavioral-digest.json"
        marker = h.digest_dir / "quarantine" / ".migrated_from_global"

        with h.config(h.cfg_a):
            n = digest.adopt_quarantined(h.PA)

        self.assertEqual(n, 0)
        # Nothing admitted → quarantine (incl. marker) retained verbatim...
        self.assertTrue(qjson.exists())
        self.assertTrue(marker.exists())
        with h.config(h.cfg_a):
            self.assertEqual(digest.quarantined_rule_count(), 1)
            # ...and no empty project store is fabricated.
            self.assertFalse(digest.scope_paths(h.PA).own_json.exists())
        # Retained legacy rules still never auto-inject anywhere.
        self.assertIsNone(h.injected_text(h.cfg_a, h.PA))
        self.assertIsNone(h.injected_text(h.cfg_a, h.PB))

    def test_partial_adoption_keeps_only_refused_rules(self):
        h = self.h
        self._write_legacy_store([self._admissible_rule(), self._rejected_rule()])
        qjson = h.digest_dir / "quarantine" / "behavioral-digest.json"

        with h.config(h.cfg_a):
            n = digest.adopt_quarantined(h.PA)
            own = digest.load_scoped(h.PA).own

        self.assertEqual(n, 1)
        self.assertEqual(len(own.strategy_rules), 1)
        self.assertIn("Co-Authored-By", own.strategy_rules[0].rule)
        # Quarantine rewritten down to the single refused rule.
        self.assertTrue(qjson.exists())
        with h.config(h.cfg_a):
            remaining = digest._load_store_file(qjson, project_label="q")
        self.assertEqual(len(remaining.strategy_rules), 1)
        self.assertNotIn("Co-Authored-By", remaining.strategy_rules[0].rule)

        # A retry must not reprocess/duplicate the already-claimed rule.
        with h.config(h.cfg_a):
            n2 = digest.adopt_quarantined(h.PA)
            own2 = digest.load_scoped(h.PA).own
        self.assertEqual(n2, 0)
        self.assertEqual(len(own2.strategy_rules), 1)


class TestClearPreservesUserMemory(unittest.TestCase):

    def setUp(self):
        self.h = IsolationHarness()
        self.addCleanup(self.h.cleanup)

    def test_clear_removes_only_cozempic_artifacts(self):
        h = self.h
        h.promote(h.cfg_a, h.PA, RULE_A)
        h.promote(h.cfg_a, h.PB, RULE_B)
        h.sync(h.cfg_a, h.PA)
        h.sync(h.cfg_a, h.PB)

        # User-owned memory alongside cozempic's injected file.
        mem = h.cfg_a / "projects" / cwd_to_project_slug(h.PA) / "memory"
        (mem / "user_note.md").write_text("# my own note\n")
        index = mem / "MEMORY.md"
        index.write_text(
            "# Memory\n"
            "- [My Note](user_note.md) — mine\n"
            "- [Cozempic Behavioral Digest](cozempic_digest.md) — behavioral rules from user corrections\n"
        )

        with h.config(h.cfg_a):
            result = digest.clear_scope(h.PA)

        self.assertTrue(result.memfile_removed)
        self.assertTrue(result.index_entry_pruned)
        # cozempic artifacts gone...
        self.assertFalse(h.mem_file(h.cfg_a, h.PA).exists())
        with h.config(h.cfg_a):
            self.assertFalse(digest.scope_paths(h.PA).own_json.exists())
        # ...user memory preserved, MEMORY.md kept with the user line only.
        self.assertTrue((mem / "user_note.md").exists())
        kept = index.read_text()
        self.assertIn("user_note.md", kept)
        self.assertNotIn("cozempic_digest.md", kept)

        # Sibling project scope untouched.
        self.assertIsNotNone(h.injected_text(h.cfg_a, h.PB))
        with h.config(h.cfg_a):
            self.assertTrue(digest.scope_paths(h.PB).own_json.exists())

    def test_clear_shared_requires_flag_and_is_scoped(self):
        h = self.h
        rid = h.promote(h.cfg_a, h.PA, RULE_A)
        with h.config(h.cfg_a):
            digest.share_rules(h.PA, [rid])
            shared_json = digest.scope_paths(h.PA).shared_json
        self.assertTrue(shared_json.exists())

        # Plain clear does NOT remove the shared store.
        with h.config(h.cfg_a):
            digest.clear_scope(h.PA)
        self.assertTrue(shared_json.exists())

        # Explicit --shared clears it.
        with h.config(h.cfg_a):
            result = digest.clear_scope(h.PB, include_shared=True)
        self.assertTrue(result.shared_cleared)
        self.assertFalse(shared_json.exists())


class TestSessionWorkingDirConflict(unittest.TestCase):
    """A transcript from another project/config must fail loudly instead of
    extracting into the current project's scope."""

    def setUp(self):
        self.h = IsolationHarness()
        self.addCleanup(self.h.cleanup)
        self.sess_a = (
            self.h.cfg_a / "projects" / cwd_to_project_slug(self.h.PA)
            / "00000000-0000-4000-8000-000000000001.jsonl"
        )
        self.sess_a.write_text("", encoding="utf-8")

    def test_matching_scope_passes(self):
        with self.h.config(self.h.cfg_a):
            # Must not raise.
            digest.assert_session_in_scope(self.sess_a, self.h.PA)

    def test_project_mismatch_raises_clear_error(self):
        # Resolve the exception class at call time (module may have been
        # reloaded by other tests in the same process).
        with self.h.config(self.h.cfg_a):
            with self.assertRaises(digest.DigestScopeConflict) as cm:
                digest.assert_session_in_scope(self.sess_a, self.h.PB)
        msg = str(cm.exception)
        self.assertIn("scope conflict", msg)
        # Both identities are shown so the user can fix --cwd.
        self.assertIn(cwd_to_project_slug(self.h.PA), msg)
        self.assertIn(cwd_to_project_slug(self.h.PB), msg)

    def test_config_mismatch_raises(self):
        # Same project slug, but transcript lives under config A while the
        # active config is B.
        with self.h.config(self.h.cfg_b):
            with self.assertRaises(digest.DigestScopeConflict) as cm:
                digest.assert_session_in_scope(self.sess_a, self.h.PA)
        msg = str(cm.exception)
        self.assertIn(str(self.h.cfg_a), msg)
        self.assertIn(str(self.h.cfg_b), msg)

    def test_cli_flush_conflict_exits_2_with_message(self):
        from cozempic.cli import cmd_digest
        err = io.StringIO()
        args = SimpleNamespace(
            digest_action="flush",
            session=str(self.sess_a),
            cwd=self.h.PB,
            rule_ids=[],
            shared=False,
        )
        with self.h.config(self.h.cfg_a), \
             patch("sys.stderr", err), \
             self.assertRaises(SystemExit) as cm_exit:
            cmd_digest(args)
        self.assertEqual(cm_exit.exception.code, 2)
        self.assertIn("scope conflict", err.getvalue())
        self.assertIn(cwd_to_project_slug(self.h.PA), err.getvalue())
        self.assertIn(cwd_to_project_slug(self.h.PB), err.getvalue())


class TestDigestParserActions(unittest.TestCase):
    """New CLI actions parse as intended (argparse regression guard)."""

    def test_share_and_clear_shared_parse(self):
        from cozempic.cli import build_parser
        parser = build_parser()
        ns = parser.parse_args(["digest", "share", "R001", "R002"])
        self.assertEqual(ns.digest_action, "share")
        self.assertEqual(ns.rule_ids, ["R001", "R002"])

        ns = parser.parse_args(["digest", "clear", "--shared"])
        self.assertEqual(ns.digest_action, "clear")
        self.assertTrue(ns.shared)

        ns = parser.parse_args(["digest", "adopt", "--shared"])
        self.assertEqual(ns.digest_action, "adopt")
        self.assertTrue(ns.shared)

    def test_share_without_ids_is_usage_error(self):
        from cozempic.cli import cmd_digest
        # argparse accepts zero ids (nargs="*"); the command itself rejects it
        # with exit 2 so hooks/users get a clear usage error.
        err = io.StringIO()
        args = SimpleNamespace(
            digest_action="share", session=None, cwd=None,
            rule_ids=[], shared=False,
        )
        with patch("sys.stderr", err), self.assertRaises(SystemExit) as cm:
            cmd_digest(args)
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("rule IDs", err.getvalue())


if __name__ == "__main__":
    unittest.main()
