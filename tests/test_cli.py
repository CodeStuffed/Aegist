"""The four commands, driven through main() like a user would."""

from __future__ import annotations

import json
import sys

import pytest
import yaml

import cli
from tests.conftest import CORPUS


@pytest.fixture
def run(settings, tmp_path, monkeypatch, capsys):
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings))
    monkeypatch.setenv("COUNCIL_SETTINGS", str(path))

    def _run(*argv):
        monkeypatch.setattr(sys, "argv", ["cli.py", *argv])
        code = cli.main()
        captured = capsys.readouterr()
        return code, captured.out, captured.err
    return _run


def test_ask_before_training_explains_what_to_do(run):
    code, _, err = run("ask", "Light bends in glass")
    assert code == 2 and "No trained model" in err and "cli.py train" in err


def test_train_then_ask_then_doctor(run, tmp_path):
    texts = tmp_path / "texts"
    texts.mkdir()
    (texts / "notes.md").write_text(CORPUS * 20)
    code, out, _ = run("train", "--data", str(texts), "--steps", "5")
    assert code == 0 and "Imported 1 file(s)" in out and "Saved: 5 steps" in out

    code, out, _ = run("ask", "We should raise our subscription price")
    assert code == 0
    for heading in ("BELIEVER", "SKEPTIC", "INVESTOR", "JUDGE", "OVERALL CONFIDENCE"):
        assert heading in out

    code, out, _ = run("ask", "--json", "--no-recheck", "Light bends in glass")
    result = json.loads(out)
    assert result["judge"]["stance"] in ("yes", "no", "mixed") and "consistency" not in result

    code, out, _ = run("doctor")
    assert code == 0 and "5 steps" in out and "2 past council session(s)" in out
