"""End-of-training eval failures are recorded and do not raise."""

from __future__ import annotations

import json
import logging

from dvae.learning_algo import run_post_training_eval


def _logger(path):
    logger = logging.getLogger("eval-status-test")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def test_failing_eval_logs_tail_and_writes_marker(tmp_path):
    script = tmp_path / "fail_eval.py"
    script.write_text(
        "import sys\n"
        "sys.stderr.write(''.join('boom %d\\n' % i for i in range(50)))\n"
        "raise SystemExit(3)\n"
    )
    log_path = tmp_path / "log.txt"
    logger = _logger(log_path)
    status = run_post_training_eval(
        str(script), "weights.pt", str(tmp_path), logger
    )
    assert status["eval_failed"] is True
    assert status["return_code"] == 3
    marker = json.loads((tmp_path / "eval_status.json").read_text())
    assert marker["eval_failed"] is True
    assert marker["return_code"] == 3
    assert marker["stderr_tail"].strip().splitlines()[-1] == "boom 49"
    assert len(marker["stderr_tail"].splitlines()) == 40
    text = log_path.read_text()
    assert "ERROR" in text
    assert "return code 3" in text
    assert "boom 49" in text


def test_successful_eval_marks_not_failed(tmp_path):
    script = tmp_path / "ok_eval.py"
    script.write_text("print('eval ok')\n")
    log_path = tmp_path / "log.txt"
    status = run_post_training_eval(
        str(script), "weights.pt", str(tmp_path), _logger(log_path)
    )
    assert status["eval_failed"] is False
    assert status["return_code"] == 0
    marker = json.loads((tmp_path / "eval_status.json").read_text())
    assert marker["eval_failed"] is False
    assert "ERROR" not in log_path.read_text()
