import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "kaggle"))
from runner import MAX_FAILS_IN_A_ROW, Runner, read_done  # noqa: E402

GPU = [{"index": 0, "name": "test"}]


def _script(tmp_path):
    (tmp_path / "t.py").write_text(
        "import sys, time, pathlib\n"
        "name, sec, rc = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])\n"
        "time.sleep(sec)\n"
        "pathlib.Path('runs_kaggle').mkdir(exist_ok=True)\n"
        "pathlib.Path('runs_kaggle', name).write_text('x')\n"
        "sys.exit(rc)\n")


def _runner(tmp_path, lines, hours=1.0, min_left=0.0, previous=()):
    return Runner(lines, GPU, hours, min_left, tmp_path, tmp_path / "out", ["runs_kaggle"], previous)


def test_runs_all_and_next_session_skips_done(tmp_path):
    _script(tmp_path)
    lines = ["t.py a 0 0", "t.py b 0 1", "t.py c 0 0"]
    counts, left = _runner(tmp_path, lines).run()
    assert counts == {"done": 2, "failed": 1, "killed": 0} and left == 0
    ledger = tmp_path / "out" / "ledger.tsv"
    assert set(read_done([ledger])) == {"t.py a 0 0", "t.py c 0 0"}
    assert (tmp_path / "out" / "results.tar.gz").exists()

    saved = tmp_path / "previous.tsv"
    saved.write_text(ledger.read_text())
    second = _runner(tmp_path, lines, previous=[saved])
    assert second.todo == ["t.py b 0 1"]                  # only the failed line is retried
    second.run()
    assert set(read_done([ledger])) == {"t.py a 0 0", "t.py c 0 0"}   # the ledger stays cumulative


def test_deadline_kills_a_running_task_and_does_not_mark_it_done(tmp_path):
    _script(tmp_path)
    r = _runner(tmp_path, ["t.py slow 60 0"], hours=1 / 3600)
    t0 = time.time()
    counts, _ = r.run()
    assert counts["killed"] == 1 and time.time() - t0 < 30
    assert read_done([tmp_path / "out" / "ledger.tsv"]) == {}


def test_no_new_task_near_the_limit(tmp_path):
    _script(tmp_path)
    counts, left = _runner(tmp_path, ["t.py a 0 0"], hours=1.0, min_left=1.0).run()
    assert left == 1 and sum(counts.values()) == 0


def test_stops_after_many_failures_in_a_row(tmp_path):
    _script(tmp_path)
    lines = [f"t.py f{i} 0 1" for i in range(MAX_FAILS_IN_A_ROW + 3)]
    counts, left = _runner(tmp_path, lines).run()
    assert counts["failed"] == MAX_FAILS_IN_A_ROW and left == 3
