import csv
from pathlib import Path

from palak_repro.report import make_report


def test_report_does_not_invent_measured_values(tmp_path):
    result = make_report({"output_dir": str(tmp_path), "targets": ["bird"]})
    assert len(result) == 6
    assert all(row["measured_asr"] == "NOT MEASURED" for row in result)
    with (tmp_path / "tables" / "comparison.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["measured_asr"] == "NOT MEASURED"
