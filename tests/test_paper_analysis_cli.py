import json
import sys

import pytest

from scripts import analyze_shc, audit_paper_sources, compare_shc_prompts, replay_shc


@pytest.mark.parametrize("module", [analyze_shc, compare_shc_prompts, replay_shc])
def test_analysis_clis_reject_synthetic_inputs_without_writing_reports(
    tmp_path, monkeypatch, module
):
    path, output = tmp_path / "fixture.jsonl", tmp_path / "report.json"
    path.write_text(json.dumps({"synthetic": True}) + "\n")
    monkeypatch.setattr(
        sys, "argv", ["analysis", "--observations", str(path), "--output", str(output)]
    )
    with pytest.raises(ValueError, match="software checks"):
        module.main()
    assert not output.exists()


def test_source_audit_cli_rejects_synthetic_metadata(tmp_path, monkeypatch):
    path, output = tmp_path / "fixture.jsonl", tmp_path / "report.json"
    path.write_text(json.dumps({"synthetic": True}) + "\n")
    monkeypatch.setattr(
        sys, "argv", ["audit", "--baseline", str(path), "--shc", str(path), "--output", str(output)]
    )
    with pytest.raises(ValueError, match="software checks"):
        audit_paper_sources.main()
    assert not output.exists()


@pytest.mark.parametrize("content", ["", " \n", "{}\n", '{"synthetic": true}\n'])
def test_report_reader_requires_nonempty_explicit_real_data(tmp_path, content):
    from scripts._report_io import read_real_rows

    path = tmp_path / "rows.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError, match="software checks"):
        read_real_rows(path)


def test_report_io_preserves_unicode_and_refuses_to_replace_evidence(tmp_path):
    from scripts._report_io import read_real_rows, write_report

    source = tmp_path / "rows.jsonl"
    source.write_text(' \n{"synthetic": false, "text": "中文"}\n', encoding="utf-8")
    rows = read_real_rows(source)
    output = tmp_path / "nested" / "report.json"
    write_report(output, {"rows": rows})
    original = output.read_bytes()
    assert "中文" in original.decode("utf-8")
    with pytest.raises(FileExistsError):
        write_report(output, {"rows": []})
    assert output.read_bytes() == original
