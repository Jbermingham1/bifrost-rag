"""Tests for the CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bifrost_rag.cli import main


def _write_corpus(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "apple", "text": "red apple fruit sweet juicy crisp"}),
                json.dumps({"id": "banana", "text": "yellow banana fruit sweet potassium"}),
                json.dumps({"id": "elephant", "text": "grey elephant animal large mammal"}),
            ]
        ),
        encoding="utf-8",
    )


def _write_cases(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {"query_id": "q1", "query": "red apple fruit sweet", "relevant_ids": ["apple"]}
                ),
                json.dumps(
                    {
                        "query_id": "q2",
                        "query": "elephant mammal large",
                        "relevant_ids": ["elephant"],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )


class TestCLI:
    def test_index_command(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        corpus = tmp_path / "corpus.jsonl"
        _write_corpus(corpus)
        exit_code = main(["index", str(corpus)])
        assert exit_code == 0
        output = json.loads(capsys.readouterr().out)
        assert output["indexed"] == 3

    def test_query_command(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        corpus = tmp_path / "corpus.jsonl"
        _write_corpus(corpus)
        exit_code = main(["query", str(corpus), "red apple fruit sweet juicy", "--top-k", "2"])
        assert exit_code == 0
        output = json.loads(capsys.readouterr().out)
        assert output["query"] == "red apple fruit sweet juicy"
        assert len(output["retrieved"]) == 2

    def test_eval_command(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        corpus = tmp_path / "corpus.jsonl"
        cases = tmp_path / "cases.jsonl"
        _write_corpus(corpus)
        _write_cases(cases)
        exit_code = main(["eval", str(corpus), str(cases), "--k", "2"])
        assert exit_code == 0
        output = json.loads(capsys.readouterr().out)
        assert output["num_cases"] == 2
        assert output["k"] == 2
        assert "precision_at_k" in output
        assert "mrr" in output

    def test_query_missing_corpus_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            main(["query", str(tmp_path / "missing.jsonl"), "anything"])

    def test_corpus_invalid_json_raises(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text("not valid json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            main(["query", str(corpus), "anything"])

    def test_blank_lines_skipped(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            "\n".join(
                [
                    json.dumps({"id": "a", "text": "alpha beta gamma"}),
                    "",
                    "   ",
                    json.dumps({"id": "b", "text": "delta epsilon zeta"}),
                ]
            ),
            encoding="utf-8",
        )
        exit_code = main(["index", str(corpus)])
        assert exit_code == 0
        output = json.loads(capsys.readouterr().out)
        assert output["indexed"] == 2
