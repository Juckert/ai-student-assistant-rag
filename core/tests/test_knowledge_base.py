import csv

from app.knowledge_base import append_manual_qa


def test_append_manual_qa_creates_csv_and_skips_duplicate(tmp_path):
    csv_path = tmp_path / "admin_added_qa.csv"

    first_path, first_added = append_manual_qa(
        "Когда физра?",
        "Информацию по физкультуре нужно уточнять отдельно.",
        csv_path=str(csv_path),
    )
    second_path, second_added = append_manual_qa(
        "Когда физра?",
        "Информацию по физкультуре нужно уточнять отдельно.",
        csv_path=str(csv_path),
    )

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert first_path == second_path
    assert first_added is True
    assert second_added is False
    assert len(rows) == 1
    assert rows[0]["question_text"] == "Когда физра?"
