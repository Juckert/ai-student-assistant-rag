import csv
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MANUAL_QA_FILENAME = "admin_added_qa.csv"
MANUAL_QA_PATH = os.path.join(DATA_DIR, MANUAL_QA_FILENAME)
MANUAL_QA_HEADERS = [
    "question_text",
    "answer_text",
    "question_topic",
    "question_year",
    "question_course",
]


def ensure_manual_qa_file(csv_path=None):
    resolved_path = csv_path or MANUAL_QA_PATH
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)

    if os.path.exists(resolved_path):
        return resolved_path

    with open(resolved_path, "w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANUAL_QA_HEADERS)
        writer.writeheader()

    return resolved_path


def append_manual_qa(
    question,
    answer,
    topic="Ручное пополнение базы",
    year="",
    course="",
    csv_path=None,
):
    resolved_path = ensure_manual_qa_file(csv_path)
    normalized_question = " ".join(question.split())
    normalized_answer = " ".join(answer.split())

    with open(resolved_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            existing_question = " ".join((row.get("question_text") or "").split())
            existing_answer = " ".join((row.get("answer_text") or "").split())
            if existing_question == normalized_question and existing_answer == normalized_answer:
                return resolved_path, False

    with open(resolved_path, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANUAL_QA_HEADERS)
        writer.writerow(
            {
                "question_text": normalized_question,
                "answer_text": normalized_answer,
                "question_topic": topic,
                "question_year": year,
                "question_course": course,
            }
        )

    return resolved_path, True
