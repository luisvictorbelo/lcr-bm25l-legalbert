import json
from pathlib import Path


def normalize_case_id(case_id: str) -> str:
    return Path(str(case_id).strip()).stem


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_training_pairs(labels_path, output_path):
    """
    Create pure GT positive pairs.

    query -> positive
    """

    labels = load_json(labels_path)

    pairs = []

    for raw_query, raw_positives in labels.items():
        query_id = normalize_case_id(raw_query)

        positives = [
            normalize_case_id(x)
            for x in raw_positives
        ]

        for positive_id in positives:
            pairs.append(
                {
                    "query_id": query_id,
                    "positive_id": positive_id,
                }
            )

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            pairs,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Pairs created: {len(pairs)}")
    print(f"Saved to: {output_path}")

    return pairs


if __name__ == "__main__":
    create_training_pairs(
        labels_path="data/labels/task1_train_labels_2025.json",
        output_path="data/processed/train_pairs.json",
    )