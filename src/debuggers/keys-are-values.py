import json
from pathlib import Path


def find_keys_that_are_also_values(json_path: Path):
    """
    Return cases that appear both as:
    - dictionary key
    - item inside any value list
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # all keys
    keys = set(data.keys())

    # flatten all values into a set
    values = {
        item
        for value_list in data.values()
        for item in value_list
    }

    # intersection
    overlap = keys & values

    return sorted(overlap)


def main():
    cwd = Path.cwd()

    json_file = cwd / "data" / "labels" / "task1_test_labels_2025.json"  # change path

    if (json_file.exists() == False):
        print('Arquivo nao encontrado')
        return

    overlap = find_keys_that_are_also_values(json_file)

    print(f"Keys total: {len(overlap)}")
    print("-" * 50)

    for case in overlap:
        print(case)


if __name__ == "__main__":
    main()