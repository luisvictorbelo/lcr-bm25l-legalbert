import json
import statistics
import os

def main():
    file_path = 'data/labels/task1_test_labels_2025.json'
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Calculate the number of candidate cases for each query
    counts = [len(candidates) for candidates in data.values()]

    if not counts:
        print("No queries found in the file.")
        return

    min_count = min(counts)
    max_count = max(counts)
    median_count = statistics.median(counts)
    mean_count = statistics.mean(counts)

    print(f"Statistics for candidate cases in {file_path}:")
    print(f"Total Queries: {len(counts)}")
    print(f"Minimum: {min_count}")
    print(f"Maximum: {max_count}")
    print(f"Median:  {median_count}")
    print(f"Mean:    {mean_count:.2f}")

if __name__ == "__main__":
    main()
