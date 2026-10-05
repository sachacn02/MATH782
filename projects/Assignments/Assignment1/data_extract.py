import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "Textual_Data"
OUTPUT_DIR = PROJECT_ROOT / "src" / "Assigments" / "Assignment1" / "data"


def is_number(value):
    try:
        float(value)
    except ValueError:
        return False
    return True


def convert_file(input_path, output_path):
    with input_path.open("r", encoding="utf-8") as input_file:
        rows = [line.split() for line in input_file if line.split()]

    if not rows:
        return

    has_header = any(not is_number(value) for value in rows[0])
    if has_header:
        header = rows.pop(0)
    else:
        header = [f"column_{index}" for index in range(1, len(rows[0]) + 1)]

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for input_path in sorted(INPUT_DIR.glob("*.txt")):
        output_path = OUTPUT_DIR / f"{input_path.stem}.csv"
        convert_file(input_path, output_path)
        print(f"Created {output_path}")


if __name__ == "__main__":
    main()
