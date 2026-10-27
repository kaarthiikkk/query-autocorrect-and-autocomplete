import csv

INPUT_FILE = "data/tamil.csv"
OUTPUT_FILE = "data/tamil_small.csv"

MAX_WORDS = 50000


with open(INPUT_FILE, "r", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as outfile:

        writer = csv.writer(outfile)

        # Keep the same format as the original dataset
        writer.writerow(["word", "frequency"])

        count = 0

        for row in reader:
            word = row.get("word", "").strip()
            frequency = row.get("frequency", "1").strip()

            if not word:
                continue

            writer.writerow([word, frequency])

            count += 1

            if count >= MAX_WORDS:
                break


print(f"Created {OUTPUT_FILE}")
print(f"Words saved: {count}")