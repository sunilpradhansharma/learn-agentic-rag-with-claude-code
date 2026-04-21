"""
word_counter.py — counts words, lines, and word frequencies in a text file.

Usage:
    python word_counter.py <path-to-file>

Example:
    python word_counter.py sample.txt
"""

import argparse
import sys
from collections import Counter

# Words we don't want to count because they appear everywhere and tell us
# nothing interesting about the content of the document.
STOPWORDS = {
    "the", "a", "an", "is", "of", "and", "to", "in",
    "it", "that", "this", "for", "on", "with",
}


def parse_args():
    # argparse builds a command-line interface for us. We declare what
    # arguments we expect, and argparse handles --help, error messages,
    # and type conversion automatically.
    parser = argparse.ArgumentParser(
        description="Count words and lines in a text file."
    )
    parser.add_argument(
        "filepath",
        help="Path to the text file you want to analyze.",
    )
    return parser.parse_args()


def read_file(filepath):
    # Try to open the file. If it doesn't exist, print a helpful message
    # and exit rather than letting Python print a confusing traceback.
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # .readlines() returns a list where each element is one line,
            # newline character included. We'll use this list for the line
            # count, then join it back into a single string for word counting.
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: file not found — '{filepath}'")
        print("Check the path and try again.")
        sys.exit(1)

    return lines


def count_words(lines):
    # Join all lines into one big string so we can split on whitespace.
    full_text = " ".join(lines)

    # .split() with no arguments splits on any whitespace (spaces, tabs,
    # newlines) and discards empty strings — more robust than .split(" ").
    raw_words = full_text.split()

    # Total word count before filtering.
    total_words = len(raw_words)

    # Normalise to lowercase and remove stopwords so our frequency counts
    # reflect meaningful content words only.
    filtered_words = [
        word.lower().strip(".,!?;:\"'()-")  # strip punctuation from edges
        for word in raw_words
        if word.lower().strip(".,!?;:\"'()-") not in STOPWORDS
        and word.lower().strip(".,!?;:\"'()-") != ""  # guard against empty strings after strip
    ]

    return total_words, filtered_words


def main():
    args = parse_args()

    # --- Read the file ---
    lines = read_file(args.filepath)
    total_lines = len(lines)

    # --- Count words ---
    total_words, filtered_words = count_words(lines)

    # Counter takes any iterable and returns a dict-like object that maps
    # each unique item to the number of times it appears.
    word_counts = Counter(filtered_words)

    # .most_common(n) returns the n highest-count items as a list of
    # (word, count) tuples, sorted from most to least common.
    top_10 = word_counts.most_common(10)

    # .most_common() with no argument returns ALL items sorted by count.
    # Reversing gives us least-common first; we take the first 5 of those.
    # We only include words that appear at least once (the full list always
    # satisfies this, but it's good to be explicit).
    all_words_sorted = word_counts.most_common()  # most → least common
    bottom_5 = list(reversed(all_words_sorted))[:5]  # flip to least → most common

    # --- Print results ---
    print(f"\nFile: {args.filepath}")
    print("-" * 40)
    print(f"Total lines : {total_lines}")
    print(f"Total words : {total_words}")

    print("\nTop 10 most common words (stopwords excluded):")
    for rank, (word, count) in enumerate(top_10, start=1):
        # f-strings let us embed variables directly in a string.
        # The format spec :<15 left-aligns the word in a 15-character field
        # so the counts line up neatly.
        print(f"  {rank:>2}. {word:<15} {count}")

    print("\nBottom 5 least common words (stopwords excluded):")
    for rank, (word, count) in enumerate(bottom_5, start=1):
        print(f"  {rank:>2}. {word:<15} {count}")

    print()  # blank line at the end for cleaner terminal output


# This guard means: only run main() when this script is executed directly
# (e.g. `python word_counter.py`). If another script imports this file,
# main() will NOT run automatically.
if __name__ == "__main__":
    main()
