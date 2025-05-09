import re

with open("output/test.txt", "r", encoding="utf-8") as f:
    for line in f:
        # Replace any number followed by a dot and space (e.g., "1. ", "12. ")
        line = re.sub(r"\d+\.\s", "", line)
        print(line, end='')  # 'end' prevents adding an extra newline since `line` already has one
