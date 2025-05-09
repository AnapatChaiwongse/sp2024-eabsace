"""
This py script is used to preprocess the "review.csv"
"""
import csv
import sys
csv.field_size_limit(sys.maxsize)

def _replace_label(score: int):
    try:
        if not isinstance(score, int):
            score = int(score)
        if not score:
            return "neutral"
        if score > 3:
            return "positive"
        elif score < 3:
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        print(str(e) + ": " + str(score) + "Type: " + str(type(score)))
        sys.exit(1)

csvfile = []
with open('reviews.csv', encoding='utf-8', newline='') as file:
    reader = csv.reader(file)
    for idx, i in enumerate(reader):
        if idx == 0:
            pass
        else:
            csvfile.append(i[1] + "The sentiment of this sentence is " + _replace_label(i[2]) + " .")

with open('clean_coursera.txt', 'a', encoding='utf-8') as file:
    for idx, i in enumerate(csvfile):
        file.writelines(i+'\n')
