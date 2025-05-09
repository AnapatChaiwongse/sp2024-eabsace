"""
This module converts an original dataset into a properly formatted instruct-dataset.
"""
import csv

def convert_atsc(input_sentence:str, line_id:str):
    """
    This function convert the original data type into the aspe subtask format.
    :input: <sentence>\x01The sentiment polarity of <aspect> is <sentiment>
    :return:    raw_text: <sentence>,
                aspectTerms: [
                    {
                        'term': <aspect>,
                        'polarity': <sentiment>,
                    }
                ]
    """
    try:
        cutpart = "The sentiment polarity of"
        sentence, label = input_sentence.split("\x01")
        label = label.replace(cutpart, "").strip()
        aspect, sentiment = label.split(" is ")
        sentiment = sentiment.replace(".", "").strip()
        output = {
            'raw_text': sentence.strip(),
            'aspectTerms': [
                {
                    'term': aspect,
                    'polarity': sentiment
                }
            ]
        }
        return output
    except ValueError:
        print("ValueError at line: " + line_id + "\nSentence: " + input_sentence)
        return {
                    'error': 'valueError',
                    'sentence': input_sentence
                }
MAX_FOLD = 5
DIRECTORY = "coursera"
FILENAME = "coursera_test"

for idx in range(MAX_FOLD):
    LOOPED_FILENAME = FILENAME # + str(idx+1)
    with open(f"{DIRECTORY}/{LOOPED_FILENAME}.csv", "w", encoding='utf-8', newline='') as csv_file:
        print("CURRENT FOLD: " + str(idx))
        filewritter = csv.writer(csv_file)
        filewritter.writerow(['raw_text', 'aspectTerms'])
        with open(f"../{DIRECTORY}/{LOOPED_FILENAME}.txt", "r", encoding="utf-8") as db_file:
            for sentence_id, line in enumerate(db_file.readlines()):
                formatted_line = convert_atsc(line, str(sentence_id))
                filewritter.writerow([formatted_line['raw_text'], formatted_line['aspectTerms']])
