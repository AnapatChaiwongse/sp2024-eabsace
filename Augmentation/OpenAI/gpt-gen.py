import os
import sys
import re
from openai import OpenAI

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

FOLD = "5"
INPUT_PATH = f"../../Datasets-{FOLD}-fold/Original/splited-{FOLD}.txt"
OUTPUT_PATH = f"outputs/splited-{FOLD}/gpt4o-mini-splited-{FOLD}.txt"
texts = []
sentiments = []

api_key = os.getenv("OPENAI_API_KEY", None)
if not api_key:
    print("OPENAI_API_KEY not found.")
    sys.exit(1)

client = OpenAI(
    api_key=api_key
)

with open(INPUT_PATH, "r", encoding='utf-8') as file:
    for line in file:
        splited = line.split('')
        texts.append(splited[0])
        sentiments.append(splited[1])

outputs = []
ORDER_COUNT = 1

for init in texts:
    print(f"{ORDER_COUNT}/{len(texts)}")
    print( f"Initial Sentence: {init}")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content":  "Generate sentences of multiaspect related to Faculty of Information and Technology course evaluation from the initial sentence."
                                f"initial sentence = {init}"
                                "The content should be related to student comments. Ensure that the generated content provides different sentiments for one or more aspects compared to the initial sentence."
                                "Aspects: instructor, content quality, motivation, organization."
                                "Sentiment: positive, negative, neutral."
                                "If an aspect is not directly mentioned, mark it as neutral."
                                "Use the simple vocab and sentence structure "
                                "Return the output in the format: sentence \x01 aspect1 \x01 sentiment1 \x01 aspect2 \x01 sentiment2 \x01 aspect3 \x01 sentiment3 \x01 aspect4 \x01 + sentiment4."
                },
                {
                    "role": "user",
                    "content": "Generate 3 sentences with different sentiments."
                }
            ]
        )
        result = response.choices[0].message.content
        outputs.append(result)
        token_usage = response.usage
        ORDER_COUNT += 1

        # print(bcolors.OKBLUE + f"Token Usage: {token_usage}" + bcolors.ENDC)
        # print("Generated Sentences:")
        # print(result)
        # print("\n" + "-"*50 + "\n")

    except Exception as e:
        print(f"An error occurred: {e}")

no_num =[]

for line in outputs:
    line = re.sub(r"\d+\.\s", "", line)
    no_num.append(line)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    for line in no_num:
        parts = line.split("\x01")
        text = parts[0].strip()

        for j in range(1, len(parts), 2):
            aspect = parts[j].strip()
            polarity = parts[j + 1].strip()
            final_line = f"{text}The sentiment polarity of {aspect} is {polarity} ."
            file.write(final_line + "\n")
