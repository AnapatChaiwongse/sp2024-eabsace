import os
import openai
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

client = OpenAI(
    api_key=os.getenv("API_KEY")
)
# print(client.models.list())

example = "This subject is really interesting but the professor is really bad. The sentiment polarity of motivation is positive. The sentiment polarity of instructor is negative."
messages = [
    {"role": "system", 
     "content": 
        "Generate the sentence with multi-aspect in the topic of faculty of technology course evaluation."
        "The generated content can be whatever the related to the student comment."
        "The aspect contain: Instuctor, Content quality, Time management, Motivation"
        "The sentiment contain: Positive, Negative, and Neutral"
        "For the generated sentence where some aspects didn't appear, add it and make it neutral."
        "Return as: sentence + \x01 + aspect1 + \x01 + sentiment1 + \x01 + aspect2 + \x01 + sentiment2 + ..."},
    {"role": "user", "content": "generate 5 sentence"}
]

try:
    responses = client.chat.completions.create(
        model="gpt-4o-mini", # system, user, developer, assistant
        messages=messages
    )
    result = responses.choices[0].message.content
    token_usage = responses.usage
    print(bcolors.OKBLUE + str(token_usage) + bcolors.ENDC)
    print(result)
    
    with open("output/test.txt", "w") as f:
        f.writelines(result)
except openai.RateLimitError as e:
    print(f"Error: {e}")
