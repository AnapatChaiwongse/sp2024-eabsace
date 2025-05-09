# file_path = ""
data = {
    "The instructor was engaging and supportive, but the content quality felt slightly outdated, the time management could be improved, and the motivation provided was adequate. \x01 Instructor \x01 Positive \x01 Content quality \x01 Neutral \x01 Time management \x01 Neutral \x01 Motivation \x01 Neutral",
    "While the course content was well-structured and insightful, the instructor's explanations were sometimes unclear, and the motivation to attend was minimal, though the time management was acceptable. \x01 Content quality \x01 Positive \x01 Instructor \x01 Negative \x01 Time management \x01 Neutral \x01 Motivation \x01 Negative",
    "The instructor was excellent in delivering the course, the content quality was outstanding, and the motivation to stay engaged was high, but time management needs slight improvement. \x01 Instructor \x01 Positive \x01 Content quality \x01 Positive \x01 Time management \x01 Neutral \x01 Motivation \x01 Positive",
    "The motivation provided during the sessions was encouraging, but the content quality was average, and the time management and instructor's teaching style were neither exceptional nor problematic. \x01 Motivation \x01 Positive \x01 Content quality \x01 Neutral \x01 Time management \x01 Neutral \x01 Instructor \x01 Neutral",
    "The course had good time management and inspiring motivation, although the instructor's interaction was limited, and the content quality could use some enhancements. \x01 Time management \x01 Positive \x01 Motivation \x01 Positive \x01 ???Instructor \x01 Neutral \x01 Content quality \x01 Neutral"
}

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

# data = {
#     "The instructor was engaging and supportive, but the content quality felt slightly outdated, the time management could be improved, and the motivation provided was adequate. \x01 Instructor \x01 Positive \x01 Content quality \x01 Neutral \x01 Time management \x01 Neutral \x01 Motivation \x01 Neutral"
# }
prs_data = []

# The sentiment polarity of """" is """"

for line in data:
    cleaned_sentence = line.split(". ")[0] + "."
    sentiment_aspect = line.split(". ")[1].split(" \x01 ")
    for idx, i in enumerate(sentiment_aspect):
        if i == "":
            sentiment_aspect.pop(idx)
    print(bcolors.FAIL + str(sentiment_aspect) + bcolors.ENDC)
    idx = 0
    while idx in range(int(len(sentiment_aspect))):
        sentiment = sentiment_aspect[idx]
        aspect = sentiment_aspect[idx + 1]
        
        cleaned_line = cleaned_sentence + "This sentiment polarity of " + aspect + " is " + sentiment + " . "
        print(bcolors.OKBLUE + cleaned_sentence + bcolors.ENDC)
        print("Sentiment: " + bcolors.OKGREEN + sentiment + bcolors.ENDC)
        print("Aspect: " + bcolors.OKCYAN + aspect + bcolors.ENDC)
        prs_data.append(cleaned_line)
        idx += 2

for i in prs_data:
    print(i)

# with open(file_path, "r", encoding="utf-8") as file:
#     for line in file:
#         data.append(line)
