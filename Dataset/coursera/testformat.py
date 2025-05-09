file = open("coursera_test.txt", "r", encoding="utf-8")
file = file.readlines()

sentence = []
sentiment = []
formatted_sentiment = []

for line in file:
    sentence.append(line.split("\x01")[0])
    sentiment.append(line.split("\x01")[1].split(" is ")[1].replace(" .", "").strip())
# print(str(len(sentence)) + " + " + str(len(sentiment)))
# print(sentiment)

for line in range(len(file)):
    formatted_sentiment.append(sentence[line]+"\x01"+sentiment[line]+"\n")

with open("testformat.txt", "w", encoding="utf-8") as f:
    f.writelines(formatted_sentiment)
