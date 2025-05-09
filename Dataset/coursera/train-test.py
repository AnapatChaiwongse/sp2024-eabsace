file = open('recoursera.txt', 'r', encoding='utf-8')
file = file.readlines()

labels = ["positive", "negative", "neutral"]

positive = []
negative = []
neutral = []

def _split_data(data:list):
    ratio = 0.2
    count = int(len(data) * ratio)
    test_data = data[:count]
    train_data = data[count:]
    return train_data, test_data

for line in file:
    if line.split("\x01")[1].split(" is ")[1].replace(" .", "").strip() == "positive":
        positive.append(line)
    elif line.split("\x01")[1].split(" is ")[1].replace(" .", "").strip() == "negative":
        negative.append(line)
    else:
        neutral.append(line)

train_data_pos, test_data_pos = _split_data(positive)
train_data_neg, test_data_neg = _split_data(negative)
train_data_neu, test_data_neu = _split_data(neutral)

train_data = train_data_pos + train_data_neg + train_data_neu
test_data = test_data_pos + test_data_neg + test_data_neu

print(f"Total data: {len(file)}")
print(f"# of train: {len(train_data)}\n# of test: {len(test_data)}\nTotal: {len(train_data + test_data)}")

with open("coursera_train.txt", "w", encoding="utf-8") as f:
    f.writelines(train_data)
with open("coursera_test.txt", "w", encoding="utf-8") as f:
    f.writelines(test_data)
