"""
This py script is used the rebalance the "reviews.csv"
"""

import random

data = []
too_short = []
pos_list = []
neg_list = []
neu_list = []

with open('clean_coursera.txt', encoding='utf-8') as file:
    for idx, i in enumerate(file):
        data.append(i)
        sentence = i.split("")[0]
        sentiment = i.split("")[1]
        if sentiment.split(" is ")[1].split(" .")[0] == "negative":
            neg_list.append(i)
        elif sentiment.split(" is ")[1].split(" .")[0] == "positive":
            pos_list.append(i)
        elif sentiment.split(" is ")[1].split(" .")[0] == "neutral":
            neu_list.append(i)
        if len(sentence.split(" ")) < 5:
            too_short.append(idx)
# for idx in too_short:
#     print(data[idx], end='')
print(f"data length: {len(data)}")
print(f"too_short lenght: {len(too_short)}")
print("="*20)
print(f"Positive: {len(pos_list)}")
print(f"Negative: {len(neg_list)}")
print(f"Neutral: {len(neu_list)}")
print(f"""{len(pos_list)} + {len(neg_list)} + {len(neu_list)} = {len(pos_list) + len(neg_list) + len(neu_list)}""")

pos_list = []
neg_list = []
neu_list = []

for i in reversed(too_short):
    data.pop(i)

for idx, i in enumerate(data):
    sentence = i.split("")[0]
    sentiment = i.split("")[1]
    if sentiment.split(" is ")[1].split(" .")[0] == "negative":
        neg_list.append(i)
    elif sentiment.split(" is ")[1].split(" .")[0] == "positive":
        pos_list.append(i)
    elif sentiment.split(" is ")[1].split(" .")[0] == "neutral":
        neu_list.append(i)
    if len(sentence.split(" ")) < 5:
        too_short.append(idx)
print("="*20)
print(f"Positive: {len(pos_list)}")
print(f"Negative: {len(neg_list)}")
print(f"Neutral: {len(neu_list)}")
print(f"""{len(pos_list)} + {len(neg_list)} + {len(neu_list)} = {len(pos_list) + len(neg_list) + len(neu_list)}""")
print("="*20)

new_pos_list = []
temp = None
for i in range(4400):
    rand_num = random.randint(0, len(pos_list))
    if rand_num == temp:
        rand_num = random.randint(0, len(pos_list))
    else:
        temp = rand_num
        new_pos_list.append(pos_list[rand_num])
new_pos_list = list(set(new_pos_list))
print(f"New positive: {len(new_pos_list)}")
print(f"Negative: {len(neg_list)}")
print(f"Neutral: {len(neu_list)}")
print(f"""{len(new_pos_list)} + {len(neg_list)} + {len(neu_list)} = {len(new_pos_list) + len(neg_list) + len(neu_list)}""")

with open("recoursera.txt", "a", encoding='utf-8') as file:
    for i in new_pos_list:
        file.writelines(i)
    for i in neg_list:
        file.writelines(i)
    for i in neu_list:
        file.writelines(i)
