# with open("../../Dataset/instructed/original/org-fold1.csv", "r", encoding="utf-8") as f:
#     for line in f:
#         print(line)

import pandas as pd

# # Load CSV
df = pd.read_csv("../../Dataset/instructed/original/org-fold1.csv")

# Check for NaN values in each column
print(df.isna().sum())
print(df[~df['aspectTerms'].apply(lambda x: isinstance(x, (str, list)))])
