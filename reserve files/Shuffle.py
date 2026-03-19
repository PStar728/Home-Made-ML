import pandas as pd


df = pd.read_csv("winequality-red.csv", sep=';')

# 2. Shuffle the entire thing
# frac=1 means 100% of the rows. random_state=42 makes it reproducible.
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Save to a new file
df_shuffled.to_csv("wine_shuffled.csv", index=False, sep=';')

print("Done! Use wine_shuffled.csv for your ML project now.")