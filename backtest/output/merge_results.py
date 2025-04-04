import pickle
# loop all the pkl files in the subfolder backup
# and merge the results
import os
import pandas as pd

all_data = []
for root, dirs, files in os.walk("backtest/output/selected_5/FinMemStrategy/backup"):
    for file in files:
        print(os.path.join(root, file))
        if file.endswith(".pkl"):
            with open(os.path.join(root, file), "rb") as f:
                data = pickle.load(f)
                all_data.append(data)

data = all_data[0]

# merge all the results
merged_data = {}
for data in all_data:
    merged_data.update(data)

# sort the dictionary by key
merged_data = dict(sorted(merged_data.items()))

# save the merged data
with open("2004-01-01_2024-01-01.pkl", "wb") as f:
    pickle.dump(merged_data, f)