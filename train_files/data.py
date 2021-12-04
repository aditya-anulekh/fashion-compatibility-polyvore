import pandas as pd
from itertools import combinations
from tqdm import tqdm

def generate_pairs(compat_file, outfits_file):
    names = [f"item_{i}" for i in range(19)]
    names.insert(0, "compat")
    data = pd.read_csv(compat_file,
                        delim_whitespace=True,
                        names=names)

    items = pd.read_json(outfits_file)

    X = []
    y = []
    for i in tqdm(range(len(data))):
        for comb in combinations(data.iloc[i,1:].dropna().values, 2):
            images = []
            for image in comb:
                item_id, idx = image.split("_")
                item_id, idx = int(item_id), int(idx)
                image = [x["item_id"] for x in list(items[items["set_id"] == item_id]["items"].iloc[0]) if x["index"] == idx][0]
                image = f"{image}.jpg"
                images.append(image)
            X.append(images)
            y.append(data.iloc[i,0])

    print(len(X))
    print(len(y))
    return X,y
