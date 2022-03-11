from tqdm import tqdm
import pandas as pd
import pickle
tqdm.pandas()


def maphex2int(collection, path="ids.pkl"):
    ids = {str(id): i+1 for i, id in enumerate(
        collection.find().distinct('_id'))}
    with open(path, 'wb') as f:
        pickle.dump(ids, f)
    return ids


def database2csv(collection, path, hex2int):
    df = pd.DataFrame(columns=["rater", "rated", "r"])
    for i in tqdm(collection.find()):
        rater = hex2int[str(i['_id'])]
        right_swipes = i['liked_users']
        for swipes in right_swipes:
            rated = hex2int[str(swipes['swipee_id'])]
            df = pd.concat(
                [df, pd.DataFrame([[rater, rated, 1.0]], columns=['rater', 'rated', 'r'])])

        left_swipes = i['disliked_users']
        for swipes in left_swipes:
            rated = hex2int[str(swipes['swipee_id'])]
            df = pd.concat(
                [df, pd.DataFrame([[rater, rated, 0.0]], columns=['rater', 'rated', 'r'])])
    df = df.astype({'rater': int, 'rated': int})
    df.to_csv(path, index=False)


def matches_to_matches_triplet(data_path, output_path):
    data = pd.read_csv(data_path, dtype={"rated": str})  # rater, rated, m
    A_col, B_col, m_col = data.columns
    # keep only matches
    print(f"N : {data.shape[0]}")
    data = data[data[m_col] > 0]
    print(f"N : {data.shape[0]} (matches only)")
    # group rated id into a set
    data = data.groupby(A_col)[[B_col]].agg(list)
    data.to_csv(output_path, index=False)
