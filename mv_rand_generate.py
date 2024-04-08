# IMPORT MODULES #
import random
import itertools
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


def mv_rand_generate(max_remove_count=5, new_num_per_origin=100):
    """
    Input: max_remove_count, new_num_per_origin
    Output: rand_sparse_df, complete_df
    """
    # COMPLETE DATA #
    wine = load_wine()

    X_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    y_df = pd.DataFrame(wine.target, columns=["target"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    X_df_scaled = pd.DataFrame(X_scaled, columns=X_df.columns)

    complete_df = pd.concat([X_df_scaled, y_df], axis=1)
    complete_df

    # FEATURE COMBINATIONS #
    features = [col for col in complete_df.columns if col != "target"]
    feature_combinations = []

    for r in range(1, max_remove_count + 1):
        feature_combinations += list(itertools.combinations(features, r))

    # RANDOM SPARSE DATA (excluding complete data) #
    features = [col for col in complete_df.columns if col != "target"]

    new_x_rows = []
    new_z_rows = []

    for index, row in complete_df.iterrows():
        random_combinations = random.sample(feature_combinations, new_num_per_origin)
        for subset in random_combinations:
            new_x_row = row.copy()
            new_x_row[list(subset)] = 0
            new_x_rows.append(new_x_row)
            new_z_rows.append(complete_df.loc[index])

    x_df = pd.concat(new_x_rows, ignore_index=True, axis=1).T
    z_df = pd.concat(new_z_rows, ignore_index=True, axis=1).T
    class_df = x_df["target"].astype(int)
    x_df = x_df.drop("target", axis=1)
    z_df = z_df.drop("target", axis=1)

    return x_df, z_df, class_df
