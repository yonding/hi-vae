# IMPORT MODULES #
import itertools
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


def mv_generate(max_remove_count=3):

    # COMPLETE DATA #
    wine = load_wine()

    X_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    y_df = pd.DataFrame(wine.target, columns=["target"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    X_df_scaled = pd.DataFrame(X_scaled, columns=X_df.columns)

    complete_df = pd.concat([X_df_scaled, y_df], axis=1)
    complete_df

    # SPARSE DATA (excluding complete data) #
    features = [col for col in complete_df.columns if col != "target"]

    new_rows = []

    for index, row in complete_df.iterrows():
        for r in range(
            1, max_remove_count
        ):  # more than one feature should be removed and more than one feature should be kept
            for subset in itertools.combinations(features, r):
                new_row = row.copy()
                new_row[list(subset)] = 0
                new_row["origin_index"] = index  # 원본 행의 인덱스를 저장
                new_rows.append(new_row)

    sparse_df = pd.concat(new_rows, ignore_index=True, axis=1).T
    sparse_df["origin_index"] = sparse_df["origin_index"].fillna(
        pd.Series(sparse_df.index)
    )
    sparse_df[["target", "origin_index"]] = sparse_df[
        ["target", "origin_index"]
    ].astype(int)

    return sparse_df, complete_df
