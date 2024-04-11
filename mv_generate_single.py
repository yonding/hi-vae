# IMPORT MODULES #
import itertools
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


def mv_generate_single(max_remove_count=3, new_num_per_origin=500, row_num=1):

    # COMPLETE DATA #
    wine = load_wine()

    X_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    y_df = pd.DataFrame(wine.target, columns=["target"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    X_df_scaled = pd.DataFrame(X_scaled, columns=X_df.columns)

    complete_df = pd.concat([X_df_scaled, y_df], axis=1)
    complete_df

    # SPARSE DATA (including complete data) #
    new_x_rows = []
    new_z_rows = []

    for index, row in complete_df.iterrows():
        new_x_row = row.copy()
        new_x_row.iloc[row_num] = 0
        new_x_rows.append(new_x_row)
        new_z_rows.append(complete_df.loc[index])

    x_df = pd.concat(new_x_rows, ignore_index=True, axis=1).T
    z_df = pd.concat(new_z_rows, ignore_index=True, axis=1).T

    # x_df = pd.concat([x_df, complete_df], ignore_index=True)
    # z_df = pd.concat([z_df, complete_df], ignore_index=True)

    class_df = x_df["target"].astype(int)
    x_df = x_df.drop("target", axis=1)
    z_df = z_df.drop("target", axis=1)

    return x_df, z_df, class_df
