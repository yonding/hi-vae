from sklearn.model_selection import train_test_split
import torch


def split_and_convert_to_tensor(x_df, z_df, y_df, val_size, test_size, random_state):

    x_train, x_temp, y_train, y_temp, z_train, z_temp = train_test_split(
        x_df,
        y_df,
        z_df,
        test_size=val_size + test_size,
        random_state=328,
    )

    x_val, x_test, y_val, y_test, z_val, z_test = train_test_split(
        x_temp,
        y_temp,
        z_temp,
        test_size=test_size / (val_size + test_size),
        random_state=random_state,
    )

    x_train = torch.tensor(x_train.values, dtype=torch.float32)
    x_val = torch.tensor(x_val.values, dtype=torch.float32)
    x_test = torch.tensor(x_test.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.int64)
    y_val = torch.tensor(y_val.values, dtype=torch.int64)
    y_test = torch.tensor(y_test.values, dtype=torch.int64)
    z_train = torch.tensor(z_train.values, dtype=torch.float32)
    z_val = torch.tensor(z_val.values, dtype=torch.float32)
    z_test = torch.tensor(z_test.values, dtype=torch.float32)

    return x_train, x_val, x_test, y_train, y_val, y_test, z_train, z_val, z_test
