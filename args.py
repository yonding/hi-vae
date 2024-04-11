import argparse
import torch


# PARSE ARGUMENTS
def parse_args():
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("--f")
    parser.add_argument(
        "--max_remove_count",        
        default=5,
        type=int,
        help="Set how many features can be removed.",
    )
    parser.add_argument(
        "--new_num_per_origin",
        default=1000,
        type=int,
        help="Set how many new rows you want to create per origin row.",
    )
    parser.add_argument(
        "--epochs", default=10000, type=int, help="Set the number of epochs."
    )
    parser.add_argument("--val_size", default=0.2, type=float, help="Set the val size.")
    parser.add_argument(
        "--test_size", default=0.1, type=float, help="Set the test size."
    )
    parser.add_argument(
        "--random_state", default=328, help="Set the random state for shuffling."
    )
    parser.add_argument("--H", default=2048, help="Set the first hidden dimension.")
    parser.add_argument("--H2", default=1024, help="Set the second hidden dimension.")
    parser.add_argument("--latent_dim", default=3, help="Set the latent dimension.")
    parser.add_argument("--batch_size", default=1024, help="Set the batch size.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    parser.add_argument("--device", default=device, help="Set the device.")

    return parser.parse_args()
