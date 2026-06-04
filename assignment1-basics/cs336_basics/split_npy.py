import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--train-output", type=str, required=True)
    parser.add_argument("--valid-output", type=str, required=True)
    parser.add_argument("--valid-ratio", type=float, default=0.01)
    args = parser.parse_args()

    data = np.load(args.input, mmap_mode="r")

    n = len(data)
    valid_size = int(n * args.valid_ratio)
    train_size = n - valid_size

    print(f"total tokens: {n}")
    print(f"train tokens: {train_size}")
    print(f"valid tokens: {valid_size}")

    train_data = np.asarray(data[:train_size])
    valid_data = np.asarray(data[train_size:])

    np.save(args.train_output, train_data)
    np.save(args.valid_output, valid_data)

    print(f"saved train to {args.train_output}")
    print(f"saved valid to {args.valid_output}")


if __name__ == "__main__":
    main()
