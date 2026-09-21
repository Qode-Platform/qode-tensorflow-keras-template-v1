"""Training entrypoint: `python -m src.train --epochs 5`."""

import argparse

import keras

from src.data import datasets
from src.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default="artifacts/model.keras")
    args = parser.parse_args()

    train, test = datasets(args.batch_size)
    model = build_model()
    model.fit(
        train,
        validation_data=test,
        epochs=args.epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(args.out, save_best_only=True),
        ],
    )
    print(model.evaluate(test, return_dict=True))


if __name__ == "__main__":
    main()
