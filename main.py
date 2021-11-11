#!/usr/bin/env python3

from pathlib import Path
import argparse
from datetime import datetime
import json

import matplotlib.pyplot as plt
import tensorflow as tf

import gen

BALL_IMG_PATH = Path("./res/ball.jpg")
VAL2017_FOLDER_PATH = Path("/home/mbernardi/extra/async/ipcv/sem_3/deep_learning/labs/5/val2017")

# Version of the format where I save the models, outputs, etc
DATA_VERSION = "v1"

CASES_PATH = Path(f"./cases/{DATA_VERSION}/")


class Case:

    def __init__(self):
        """
        Represents a training trial.

        Contains a model, all hyperparameters, a description, etc.
        """

        # ID of case
        self.id = datetime.now().isoformat(timespec="seconds")

        # Samples per batch
        self.batch_size = 16

        # Number of batches per epoch
        self.num_batches = 3

        # Number of epochs
        self.num_epochs = 10

        # Optimizer and loss, use Keras objects and not strings
        self.lr = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.loss = tf.keras.losses.BinaryCrossentropy()

        # Model
        self.model = tf.keras.Sequential()

        # Notes
        self.notes = ""

        # Define model

        self.model.add(tf.keras.layers.Conv2D(
                32, (3, 3),
                activation='relu', padding='same',
                #, input_shape=(32, 32, 3)))
            ))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(
                64, (3, 3),
                activation='relu', padding='same',
            ))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        # self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        # self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2DTranspose(
                16, (3, 3), strides=2,
                activation='sigmoid', padding='same',
            ))
        self.model.add(tf.keras.layers.Conv2DTranspose(
                1, (3, 3), strides=2,
                activation='sigmoid', padding='same',
            ))

        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def save_description(self, json_path):
        """
        Saves a JSON with a description of the case.
        """

        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        summary = "\n".join(summary)

        data = {
                "id": self.id,
                "batch_size": self.batch_size,
                "num_batches": self.num_batches,
                "lr": self.lr,
                "num_epochs": self.num_epochs,
                "optimizer": str(self.optimizer),
                "loss": str(self.loss),
                "notes": self.notes,
                "model_summary": summary,
            }

        with open(json_path, "w") as f:
            json.dump(data, f)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t",
            action="store_true",
            help="Train network",
        )
    parser.add_argument("--eval", "-e",
            type=str,
            help="Evaluate network",
        )
    parser.add_argument("--list", "-l",
            action="store_true",
            help="List cases",
        )

    args = parser.parse_args()

    if args.train:

        case = Case()
        case_path = CASES_PATH / case.id
        case_path.mkdir(parents=True)

        checkpoint_path = case_path / "cp.ckpt"

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                verbose=1
            )

        train_dataset = tf.data.Dataset.from_generator(
                lambda: gen.image_generator(
                    VAL2017_FOLDER_PATH,
                    BALL_IMG_PATH,
                    batch_size=case.batch_size,
                    num_batches=case.num_batches,
                    shuffle=True,
                ),
                output_signature=(
                    tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
                )
            )

        val_dataset = tf.data.Dataset.from_generator(
                lambda: gen.image_generator(
                    VAL2017_FOLDER_PATH,
                    BALL_IMG_PATH,
                    batch_size=case.batch_size,
                    num_batches=case.num_batches,
                    shuffle=False,
                ),
                output_signature=(
                    tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
                )
            )

        case.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=case.num_epochs,
                verbose=1,
                callbacks=[cp_callback],
            )

        case.model.save(case_path)
        case.save_description(case_path / "case.json")

    if args.eval:
        case_id = args.eval

        case_path = CASES_PATH / case_id
        model = tf.keras.models.load_model(case_path)

        image_gen = gen.image_generator(VAL2017_FOLDER_PATH, BALL_IMG_PATH, batch_size=16, shuffle=False)
        x, y = next(image_gen)
        img = x[0]
        y_est = model.predict(x)
        plt.imshow(y_est[0].squeeze())
        plt.show()

    if args.list:
        for case_path in sorted(CASES_PATH.iterdir()):
            print(case_path.name)
            with open(case_path / "case.json", "r") as f:
                data = json.load(f)

                print(data["model_summary"])

if __name__ == "__main__":
    main()
