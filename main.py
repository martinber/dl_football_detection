#!/usr/bin/env python3

from pathlib import Path
import argparse
from datetime import datetime
import json

import matplotlib.pyplot as plt

import gen

BALL_IMG_PATH = Path("./res/ball.jpg")
VAL2017_FOLDER_PATH = Path("/home/mbernardi/extra/async/ipcv/sem_3/deep_learning/labs/5/val2017")

# Version of the format where I save the models, cases, etc. If in the future
# I change the format I can just change the string, so a new folder will be made
# and old things will be left ignored in old folder
DATA_VERSION = "v1"
CASES_PATH = Path(f"./cases/{DATA_VERSION}/")


class Case:

    def __init__(self):
        """
        Represents a training trial.

        Contains a model, all hyperparameters, a description, etc.
        """

        import tensorflow as tf

        # ID of case
        self.id = datetime.now().isoformat(timespec="seconds")

        # Samples per batch
        self.batch_size = 16

        # Number of batches per epoch
        self.num_batches = 3

        # Number of epochs
        self.num_epochs = 100

        # Optimizer and loss, use Keras objects and not strings
        self.lr = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.loss = tf.keras.losses.BinaryCrossentropy()

        # Parameters of the data generator. Can only contain serializable things
        self.gen_params = {
                "ground_truth_shape": "rect",
            }

        # Model
        self.model = tf.keras.Sequential()

        # Notes
        self.notes = ""

        # Define model

        self.model.add(tf.keras.layers.Conv2D(
                32, (3, 3),
                # dilation_rate=4,
                activation='relu', padding='same',
                #, input_shape=(32, 32, 3)))
            ))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(
                64, (3, 3),
                # dilation_rate=4,
                activation='relu', padding='same',
            ))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        # self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        # self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2DTranspose(
                16, (3, 3), strides=2,
                # dilation_rate=4,
                activation='sigmoid', padding='same',
            ))
        self.model.add(tf.keras.layers.Conv2DTranspose(
                1, (3, 3), strides=2,
                # dilation_rate=4,
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
                "gen_params": self.gen_params,
                "notes": self.notes,
                "model_summary": summary,
            }

        with open(json_path, "w") as f:
            json.dump(data, f)

def train(args):

    import tensorflow as tf

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
                params=case.gen_params,
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
                params=case.gen_params,
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

def eval_(args):

    import tensorflow as tf

    case_id = args.id

    case_path = CASES_PATH / case_id
    model = tf.keras.models.load_model(case_path)

    with open(case_path / "case.json", "r") as f:
        case_description = json.load(f)

    image_gen = gen.image_generator(
            VAL2017_FOLDER_PATH,
            BALL_IMG_PATH,
            batch_size=16,
            shuffle=True,
            params=case_description["gen_params"],
        )
    x, y = next(image_gen)
    img = x[0]
    ground_truth = y[0]
    y_est = model.predict(x)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    aximg = ax1.imshow(img)
    fig.colorbar(aximg, ax=ax1)
    ax1.set_title("Input")

    aximg = ax2.imshow(ground_truth.squeeze())
    fig.colorbar(aximg, ax=ax2)
    ax2.set_title("Ground truth")

    aximg = ax3.imshow(y_est[0].squeeze())
    fig.colorbar(aximg, ax=ax3)
    ax3.set_title("Output")

    aximg = ax4.imshow(y_est[0].squeeze()[1:-1, 1:-1])
    fig.colorbar(aximg, ax=ax4)
    ax4.set_title("Output cropped")

    plt.show()

def list_(args):

    for case_path in sorted(CASES_PATH.iterdir()):

        if not (case_path / "saved_model.pb").is_file():
            # Training didn't finish in this case
            continue

        with open(case_path / "case.json", "r") as f:
            data = json.load(f)


            if args.filter:
                if not str(data[args.filter[0]]) == args.filter[1]:
                    continue

            if args.verbose:
                print("------------------------")
                for key, value in data.items():
                    print(key, ":", value)

            else:
                print(data["id"])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser(
            "train",
            help=("Train case. It will create the model defined in the source "
                  "code, train it, and save everything in the \"cases\" folder: "
                  "the model, a json description, etc.")
        )
    parser_train.set_defaults(func=train)

    parser_eval = subparsers.add_parser(
            "eval",
            help=("Evaluate case. It will load the given model, evaluate it, "
                  "show plots, etc.")
        )
    parser_eval.add_argument("id",
            type=str,
            help="ID of case to evaluate",
        )
    parser_eval.set_defaults(func=eval_)

    parser_eval = subparsers.add_parser(
            "list",
            help=("List cases. It searchs all the saved cases/models and lists "
                  "them. Allows to filter results and select information to "
                  "show."))
    parser_eval.add_argument("--verbose", "-v",
            action="store_true",
            help="Show all information about case",
        )
    parser_eval.add_argument("--filter", "-f",
            type=str,
            nargs=2,
            help="Filter field of case description",
        )
    parser_eval.set_defaults(func=list_)

    args = parser.parse_args()
    # Call function corresponding to the selected subparser
    args.func(args)
