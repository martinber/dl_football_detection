from pathlib import Path
from datetime import datetime
import json

import tensorflow as tf

import gen

# TODO: Recwive as arguments
BALL_IMG_PATH = Path("./res/ball.jpg")
VAL2017_FOLDER_PATH = Path("/home/mbernardi/extra/async/ipcv/sem_3/deep_learning/labs/5/val2017")
DATA_VERSION = "v3"
CASES_PATH = Path(f"./cases/{DATA_VERSION}/")


class Case:

    def __init__(self, model, batch_size, num_batches, num_epochs,
            optimizer, loss, gen_params, notes):
        """
        Represents a training trial.

        Contains a model, all hyperparameters, a description, etc.
        """

        # ID of case
        self.id = datetime.now().isoformat(timespec="seconds")

        # Samples per batch
        self.batch_size = batch_size

        # Number of batches per epoch
        self.num_batches = num_batches

        # Number of epochs
        self.num_epochs = num_epochs

        # Optimizer and loss, use Keras objects and not strings
        self.optimizer = optimizer
        self.loss = loss

        # Parameters of the data generator. Can only contain serializable things
        self.gen_params = gen_params

        # Model
        self.model = model
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        # Notes
        self.notes = notes


    def save_description(self, json_path):
        """
        Saves a JSON with a description of the case.
        """

        # Dictionaries with configuration of layers
        layers_config = [l.get_config() for l in self.model.layers]

        # Save summary as string
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        summary = "\n".join(summary)

        data = {
                "id": self.id,
                "batch_size": self.batch_size,
                "num_batches": self.num_batches,
                "num_epochs": self.num_epochs,
                "optimizer": str(self.optimizer.get_config()),
                "loss": str(self.loss.get_config()),
                "gen_params": self.gen_params,
                "notes": self.notes,
                "layers_config": layers_config,
                "model_summary": summary,
            }

        with open(json_path, "w") as f:
            json.dump(data, f)

def train_case(case):

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
                evaluation=False,
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
                evaluation=False,
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


def train():

    import tensorflow as tf

    # Define model
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(
            64, (3, 3),
            # dilation_rate=2,
            activation='relu', padding='same',
        ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
    model.add(tf.keras.layers.Conv2D(
            64, (3, 3),
            # dilation_rate=2,
            activation='relu', padding='same',
        ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
    # model.add(tf.keras.layers.Conv2D(
    #         64, (3, 3),
    #         # dilation_rate=2,
    #         activation='relu', padding='same',
    #     ))
    # model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
    # model.add(tf.keras.layers.Conv2D(
    #         64, (3, 3),
    #         # dilation_rate=2,
    #         activation='relu', padding='same',
    #     ))
    # model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))

    # model.add(tf.keras.layers.Conv2DTranspose(
    #         16, (3, 3), strides=2,
    #         # dilation_rate=2,
    #         activation='sigmoid', padding='same',
    #     ))
    # model.add(tf.keras.layers.Conv2DTranspose(
    #         16, (3, 3), strides=2,
    #         # dilation_rate=2,
    #         activation='sigmoid', padding='same',
    #     ))
    model.add(tf.keras.layers.Conv2DTranspose(
            16, (3, 3), strides=2,
            # dilation_rate=2,
            activation='sigmoid', padding='same',
        ))
    model.add(tf.keras.layers.Conv2DTranspose(
            1, (3, 3), strides=2,
            # dilation_rate=2,
            activation='sigmoid', padding='same',
        ))

    case = Case(
            model=model,
            batch_size=16,
            num_batches=3,
            num_epochs=100,
            optimizer=tf.keras.optimizers.Adam(lr=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            gen_params={
                # Shape of object in ground truth, "rect" or "ellipse"
                "ground_truth_shape": "rect",

                # Needed divisibility of the width and height of images. Depends
                # in amount of downsampling
                "divisibility": 4,

                # Size of images, make divisible by previous parameter or
                # otherwise padding will be added.
                # Used in training dataset but also in validation dataset during
                # training, but not during evaluation.
                "train_val_img_size": (100, 100),
            },
            notes="",
        )

    train_case(case)

    case = Case(
            model=model,
            batch_size=16,
            num_batches=10,
            num_epochs=100,
            optimizer=tf.keras.optimizers.Adam(lr=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            gen_params={
                # Shape of object in ground truth, "rect" or "ellipse"
                "ground_truth_shape": "rect",

                # Needed divisibility of the width and height of images. Depends
                # in amount of downsampling
                "divisibility": 4,

                # Size of images, make divisible by previous parameter or
                # otherwise padding will be added.
                # Used in training dataset but also in validation dataset during
                # training, but not during evaluation.
                "train_val_img_size": (100, 100),
            },
            notes="",
        )

    train_case(case)

    case = Case(
            model=model,
            batch_size=16,
            num_batches=3,
            num_epochs=100,
            optimizer=tf.keras.optimizers.Adam(lr=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            gen_params={
                # Shape of object in ground truth, "rect" or "ellipse"
                "ground_truth_shape": "rect",

                # Needed divisibility of the width and height of images. Depends
                # in amount of downsampling
                "divisibility": 4,

                # Size of images, make divisible by previous parameter or
                # otherwise padding will be added.
                # Used in training dataset but also in validation dataset during
                # training, but not during evaluation.
                "train_val_img_size": (68, 68),
            },
            notes="",
        )

    train_case(case)

    case = Case(
            model=model,
            batch_size=16,
            num_batches=10,
            num_epochs=100,
            optimizer=tf.keras.optimizers.Adam(lr=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            gen_params={
                # Shape of object in ground truth, "rect" or "ellipse"
                "ground_truth_shape": "rect",

                # Needed divisibility of the width and height of images. Depends
                # in amount of downsampling
                "divisibility": 4,

                # Size of images, make divisible by previous parameter or
                # otherwise padding will be added.
                # Used in training dataset but also in validation dataset during
                # training, but not during evaluation.
                "train_val_img_size": (68, 68),
            },
            notes="",
        )

    train_case(case)

    # Define model
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(
            64, (3, 3),
            # dilation_rate=2,
            activation='relu', padding='same',
        ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
    model.add(tf.keras.layers.Conv2D(
            64, (3, 3),
            # dilation_rate=2,
            activation='relu', padding='same',
        ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
    model.add(tf.keras.layers.Conv2D(
            64, (3, 3),
            # dilation_rate=2,
            activation='relu', padding='same',
        ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
    model.add(tf.keras.layers.Conv2D(
            64, (3, 3),
            # dilation_rate=2,
            activation='relu', padding='same',
        ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))

    model.add(tf.keras.layers.Conv2DTranspose(
            16, (3, 3), strides=2,
            # dilation_rate=2,
            activation='sigmoid', padding='same',
        ))
    model.add(tf.keras.layers.Conv2DTranspose(
            16, (3, 3), strides=2,
            # dilation_rate=2,
            activation='sigmoid', padding='same',
        ))
    model.add(tf.keras.layers.Conv2DTranspose(
            16, (3, 3), strides=2,
            # dilation_rate=2,
            activation='sigmoid', padding='same',
        ))
    model.add(tf.keras.layers.Conv2DTranspose(
            1, (3, 3), strides=2,
            # dilation_rate=2,
            activation='sigmoid', padding='same',
        ))

    case = Case(
            model=model,
            batch_size=16,
            num_batches=10,
            num_epochs=100,
            optimizer=tf.keras.optimizers.Adam(lr=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            gen_params={
                # Shape of object in ground truth, "rect" or "ellipse"
                "ground_truth_shape": "rect",

                # Needed divisibility of the width and height of images. Depends
                # in amount of downsampling
                "divisibility": 16,

                # Size of images, make divisible by previous parameter or
                # otherwise padding will be added.
                # Used in training dataset but also in validation dataset during
                # training, but not during evaluation.
                "train_val_img_size": (64, 64),
            },
            notes="",
        )

    train_case(case)

    case = Case(
            model=model,
            batch_size=16,
            num_batches=3,
            num_epochs=100,
            optimizer=tf.keras.optimizers.Adam(lr=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            gen_params={
                # Shape of object in ground truth, "rect" or "ellipse"
                "ground_truth_shape": "rect",

                # Needed divisibility of the width and height of images. Depends
                # in amount of downsampling
                "divisibility": 16,

                # Size of images, make divisible by previous parameter or
                # otherwise padding will be added.
                # Used in training dataset but also in validation dataset during
                # training, but not during evaluation.
                "train_val_img_size": (64, 64),
            },
            notes="",
        )

    train_case(case)
