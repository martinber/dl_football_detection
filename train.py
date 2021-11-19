from pathlib import Path
from datetime import datetime
import json

import tensorflow as tf

import gen

# TODO: Recwive as arguments
BALL_IMG_PATH = Path("./res/ball.jpg")
VAL2017_FOLDER_PATH = Path("/home/mbernardi/extra/async/ipcv/sem_3/deep_learning/labs/5/val2017")
DATA_VERSION = "v4"
CASES_PATH = Path(f"./cases/{DATA_VERSION}/")


class Case:

    def __init__(self, model, batch_size, num_batches, num_epochs,
            optimizer, loss, metrics, gen_params, notes):
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

        # Metrics, list of Keras objects
        self.metrics = metrics

        # History of the training, empty for now
        self.history = []

        # Results of each metric evaluation, empty for now
        self.eval = []

        # Parameters of the data generator. Can only contain serializable things
        self.gen_params = gen_params

        # Model
        self.model = model
        self.model.compile(optimizer=self.optimizer, loss=self.loss,
                metrics=self.metrics)

        # Notes
        self.notes = notes

    def set_history(self, history):
        """
        Save in this object the result of the history of the fit() of the model.

        Takes the History object returned by fit()
        """
        self.history = history.history

    def set_eval(self, evaluation):
        """
        Save in this object the result of the evaluation of the the model.

        Takes the object returned by evaluate()
        """
        self.eval = dict(zip(self.model.metrics_names, evaluation))

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
                "metrics": [str(m.get_config()) for m in self.metrics],
                "history": self.history,
                "eval": self.eval,
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

    history = case.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=case.num_epochs,
            verbose=1,
            callbacks=[cp_callback],
        )

    case.set_history(history)

    eval_dataset = tf.data.Dataset.from_generator(
            lambda: gen.image_generator(
                VAL2017_FOLDER_PATH,
                BALL_IMG_PATH,
                batch_size=16,
                num_batches=10,
                shuffle=False,
                params=case.gen_params,
                evaluation=True,
            ),
            output_signature=(
                tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
            )
        )
    case.set_eval(case.model.evaluate(eval_dataset))

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

    for size in [68, 100, 200]:

        case = Case(
                model=model,
                batch_size=16,
                num_batches=10,
                num_epochs=100,
                optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=[
                    tf.keras.metrics.BinaryCrossentropy(from_logits=False),
                    tf.keras.metrics.BinaryAccuracy(threshold=0.5),
                    tf.keras.metrics.Precision(thresholds=0.5),
                    tf.keras.metrics.Recall(thresholds=0.5),
                    tf.keras.metrics.MeanAbsoluteError(),
                    tf.keras.metrics.MeanSquaredError(),
                ],
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
                    "train_val_img_size": (200, 200),
                },
                notes="",
            )

        train_case(case)
