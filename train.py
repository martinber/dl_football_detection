from pathlib import Path
from datetime import datetime
import json

import tensorflow as tf

import gen

VAL2017_FOLDER_PATH = Path("/home/mbernardi/extra/async/ipcv/sem_3/deep_learning/labs/5/val2017")

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

def train_case(case, cases_path):

    case_path = cases_path / case.id
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

def u_net(input_size=(128, 128, 3), n_filters=32, n_classes=3):
    """
    Combine both encoder and decoder blocks according to the U-Net research paper

    Return the model as output

    Code taken from https://github.com/VidushiBhatia/U-Net-Implementation
    """

    def encoder_block(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
        """
        This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning.
        Dropout can be added for regularization to prevent overfitting.
        The block returns the activation values for next layer along with a skip connection which will be used in the decoder
        """
        # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow
        # Proper initialization prevents from the problem of exploding and vanishing gradients
        # 'Same' padding will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size)
        conv = tf.keras.layers.Conv2D(n_filters,
                      3,   # Kernel size
                      activation='relu',
                      padding='same',
                      kernel_initializer='HeNormal')(inputs)
        conv = tf.keras.layers.Conv2D(n_filters,
                      3,   # Kernel size
                      activation='relu',
                      padding='same',
                      kernel_initializer='HeNormal')(conv)

        # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
        conv = tf.keras.layers.BatchNormalization()(conv, training=False)

        # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
        if dropout_prob > 0:
            conv = tf.keras.layers.Dropout(dropout_prob)(conv)

        # Pooling reduces the size of the image while keeping the number of channels same
        # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
        # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
        if max_pooling:
            next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)
        else:
            next_layer = conv

        # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions
        skip_connection = conv

        return next_layer, skip_connection

    def decoder_block(prev_layer_input, skip_layer_input, n_filters=32):
        """
        Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
        merges the result with skip layer results from encoder block
        Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
        The function returns the decoded layer output
        """
        # Start with a transpose convolution layer to first increase the size of the image
        up = tf.keras.layers.Conv2DTranspose(
                     n_filters,
                     (3,3),    # Kernel size
                     strides=(2,2),
                     padding='same')(prev_layer_input)

        # Merge the skip connection from previous block to prevent information loss
        merge = tf.keras.layers.concatenate([up, skip_layer_input], axis=3)

        # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
        # The parameters for the function are similar to encoder
        conv = tf.keras.layers.Conv2D(n_filters,
                     3,     # Kernel size
                     activation='relu',
                     padding='same',
                     kernel_initializer='HeNormal')(merge)
        conv = tf.keras.layers.Conv2D(n_filters,
                     3,   # Kernel size
                     activation='relu',
                     padding='same',
                     kernel_initializer='HeNormal')(conv)
        return conv

    # Input size represent the size of 1 image (the size used for pre-processing)
    inputs = tf.keras.layers.Input(input_size)

    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image
    cblock1 = encoder_block(inputs, n_filters,dropout_prob=0, max_pooling=True)
    cblock2 = encoder_block(cblock1[0],n_filters*2,dropout_prob=0, max_pooling=True)
    cblock3 = encoder_block(cblock2[0], n_filters*4,dropout_prob=0, max_pooling=True)
    cblock4 = encoder_block(cblock3[0], n_filters*8,dropout_prob=0.3, max_pooling=True)
    cblock5 = encoder_block(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False)

    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
    ublock6 = decoder_block(cblock5[0], cblock4[1],  n_filters * 8)
    ublock7 = decoder_block(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = decoder_block(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = decoder_block(ublock8, cblock1[1],  n_filters)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size.
    # Observe the number of channels will be equal to number of output classes
    conv9 = tf.keras.layers.Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(ublock9)

    conv10 = tf.keras.layers.Conv2D(n_classes, 1, activation='sigmoid', padding='same')(conv9)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model

def train(cases_path):

    import tensorflow as tf

    """
    # Define model
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(
            8, (3, 3),
            # dilation_rate=2,
            activation='relu', padding='same',
        ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
    model.add(tf.keras.layers.Conv2D(
            8, (3, 3),
            # dilation_rate=2,
            activation='relu', padding='same',
        ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
    model.add(tf.keras.layers.Conv2D(
            8, (3, 3),
            # dilation_rate=2,
            activation='relu', padding='same',
        ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
    model.add(tf.keras.layers.Conv2D(
            8, (3, 3),
            # dilation_rate=2,
            activation='relu', padding='same',
        ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))

    model.add(tf.keras.layers.Conv2DTranspose(
            8, (3, 3), strides=2,
            # dilation_rate=2,
            activation='sigmoid', padding='same',
        ))
    model.add(tf.keras.layers.Conv2DTranspose(
            8, (3, 3), strides=2,
            # dilation_rate=2,
            activation='sigmoid', padding='same',
        ))
    model.add(tf.keras.layers.Conv2DTranspose(
            8, (3, 3), strides=2,
            # dilation_rate=2,
            activation='sigmoid', padding='same',
        ))
    model.add(tf.keras.layers.Conv2DTranspose(
            1, (3, 3), strides=2,
            # dilation_rate=2,
            activation='sigmoid', padding='same',
        ))
    """
    model = u_net(input_size=(None,None,3), n_filters=32, n_classes=1)

    # for size in [68, 100, 200]:
    for size in [128, 256]:

        case = Case(
                model=model,
                batch_size=16,
                # num_batches=3,
                num_batches=10,
                num_epochs=100,
                # num_epochs=1000,
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
                    # Dataset: Background and ball images
                    "folder_path": str(VAL2017_FOLDER_PATH),
                    "obj_path": str(Path("./res/ball_transparent.png")),

                    # Shape of object in ground truth, "rect" or "ellipse"
                    "ground_truth_shape": "ellipse",

                    # Needed divisibility of the width and height of images. Depends
                    # in amount of downsampling
                    "divisibility": 32,

                    # Size of images, make divisible by previous parameter or
                    # otherwise padding will be added.
                    # Used in training dataset but also in validation dataset during
                    # training, but not during evaluation.
                    "train_val_img_size": (size, size),
                },
                notes="",
            )

        train_case(case, cases_path)
