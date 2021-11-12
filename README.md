# dl_football_detection

Detection and localization of football/soccer ball in images using Deep
Learning.

Project for Deep Learning Master course, topic chosen as an excuse to do:

- Save and load models, to be able to evaluate old models and compare.

    - When running `main.py train` a model is trained and saved in a folder,
      indicating also the preprocessing done, the hyperparameters, notes, etc.
      This is called a "case".

    - When running `main.py list` it prints list of "cases", with description of
      hyperparameters, notes, etc.

    - When running `main.py eval` a "case" is loaded and evaluated, plots are
      shown, etc.

- FCN (Fully Convolutional Network).

    - The input is an image, and the outputs is an image with probabilities of
      the ball being in that position.

    - Generally several convolutional layers are used to downsample and then
      transposed convolutional layers are used to upsample.

    - The original idea is to train convolutional filters that can easily detect
      the patters in the ball, to do simple object detection without region
      proposals, etc.

- Automatize hyperparameter search.

- Generate datasets on the go.

## Best so far

```
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
```

## Links

- https://github.com/tensorflow/models/blob/master/research/object_detection/README.md
- https://www.kaggle.com/gpiosenka/balls-image-classification
- COCO 2017 Val images 5K/1GB: https://cocodataset.org/#download
- https://learnopencv.com/cnn-fully-convolutional-image-classification-with-tensorflow/
- https://medium.com/mindboard/image-classification-with-variable-input-resolution-in-keras-cbfbe576126f
