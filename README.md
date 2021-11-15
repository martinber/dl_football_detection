# dl_football_detection

Detection and localization of football/soccer ball in images using Deep
Learning.

Project for a Deep Learning Master program course, the problem itself it easy
because the idea was to have these features:

- Save and load models, to be able to evaluate old models and compare.

    - When running `main.py train` a model is trained and saved in a folder,
      also a file is saved listing the preprocessing done, the hyperparameters,
      notes, etc. This is called a "case".

    - When running `main.py list` it prints list of "cases", with description of
      hyperparameters, notes, etc.

    - When running `main.py eval` the model of a "case" is loaded and evaluated,
      plots are shown, etc.

- FCN (Fully Convolutional Network).

    - The input is an image, and the outputs is an image with probabilities of
      the ball being in that position.

    - Generally, several convolutional layers are used to downsample and then
      transposed convolutional layers are used to upsample.

    - The original idea is to train convolutional filters that can easily detect
      the patters in the ball, to do simple object detection without region
      proposals, etc.

- Automatize hyperparameter search.

- Generate datasets on the go.

    - The COCO Val 2017 images dataset is used but it could be just any amount
      of images in a folder. Not all images are used, depends in the batch size
      and amount of epochs.

    - Images from the folder are loaded and a ball is pasted in a random
      position over them. At the same time, a ground truth image is done which
      is a black image with a white ball.

    - Padding is done to make all images in the batch to have same size.

    - Several simple image processing operations can be done.

## Links

- https://github.com/tensorflow/models/blob/master/research/object_detection/README.md
- https://www.kaggle.com/gpiosenka/balls-image-classification
- COCO 2017 Val images 5K/1GB: https://cocodataset.org/#download
- https://learnopencv.com/cnn-fully-convolutional-image-classification-with-tensorflow/
- https://medium.com/mindboard/image-classification-with-variable-input-resolution-in-keras-cbfbe576126f
