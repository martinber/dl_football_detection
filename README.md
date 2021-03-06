# dl_football_detection

Detection and localization of football/soccer ball in images using Deep
Learning.

In the `report/` folder there are more explanations.

Project for a Deep Learning Master program course, the problem itself its very
easy to solve. This was a learning exercise to try some things:

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

    - A U-net model is also tested, which is basically the same idea.

- Generate datasets on the go.

    - The COCO Val 2017 images dataset is used but it could be just any amount
      of images in a folder. Not all images are used, depends in the batch size
      and amount of epochs.

    - Images from the folder are loaded and a ball is pasted in a random
      position over them. At the same time, a ground truth image is done which
      is a black image with a white ball.

    - Padding is done to make all images in the batch to have same size.

    - Several simple image processing operations can be done with the ball:
      rotation, scaling, brightness.

## Usage

First, download the
[COCO 2017 Val dataset](http://images.cocodataset.org/zips/val2017.zip) and
write the path in `train.py`.

Read `./main.py -h` for more.

- `./main.py train`: Trains models and saves them automatically. Before running,
  define the models and parameters to use in the function `train()` in
  `train.py`.

- `./main.py list`: List all the trained models. Use `-v` for more info.

- `./main.py eval {model_id} -e`: Evaluate trained model using examples.

- `./main.py eval {model_id} -p`: Show plots of evolution of metrics for each
  learning epoch.

## Colab usage

This can be used in Google Colab, a not very polished example is in
`colab_run.ipynb`

## Attributions

- `ball_2_transparent`: A. Scott Fulkerson, CC BY-SA 4.0, via Wikimedia Commons
- U-net implementation: https://github.com/VidushiBhatia/U-Net-Implementation

## Some links

- https://github.com/tensorflow/models/blob/master/research/object_detection/README.md
- https://github.com/VidushiBhatia/U-Net-Implementation
- https://www.kaggle.com/gpiosenka/balls-image-classification
- COCO 2017 Val images 5K/1GB: https://cocodataset.org/#download
- https://learnopencv.com/cnn-fully-convolutional-image-classification-with-tensorflow/
- https://medium.com/mindboard/image-classification-with-variable-input-resolution-in-keras-cbfbe576126f
- https://stackoverflow.com/questions/55929401/how-to-specify-model-compile-for-binary-crossentropy-activation-sigmoid-and-act
