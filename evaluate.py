from pathlib import Path
import json

import tensorflow as tf
import matplotlib.pyplot as plt

import gen

# TODO: Recwive as arguments
BALL_IMG_PATH = Path("./res/ball.jpg")
VAL2017_FOLDER_PATH = Path("/home/mbernardi/extra/async/ipcv/sem_3/deep_learning/labs/5/val2017")
DATA_VERSION = "v4"
CASES_PATH = Path(f"./cases/{DATA_VERSION}/")

def plot_example(model, gen_params):

    image_gen = gen.image_generator(
            VAL2017_FOLDER_PATH,
            BALL_IMG_PATH,
            batch_size=1,
            shuffle=True,
            params=gen_params,
            evaluation=True,
        )
    x, y = next(image_gen)
    img = x[0]
    ground_truth = y[0]
    y_est = model.predict(x)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    aximg = ax1.imshow(img)
    # fig.colorbar(aximg, ax=ax1)
    ax1.set_title("Input")

    aximg = ax2.imshow(ground_truth.squeeze())
    fig.colorbar(aximg, ax=ax2)
    ax2.set_title("Ground truth")

    aximg = ax3.imshow(y_est[0].squeeze())
    fig.colorbar(aximg, ax=ax3)
    ax3.set_title("Output")

    # aximg = ax4.imshow(y_est[0].squeeze()[1:-1, 1:-1])
    # fig.colorbar(aximg, ax=ax4)
    # ax4.set_title("Output cropped")

    plt.show()

def plot_history(history):
    # 1, 2, 3, .... num_epochs
    epochs = range(1, len(history["loss"]) + 1)

    # Get names of metrics ignoring the ones that are "val_"
    metrics = [name for name in history.keys() if not "val_" in name]
    print(history.keys())
    print(metrics)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    for metric, values in history.items():
        ax1.plot(epochs, values, label=metric)

    ax1.legend()
    plt.show()

def evaluate(case_id):

    case_path = CASES_PATH / case_id
    model = tf.keras.models.load_model(case_path)

    with open(case_path / "case.json", "r") as f:
        case_description = json.load(f)

    evaluation = case_description["eval"]
    for metric, value in evaluation.items():
        print(metric, "=", value)

    plot_history(case_description["history"])
    plot_example(model, case_description["gen_params"])
