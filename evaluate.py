from pathlib import Path
import json

import tensorflow as tf
import matplotlib.pyplot as plt

import gen

def plot_example(model, gen_params):

    image_gen = gen.image_generator(
            batch_size=1,
            num_batches=None,
            shuffle=True,
            params=gen_params,
            evaluation=True,
        )

    fig = plt.figure()
    while True:


        x, y = next(image_gen)
        img = x[0]
        ground_truth = y[0]
        y_est = model.predict(x)

        (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2)

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

        plt.draw()
        plt.pause(0.001)
        if input("Press enter to continue or q and enter to exit...") == "q":
            break

        fig.clear()

def plot_history(history):
    # 1, 2, 3, .... num_epochs
    epochs = range(1, len(history["loss"]) + 1)

    # Get names of metrics ignoring the ones that are "val_"
    metrics = [name for name in history.keys() if not "val_" in name]

    fig, (ax1, ax2) = plt.subplots(2, 1)

    for i, metric in enumerate(metrics):
        val_metric = "val_" + metric
        ax1.plot(epochs, history[metric],
                label=metric,
                linestyle="dashed", color="C"+str(i)
            )
        ax1.plot(epochs, history[val_metric],
                label=metric,
                linestyle="solid", color="C"+str(i)
            )

    ax1.legend()
    plt.show()

def evaluate(case_id, cases_path, history, examples):

    case_path = cases_path / case_id
    model = tf.keras.models.load_model(case_path)

    with open(case_path / "case.json", "r") as f:
        case_description = json.load(f)

    evaluation = case_description["eval"]
    for metric, value in evaluation.items():
        print(metric, "=", value)

    if history:
        plot_history(case_description["history"])
    if examples:
        plot_example(model, case_description["gen_params"])
