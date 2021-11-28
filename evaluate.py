from pathlib import Path
import json

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology

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
        ground_truth = y[0].squeeze()
        y_est = model.predict(x)[0].squeeze()

        (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2)

        aximg = ax1.imshow(img)
        # fig.colorbar(aximg, ax=ax1)
        ax1.set_title("Input")

        aximg = ax2.imshow(ground_truth)
        fig.colorbar(aximg, ax=ax2)
        ax2.set_title("Ground truth")

        aximg = ax3.imshow(y_est)
        fig.colorbar(aximg, ax=ax3)
        ax3.set_title("Output")

        aximg = ax4.imshow(threshold_img(y_est))
        ax4.set_title("Output thresholded")

        true_pos = get_ball_pos(ground_truth)
        pos = get_ball_pos(y_est)
        ax2.scatter(true_pos[0], true_pos[1], marker="x", color="green", s=30)
        ax3.scatter(true_pos[0], true_pos[1], marker="x", color="green", s=30)
        ax4.scatter(true_pos[0], true_pos[1], marker="x", color="green", s=30)
        ax2.scatter(pos[0], pos[1], marker="x", color="red", s=30)
        ax3.scatter(pos[0], pos[1], marker="x", color="red", s=30)
        ax4.scatter(pos[0], pos[1], marker="x", color="red", s=30)

        plt.draw()
        plt.pause(0.001)
        if input("Press enter to continue or q and enter to exit...") == "q":
            break

        fig.clear()

def plot_history(history, history_ignore):
    # 1, 2, 3, .... num_epochs
    epochs = range(1, len(history["loss"]) + 1)

    # Get names of metrics ignoring the ones that are "val_"
    metrics = [name for name in history.keys() if not "val_" in name]

    fig, (ax1, ax2) = plt.subplots(2, 1)

    for i, metric in enumerate(metrics):
        if metric in history_ignore:
            continue

        val_metric = "val_" + metric
        ax1.plot(epochs, history[metric],
                label=metric,
                linestyle="dashed", color="C"+str(i)
            )
        ax1.plot(epochs, history[val_metric],
                label=val_metric,
                linestyle="solid", color="C"+str(i)
            )

    ax1.legend()
    plt.show()

def threshold_img(y):
    """
    Thresholds output of neural network and applies morphological operations
    """
    mask = y > 0.5

    # Apply morphological erosion
    kernel = skimage.morphology.disk(radius=2)
    mask = skimage.morphology.erosion(mask, kernel)
    return mask

def get_ball_pos(heatmap):
    """
    Returns ball position from output of neural network.
    """
    mask = threshold_img(heatmap)

    if not mask.any():
        # If mask is empty
        return 0, 0

    # Leave only the biggest blob in the mask
    labeled_mask = skimage.measure.label(mask)
    blobs_sizes = np.bincount(labeled_mask.ravel())[1:] # From 1 to ignore background
    biggest_blob_label = blobs_sizes.argmax() # Add 1 because we removed background
    mask = (labeled_mask == biggest_blob_label + 1)

    # Get center of mass
    total = mask.sum()
    h, w = heatmap.shape
    x_coords = np.arange(0, w)
    y_coords = np.arange(0, h)

    x = (x_coords * mask.sum(axis=0)).sum() / total
    y = (y_coords * mask.sum(axis=1)).sum() / total

    return x, y


def evaluate(case_id, cases_path, history, history_ignore, examples):

    case_path = cases_path / case_id
    model = tf.keras.models.load_model(case_path)

    with open(case_path / "case.json", "r") as f:
        case_description = json.load(f)

    evaluation = case_description["eval"]
    for metric, value in evaluation.items():
        print(metric, "=", value)

    if history:
        plot_history(case_description["history"], history_ignore)
    if examples:
        plot_example(model, case_description["gen_params"])
