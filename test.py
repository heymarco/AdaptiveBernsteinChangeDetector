import numpy as np
from sklearn.preprocessing import MinMaxScaler

from hcd import HCD
from changeds import SortedMNIST, SortedCIFAR10, SortedCIFAR100, SortedFashionMNIST
from components.experiment_logging import logger

import matplotlib.pyplot as plt


def plot_results(x, drift_scores, real_changes, change_points, detection_points, xlims = False):
    if len(real_changes):
        for line in real_changes:
            plt.axvline(line, c="black", lw=3, label="true change")
    if len(change_points):
        for line in change_points:
            plt.axvline(line, c="red", lw=1, label="change point")
    if len(detection_points):
        for line in detection_points:
            plt.axvline(line, c="green", label="detection point")
    plt.scatter(x, drift_scores, alpha=0.2)
    if xlims:
        plt.xlim(xlims[0], xlims[1])
    plt.legend()
    plt.show()


def preprocess(x: np.ndarray):
    return MinMaxScaler().fit_transform(x)


if __name__ == '__main__':
    # Setup
    start_index = 0
    end_index = np.infty
    warm_start = 200
    update_epochs = 10
    stream = SortedCIFAR10(preprocess=preprocess)
    detector = HCD(delta=0.01, bound="bernstein", update_epochs=update_epochs, new_ae=True)

    # Result tracking
    loss = []
    change_points = []
    real_changes = []
    detection_points = []

    # Warm start
    if warm_start > 0:
        data = np.array([
            stream.next_sample()[0]
            for _ in range(warm_start)
        ])
        detector.pre_train(data)

    # Execution
    stream.sample_idx = start_index
    while stream.has_more_samples():
        if stream.sample_idx >= end_index:
            break
        logger.track_time()
        logger.track_index(stream.sample_idx)
        next_sample, _, is_change = stream.next_sample()
        logger.track_is_change(is_change)
        detector.add_element(next_sample)
        real_changes.append(is_change)
        loss.append(detector.last_loss())
        if detector.detected_change():
            print("Detected change at index {}".format(detector.last_change_point))
            change_points.append(detector.last_change_point)
            detection_points.append(detector.last_detection_point)
        logger.finalize_round()
    logger.save()

    # Plotting
    vlines_real = [i + warm_start + start_index for i in range(len(real_changes)) if real_changes[i]]
    change_points = np.array(change_points) + start_index + warm_start
    detection_points = np.array(detection_points) + start_index + warm_start
    x = np.arange(len(loss)) + start_index + warm_start
    plot_results(x, loss, vlines_real, change_points, detection_points)
