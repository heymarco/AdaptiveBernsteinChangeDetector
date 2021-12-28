import numpy as np

from hcd import HCD
from changeds import SortedMNIST

import matplotlib.pyplot as plt


def plot_results(x, drift_scores, real_changes, change_points, detection_points, xlims = False):
    if len(vlines_real):
        for line in real_changes:
            plt.axvline(line, c="black", lw=3, label="true change")
    if len(change_points):
        for line in change_points:
            plt.axvline(line, c="red", lw=1, label="change point")
    if len(detection_points):
        for line in detection_points:
            plt.axvline(line, c="green", label="detection point")
    plt.plot(x, drift_scores, marker="o")
    if xlims:
        plt.xlim(xlims[0], xlims[1])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Setup
    start_index = 6400
    end_index = start_index + 1100
    warm_start = 100
    stream = SortedMNIST()
    detector = HCD(delta=0.01, bound="bernstein", warm_start=warm_start)

    # Result tracking
    loss = []
    change_points = []
    real_changes = []
    detection_points = []

    # Execution
    stream.sample_idx = start_index
    while stream.has_more_samples():
        if stream.sample_idx == end_index:
            break
        next_sample, _, is_change = stream.next_sample()
        next_sample = next_sample / 255.0
        detector.add_element(next_sample)
        if detector.warm_start_finished():
            real_changes.append(is_change)
            if detector.detected_change():
                print("Detected change at index {}".format(detector.last_change_point))
                change_points.append(detector.last_change_point)
                detection_points.append(detector.last_detection_point)
            else:
                loss.append(detector.window._losses()[-1])

    # Plotting
    vlines_real = [i + warm_start + start_index for i in range(len(real_changes)) if real_changes[i]]
    change_points = np.array(change_points) + start_index + warm_start
    detection_points = np.array(detection_points) + start_index + warm_start
    x = np.arange(len(loss)) + start_index + warm_start
    plot_results(x, loss, vlines_real, change_points, detection_points)
