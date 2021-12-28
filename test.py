import numpy as np

from hcd import HCD
from changeds import SortedMNIST

import matplotlib.pyplot as plt


if __name__ == '__main__':
    start_index = 6400
    end_index = start_index + 1100
    warm_start = 100

    loss = []
    change_points = []
    real_changes = []
    detection_points = []
    detector = HCD(delta=0.0001, bound="bernstein", warm_start=warm_start)
    stream = SortedMNIST()
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
                loss.append(detector.window.w[-1][0])
    vlines_real = [i for i in range(len(real_changes)) if real_changes[i]]
    if len(vlines_real):
        for line in vlines_real:
            plt.axvline(line + start_index + warm_start, c="black", lw=3, label="true change")
    if len(change_points):
        for line in change_points:
            plt.axvline(line + start_index + warm_start, c="red", lw=1, label="change point")
    if len(detection_points):
        for line in detection_points:
            plt.axvline(line + start_index + warm_start, c="green", label="detection point")
    plt.plot(np.arange(len(loss)) + start_index + warm_start, loss, marker="o")
    # plt.xlim(6850, 6950)
    plt.legend()
    plt.show()
