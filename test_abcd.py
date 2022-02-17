import numpy as np
from sklearn.preprocessing import MinMaxScaler

from approach import ABCD
from changeds import SortedMNIST, SortedCIFAR10, SortedCIFAR100, SortedFashionMNIST, HIPE, LED, RBF, HAR, RandomOrderMNIST
from components.experiment_logging import logger


def preprocess(x: np.ndarray):
    return MinMaxScaler().fit_transform(x)


if __name__ == '__main__':
    # Setup
    start_index = 0
    end_index = np.infty
    warm_start = 100
    update_epochs = 100
    delta = 0.05
    stream = LED(preprocess=preprocess, n_per_concept=3000)
    detector = ABCD(delta=delta, bound="bernstein",
                    update_epochs=update_epochs, new_ae=True)

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
        loss.append(detector.metric())
        if detector.detected_change():
            print("Detected change at index {}".format(detector.last_change_point))
            change_points.append(detector.last_change_point)
            detection_points.append(detector.last_detection_point)
        logger.finalize_round()
    logger.save()
