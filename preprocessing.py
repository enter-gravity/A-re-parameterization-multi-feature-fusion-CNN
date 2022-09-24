import pickle
from pathlib import Path
import numpy as np
import wfdb

PATH = Path("dataset")
sampling_rate = 360

invalid_labels = ['|', '~', '!', '+', '[', ']', '"', 'x']

tol = 0.05


def worker(record):
    if record == 114:
        signal = wfdb.rdrecord((PATH / record).as_posix(), channels=[1]).p_signal[:, 0]
    else:
        signal = wfdb.rdrecord((PATH / record).as_posix(), channels=[0]).p_signal[:, 0]

    annotation = wfdb.rdann((PATH / record).as_posix(), extension="atr")
    r_peaks, labels = annotation.sample, np.array(annotation.symbol)

    indices = [i for i, label in enumerate(labels) if label not in invalid_labels]
    r_peaks, labels = r_peaks[indices], labels[indices]

    newR = []
    for r_peak in r_peaks:
        r_left = np.maximum(r_peak - int(tol * sampling_rate), 0)
        r_right = np.minimum(r_peak + int(tol * sampling_rate), len(signal))
        newR.append(r_left + np.argmax(signal[r_left:r_right]))
    r_peaks = np.array(newR, dtype="int")

    AAMI = {
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,  # N
        "A": 1, "a": 1, "S": 1, "J": 1,  # SVEB
        "V": 2, "E": 2,  # VEB
        "F": 3,  # F
        "/": 4, "f": 4, "Q": 4  # Q
    }
    categories = [AAMI[label] for label in labels]

    return {
        "record": record,
        "signal": signal, "r_peaks": r_peaks, "categories": categories
    }


if __name__ == "__main__":

    train_records = [
        '101', '102', '104', '107', '217', '106', '108', '109', '112', '114',
        '115', '116', '118', '119','122', '124', '201', '203', '205', '207',
        '208', '209', '215', '220','223', '230','100', '103', '105', '111',
        '113', '117', '121', '123', '200', '202','210', '212', '213', '214',
        '219', '221', '222', '228', '231', '232','233', '234'
    ]
    print("train processing...")

    train_data = []
    for records in train_records:
        result = worker(records)
        train_data.append(result)

    with open((PATH / "db.pkl").as_posix(), "wb") as f:
        pickle.dump(train_data, f, protocol=4)

    print("ok!")
