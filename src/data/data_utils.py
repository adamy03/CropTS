from calendar import c
import signal
import pandas as pd
import numpy as np
import pickle as pkl
import os
import h5py

from typing import Iterable
from tqdm import tqdm
from sklearn.model_selection import train_test_split

LABEL_COL = "LABEL"
POINT_ID_COL = "POINT_ID"
RANDOM_STATE = 42


def add_lucas_labels(
    signals,
    labels,
):
    """Adds LUCAS labels to SAR signals based on POINT_ID

    Args:
        signals (str): df of SAR signals
        labels (str): LUCAS labels

    Returns:
        _type_: dataframe with LUCAS labels
    """
    print("Loading files")
    print(f"Creating dataset of size {len(signals)}")

    matched_labels = []
    pbar = tqdm(total=len(signals))

    for index, row in signals.iterrows():
        label = str(labels.loc[labels["POINT_ID"] == row["POINT_ID"]]["LC1"].item())
        if "B" not in label:
            label = "NOT_CROP"

        matched_labels.append(label)

        pbar.update(1)

    signals["LABEL"] = matched_labels

    return signals


def drop_labels(crop_data: pd.DataFrame, label_col="LABEL", min_labels=1000):
    """
    Drop labels from the crop_data DataFrame that have fewer than min_labels occurrences.

    Args:
        crop_data (pd.DataFrame): The DataFrame containing the crop data.
        label_col (str, optional): The name of the column containing the labels. Defaults to 'LABEL'.
        min_labels (int, optional): The minimum number of occurrences required for a label to be kept. Defaults to 1000.

    Returns:
        pd.DataFrame: The crop_data DataFrame with dropped labels.
    """

    counts = np.unique(crop_data[label_col], return_counts=True)
    to_drop = [counts[0][i] for i in range(len(counts[0])) if counts[1][i] < min_labels]
    crop_data = crop_data.loc[~crop_data[label_col].isin(to_drop)]

    print("Dropped classes:", to_drop)
    print("Dataset Length: ", len(crop_data))

    return crop_data


def encode_labels(labels):
    """Convert labels to one hot encoded vectors

    Args:
        labels (_type_): list of labels to convert

    Returns:
        _type_: Dict mapping labels to one hot encoded indicies
    """
    # Return dict mapping labels to one hot encoded indicies
    labels = np.unique(labels)
    encoded_vecs = np.identity(len(labels))
    keys = dict(zip(labels, [encoded_vecs[i, :] for i in range(len(labels))]))

    return keys


def convert_label(encoded_vec, keys):
    """Convert one hot encoded vector to associated label

    Args:
        encoded_vec (_type_): Vector to convert
        keys (_type_): Mapping of labels to one hot encoded indicies

    Returns:
        _type_: Label
    """
    label = list(keys.keys())[int(np.where(encoded_vec == 1)[0].item())]
    return label


def load_data(
    data_path: str,
):
    """Load hdf data to memory

    Args:
        data_path (str): hdf file path

    Returns:
        _type_: Dict of data
    """
    with h5py.File(data_path, "r") as f:
        keys = [key for key in f.keys()]
        data = []
        for key in keys:
            data.append(f[key][:])

    return dict(zip(keys, data))


def get_band_arrays(data, bands, sort_func=None):
    """
    Retrieve arrays of band data from a given dataset.

    Args:
        data (pandas.DataFrame): The dataset containing the band data.
        bands (list): A list of band names to retrieve.
        sort_func (function, optional): A function to sort the band columns. Defaults to None.

    Returns:
        dict: A dictionary where the keys are the band names and the values are the corresponding arrays of band data.
    """
    band_data = {}

    for band in bands:
        col_mask = data.columns.str.contains(band).tolist()
        band_cols = data.columns[col_mask].tolist()

        if sort_func:
            band_cols.sort(key=sort_func)

        band_data[band] = data.loc[:, band_cols].to_numpy()

    return band_data


def dataset_split(data, labels, partition=[0.6, 0.1, 0.3]):
    """
    Split the dataset into training, validation, and testing sets.

    Args:
        data (array-like): The input data.
        labels (array-like): The corresponding labels for the input data.
        partition (list, optional): The partition ratios for training, validation, and testing sets.
            Defaults to [0.6, 0.1, 0.3].

    Returns:
        dict: A dictionary containing the split dataset with the following keys:
            - 'train_signals': The training data.
            - 'train_labels': The corresponding labels for the training data.
            - 'val_signals': The validation data.
            - 'val_labels': The corresponding labels for the validation data.
            - 'test_signals': The testing data.
            - 'test_labels': The corresponding labels for the testing data.
    """
    train_signals, test_signals, train_labels, test_labels = train_test_split(
        data, labels, test_size=partition[2], random_state=RANDOM_STATE,
        stratify=np.unique(labels)
    )
    train_signals, val_signals, train_labels, val_labels = train_test_split(
        train_signals, train_labels, test_size=partition[1], random_state=RANDOM_STATE,
        stratify=np.unique(labels)
    )
    
    assert len(train_signals) == len(train_labels)
    assert len(val_signals) == len(val_labels)
    assert len(test_signals) == len(test_labels)
    assert len(np.unique(train_labels)) == len(np.unique(val_labels)) == len(np.unique(test_labels))

    dataset = {
        "train_signals": train_signals,
        "train_labels": train_labels,
        "val_signals": val_signals,
        "val_labels": val_labels,
        "test_signals": test_signals,
        "test_labels": test_labels,
    }

    return dataset
