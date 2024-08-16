import numpy as np
import lib
import typing as ty
import pdb

def dropout_features0(D: 'Dataset', train_p: float, val_p: float, test_p: float, seed: int) -> 'Dataset':
    def dropout(array: np.ndarray, p: float, rng: ty.Optional[np.random.RandomState] = None) -> np.ndarray:
        """ Randomly drop features with probability p """
        if array is None:
            return None
        if rng is None:
            rng = np.random
        mask = rng.binomial(1, 1 - p, size=array.shape)
        return array * mask

    def categorical_dropout(array: np.ndarray, p: float, rng: ty.Optional[np.random.RandomState] = None) -> np.ndarray:
        """ Randomly drop categorical features with probability p by setting them to NaN """
        if array is None:
            return None
        if rng is None:
            rng = np.random
        mask = rng.binomial(1, 1 - p, size=array.shape)
        array[mask == 0] = 'nan'
        return array

    # Create a random state for train dropout to ensure different masks each time
    train_rng = np.random.RandomState()

    # Dropout for train set with random probability (no seed control)
    if D.N is not None:
        D.N['train'] = dropout(D.N['train'], train_p, train_rng).astype(D.N['train'].dtype)
    if D.C is not None:
        D.C['train'] = categorical_dropout(D.C['train'], train_p, train_rng)

    # Use a specific seed for validation and test sets to ensure the same mask
    val_rng = np.random.RandomState(seed)
    test_rng = np.random.RandomState(seed)

    if D.N is not None:
        val_mask = dropout(np.ones_like(D.N['val']), val_p, val_rng).astype(D.N['val'].dtype)
        test_mask = dropout(np.ones_like(D.N['test']), test_p, test_rng).astype(D.N['test'].dtype)

        D.N['val'] *= val_mask
        D.N['test'] *= test_mask
    if D.C is not None:
        D.C['val'] = categorical_dropout(D.C['val'], val_p, val_rng)
        D.C['test'] = categorical_dropout(D.C['test'], test_p, test_rng)

    return D


def dropout_features1(D: 'Dataset', train_p: float, val_p: float, test_p: float, seed: int) -> 'Dataset':
    def dropout(array: np.ndarray, p: float, rng: ty.Optional[np.random.RandomState] = None) -> np.ndarray:
        """ Randomly drop features with probability p and replace with 1 """
        if array is None:
            return None
        if rng is None:
            rng = np.random
        mask = rng.binomial(1, 1 - p, size=array.shape)
        return np.where(mask, array, 1)  # Replace dropped elements with 1

    def categorical_dropout(array: np.ndarray, p: float, rng: ty.Optional[np.random.RandomState] = None) -> np.ndarray:
        """ Randomly drop categorical features with probability p by setting them to NaN """
        if array is None:
            return None
        if rng is None:
            rng = np.random
        mask = rng.binomial(1, 1 - p, size=array.shape)
        array[mask == 0] = 'nan'
        return array

    # Create a random state for train dropout to ensure different masks each time
    train_rng = np.random.RandomState()

    # Dropout for train set with random probability (no seed control)
    if D.N is not None:
        D.N['train'] = dropout(D.N['train'], train_p, train_rng).astype(D.N['train'].dtype)
    if D.C is not None:
        D.C['train'] = categorical_dropout(D.C['train'], train_p, train_rng).astype(D.C['train'].dtype)

    # Use a specific seed for validation and test sets to ensure the same mask
    val_rng = np.random.RandomState(seed)
    test_rng = np.random.RandomState(seed)

    if D.N is not None:
        D.N['val'] = dropout(D.N['val'], val_p, val_rng).astype(D.N['val'].dtype)
        D.N['test'] = dropout(D.N['test'], test_p, test_rng).astype(D.N['test'].dtype)
    if D.C is not None:
        D.C['val'] = categorical_dropout(D.C['val'], val_p, val_rng).astype(D.C['val'].dtype)
        D.C['test'] = categorical_dropout(D.C['test'], test_p, test_rng).astype(D.C['test'].dtype)

    return D

def dropout_featuresAVG(D: 'Dataset', train_p: float, val_p: float, test_p: float, seed: int) -> 'Dataset':
    def dropout(array: np.ndarray, p: float, rng: ty.Optional[np.random.RandomState] = None) -> np.ndarray:
        """ Randomly drop features with probability p and replace with column mean """
        if array is None:
            return None
        if rng is None:
            rng = np.random
        mask = rng.binomial(1, 1 - p, size=array.shape)
        # Calculate column means where mask is 1 (i.e., not dropped)
        col_means = np.sum(array * mask, axis=0) / np.maximum(np.sum(mask, axis=0), 1)
        # Replace dropped elements with column means
        return np.where(mask, array, col_means)
    
    def categorical_dropout(array: np.ndarray, p: float, rng: ty.Optional[np.random.RandomState] = None) -> np.ndarray:
        """ Randomly drop categorical features with probability p by setting them to NaN """
        if array is None:
            return None
        if rng is None:
            rng = np.random
        mask = rng.binomial(1, 1 - p, size=array.shape)
        array[mask == 0] = 'nan'
        return array

    # Create a random state for train dropout to ensure different masks each time
    train_rng = np.random.RandomState()

    # Dropout for train set with random probability (no seed control)
    if D.N is not None:
        D.N['train'] = dropout(D.N['train'], train_p, train_rng).astype(D.N['train'].dtype)
    if D.C is not None:
        D.C['train'] = categorical_dropout(D.C['train'], train_p, train_rng).astype(D.C['train'].dtype)

    # Use a specific seed for validation and test sets to ensure the same mask
    val_rng = np.random.RandomState(seed)
    test_rng = np.random.RandomState(seed)

    if D.N is not None:
        D.N['val'] = dropout(D.N['val'], val_p, val_rng).astype(D.N['val'].dtype)
        D.N['test'] = dropout(D.N['test'], test_p, test_rng).astype(D.N['test'].dtype)
    if D.C is not None:
        D.C['val'] = categorical_dropout(D.C['val'], val_p, val_rng).astype(D.C['val'].dtype)
        D.C['test'] = categorical_dropout(D.C['test'], test_p, test_rng).astype(D.C['test'].dtype)

    return D