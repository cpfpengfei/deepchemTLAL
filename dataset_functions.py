## the following functions are ported over from deepchem 2.4.0 to convert datasets back and forth easily
# all original code and rights: https://github.com/deepchem/deepchem/blob/master/deepchem/data/datasets.py#L516-L549
# changes: from class method --> function manipulating dataset directly

import pandas as pd 
import numpy as np
from deepchem.data.datasets import NumpyDataset

def to_dataframe(dataset):
    """Construct a pandas DataFrame containing the data from this Dataset.
    Returns
    -------
    pandas dataframe. If there is only a single feature per datapoint,
    will have column "X" else will have columns "X1,X2,..." for
    features.  If there is only a single label per datapoint, will
    have column "y" else will have columns "y1,y2,..." for labels. If
    there is only a single weight per datapoint will have column "w"
    else will have columns "w1,w2,...". Will have column "ids" for
    identifiers.
    """
    X = dataset.X
    y = dataset.y
    w = dataset.w
    ids = dataset.ids
    if len(X.shape) == 1 or X.shape[1] == 1:
        columns = ['X']
    else:
        columns = [f'X{i+1}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=columns)
    if len(y.shape) == 1 or y.shape[1] == 1:
        columns = ['y']
    else:
        columns = [f'y{i+1}' for i in range(y.shape[1])]
    y_df = pd.DataFrame(y, columns=columns)
    if len(w.shape) == 1 or w.shape[1] == 1:
        columns = ['w']
    else:
        columns = [f'w{i+1}' for i in range(w.shape[1])]
    w_df = pd.DataFrame(w, columns=columns)
    ids_df = pd.DataFrame(ids, columns=['ids'])
    return pd.concat([X_df, y_df, w_df, ids_df], axis=1, sort=False)


def from_dataframe(df, X=None, y=None, w=None, ids=None):
    """Construct a Dataset from the contents of a pandas DataFrame.
    Parameters
    ----------
    df: DataFrame
      the pandas DataFrame
    X: string or list of strings
      the name of the column or columns containing the X array.  If
      this is None, it will look for default column names that match
      those produced by to_dataframe().
    y: string or list of strings
      the name of the column or columns containing the y array.  If
      this is None, it will look for default column names that match
      those produced by to_dataframe().
    w: string or list of strings
      the name of the column or columns containing the w array.  If
      this is None, it will look for default column names that match
      those produced by to_dataframe().
    ids: string
      the name of the column containing the ids.  If this is None, it
      will look for default column names that match those produced by
      to_dataframe().
    """
    # Find the X values.
    if X is not None:
        X_val = df[X]
    elif 'X' in df.columns:
        X_val = df['X']
    else:
        columns = []
        i = 1
        while f'X{i}' in df.columns:
            columns.append(f'X{i}')
            i += 1
        X_val = df[columns]
    if len(X_val.shape) == 1:
        X_val = np.expand_dims(X_val, 1)

    # Find the y values.
    if y is not None:
        y_val = df[y]
    elif 'y' in df.columns:
        y_val = df['y']
    else:
        columns = []
        i = 1
        while f'y{i}' in df.columns:
            columns.append(f'y{i}')
            i += 1
        y_val = df[columns]
    if len(y_val.shape) == 1:
        y_val = np.expand_dims(y_val, 1)

    # Find the w values.
    if w is not None:
        w_val = df[w]
    elif 'w' in df.columns:
        w_val = df['w']
    else:
        columns = []
        i = 1
        while f'w{i}' in df.columns:
            columns.append(f'w{i}')
            i += 1
        w_val = df[columns]
    if len(w_val.shape) == 1:
        w_val = np.expand_dims(w_val, 1)

    # Find the ids.
    if ids is not None:
        ids_val = df[ids]
    elif 'ids' in df.columns:
        ids_val = df['ids']
    else:
        ids_val = None
    
    return NumpyDataset(X_val, y_val, w_val, ids_val)

