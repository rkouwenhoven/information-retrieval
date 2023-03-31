import logging
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn import preprocessing

from fast_forward.ranking import Ranking

LOGGER = logging.getLogger(__name__)


def interpolate_cc(alpha: float, score1: float, score2: float) -> float:
    """Interpolate scores using convex combination: r1 * alpha + r2 * (1 - alpha).

    Args:
        alpha (float):
        score1 (float):
        score2 (float):

    Returns:
        float: Interpolated score
    """
    return alpha * score1 + (1 - alpha) * score2


def interpolate_rrf(alpha: float, score1: float, score2: float) -> float:
    """Interpolate scores using reciprocal rank fusion: 1 / (alpha + r1) + 1 / (alpha + r2).

        Args:
            alpha (float):
            score1 (float):
            score2 (float):

        Returns:
            float: Interpolated score
        """
    return 1 / (alpha + score1) + 1 / (alpha + score2)


def interpolate(
        r1: Ranking, r2: Ranking, alpha: float, name: str = None, sort: bool = True, useCc: bool = True
) -> Ranking:
    """Interpolate scores. For each query-doc pair:
        * If the pair has only one score, ignore it.
        * If the pair has two scores, interpolate.

    Args:
        r1 (Ranking): Scores from the first retriever.
        r2 (Ranking): Scores from the second retriever.
        alpha (float): Interpolation weight.
        name (str, optional): Ranking name. Defaults to None.
        sort (bool, optional): Whether to sort the documents by score. Defaults to True.
        useCc (bool, optional): Whether to use convex combination or RRF for interpolation. Defaults to True.

    Returns:
        Ranking: Interpolated ranking.
    """
    assert r1.q_ids == r2.q_ids
    results = defaultdict(dict)
    for q_id in r1:
        for doc_id in r1[q_id].keys() & r2[q_id].keys():
            score1 = r1[q_id][doc_id]
            score2 = r2[q_id][doc_id]

            if useCc:
                result = interpolate_cc(alpha, score1, score2)
            else:
                result = interpolate_rrf(alpha, score1, score2)
            results[q_id][doc_id] = result
    return Ranking(results, name=name, sort=sort, copy=False)


def normalizeMinMax(results: defaultdict):
    df = pd.DataFrame({'document_id': results.keys(), 'score': results.values()})
    min_max_scaler = preprocessing.MinMaxScaler()
    x = df['score'].values.reshape(-1, 1)  # returns a numpy array
    x = min_max_scaler.fit_transform(x)
    x = np.rint(x * 3)
    df['score'] = x
    return dict(zip(df['document_id'], df['score']))


def normalizeMax(results: defaultdict):
    df = pd.DataFrame({'document_id': results.keys(), 'score': results.values()})
    #If also deleting an offset
    # df['score'] = df['score'] - offset
    df['score'] = df['score'] / df['score'].max()

    # This normalize is wack
    # df['score'] = preprocessing.normalize(df['score'].values.reshape(-1, 1))
    # print(df['score'])

    # df['score'] = np.rint(df['score'] * 3)
    return dict(zip(df['document_id'], df['score']))

def normalizeMeanStd(results: defaultdict):
    df = pd.DataFrame({'document_id': results.keys(), 'score': results.values()})
    # If also deleting an offset
    # df['score'] = df['score'] - offset
    df['score'] = df['score'] - df['score'].mean()
    df['score'] = df['score'] / df['score'].max()
    df['score'] = df['score'] / df['score'].std()

    return dict(zip(df['document_id'], df['score']))
