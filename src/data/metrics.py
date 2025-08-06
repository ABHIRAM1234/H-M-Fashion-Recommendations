"""
Evaluation metrics for the H&M Fashion Recommendation System.

This module implements standard recommendation system evaluation metrics
including MAP@k, Recall@k, and Hit Rate@k. These metrics are used to
assess the quality of recommendation predictions and model performance.
"""

from typing import Iterable
import numpy as np


def _ap_at_k(actual, predicted, k=10):
    """Calculate Average Precision at k for a single user.
    
    This is a helper function that computes the average precision for one
    user's recommendations. It's used by the main map_at_k function.
    
    Parameters
    ----------
    actual : list
        List of actual relevant items
    predicted : list
        List of predicted items in order
    k : int
        Number of top predictions to consider
        
    Returns
    -------
    float
        Average precision score for this user
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    # Calculate precision at each position where we hit a relevant item
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:  # Avoid duplicates
            num_hits += 1.0
            score += num_hits / (i + 1.0)  # Precision at position i+1

    if actual is None:
        return 0.0

    return score / min(len(actual), k)


def _rk(actual, predicted, k=10):
    """Calculate Recall at k for a single user.
    
    This is a helper function that computes the recall for one user's
    recommendations. It's used by the main recall_at_k function.
    
    Parameters
    ----------
    actual : list
        List of actual relevant items
    predicted : list
        List of predicted items in order
    k : int
        Number of top predictions to consider
        
    Returns
    -------
    float
        Recall score for this user
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    # Count how many relevant items are in the top-k predictions
    score = sum([1 for r in actual if r in predicted]) / len(actual)

    return score


def map_at_k(actual: Iterable, predicted: Iterable, k: int = 12) -> float:
    """Compute Mean Average Precision at k.
    
    This is the primary evaluation metric for the H&M competition. MAP@k
    measures the quality of ranked recommendations by considering both
    the relevance of items and their position in the recommendation list.
    
    A higher MAP@k indicates better recommendation quality, with perfect
    recommendations having MAP@k = 1.0.

    Parameters
    ----------
    actual : Iterable
        Iterable of actual relevant items for each user
    predicted : Iterable
        Iterable of predicted item lists for each user
    k : int, optional
        Number of top predictions to consider, by default 12

    Returns
    -------
    float
        Mean Average Precision at k across all users
    """
    return np.mean(
        [_ap_at_k(a, p, k) for a, p in zip(actual, predicted) if a is not None]
    )


def recall_at_k(actual: Iterable, predicted: Iterable, k: int = 12) -> float:
    """Compute Recall at k.
    
    Recall@k measures the proportion of relevant items that are successfully
    recommended in the top-k predictions. It focuses on coverage of relevant
    items rather than ranking quality.

    Parameters
    ----------
    actual : Iterable
        Iterable of actual relevant items for each user
    predicted : Iterable
        Iterable of predicted item lists for each user
    k : int, optional
        Number of top predictions to consider, by default 12

    Returns
    -------
    float
        Recall at k across all users
    """
    return np.mean([_rk(a, p, k) for a, p in zip(actual, predicted)])


def hr_at_k(actual: Iterable, predicted: Iterable, k: int = 10) -> float:
    """Compute Hit Rate at k.
    
    Hit Rate@k measures the proportion of users who have at least one
    relevant item in their top-k recommendations. It's a binary metric
    that indicates whether the system can successfully recommend at least
    one relevant item to each user.

    Parameters
    ----------
    actual : Iterable
        Iterable of actual relevant items for each user
    predicted : Iterable
        Iterable of predicted item lists for each user
    k : int, optional
        Number of top predictions to consider, by default 10

    Returns
    -------
    float
        Hit Rate at k across all users
    """
    count = 0
    for i, actual_i in enumerate(actual):
        # Check if any relevant item is in the top-k predictions
        for p in predicted[i][:k]:
            if p in actual_i:
                count += 1  # User has at least one hit
                break  # No need to check further for this user
    return count / len(actual)
