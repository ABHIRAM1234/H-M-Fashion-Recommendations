"""
Utility functions for the H&M Fashion Recommendation System.

This module contains helper functions for date calculations, memory optimization,
data merging, and embedding similarity computations used throughout the
recommendation pipeline.
"""

from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
from tqdm._tqdm_notebook import tqdm_notebook as tqdm


def calc_valid_date(week_num: int, last_date: str = "2020-09-29") -> Tuple[str]:
    """Calculate start and end date of a given week number.
    
    This function is used to determine the time windows for training and validation
    data splits. It calculates the date range for a specific week relative to
    the last available date in the dataset.

    Parameters
    ----------
    week_num : int
        Week number (0 = most recent week, 1 = previous week, etc.)
    last_date : str, optional
        The last day in the dataset, by default "2020-09-29"

    Returns
    -------
    Tuple[str]
        (start_date, end_date) in "YYYY-MM-DD" format
        
    Examples
    --------
    >>> calc_valid_date(0, "2020-09-29")
    ('2020-09-22', '2020-09-29')
    >>> calc_valid_date(1, "2020-09-29") 
    ('2020-09-15', '2020-09-22')
    """
    # Calculate the end date for the specified week
    end_date = pd.to_datetime(last_date) - pd.Timedelta(days=7 * week_num - 1)
    # Calculate the start date (7 days before end date)
    start_date = end_date - pd.Timedelta(days=7)

    # Convert to string format for consistency
    end_date = end_date.strftime("%Y-%m-%d")
    start_date = start_date.strftime("%Y-%m-%d")
    return start_date, end_date


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Reduce memory usage by optimizing data types.
    
    This function analyzes each column in the dataframe and converts data types
    to the most memory-efficient format possible. It's crucial for handling
    large datasets within memory constraints (50GB limit in this project).
    
    The function:
    1. Identifies integer columns that can be downcast to smaller types
    2. Converts float columns to float32 when possible
    3. Handles missing values appropriately for integer conversions
    4. Reports memory savings

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to optimize memory usage for
    verbose : bool, optional
        Whether to print detailed optimization process, by default False

    Returns
    -------
    pd.DataFrame
        Memory-optimized dataframe with same data but smaller memory footprint
        
    References
    ----------
    .. [1] https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
    """
    # Calculate initial memory usage in MB
    start_mem_usg = df.memory_usage().sum() / 1024**2
    if verbose:
        print("Memory usage of dataframe is :", start_mem_usg, " MB")
    
    # Track columns that had missing values filled in
    NAlist = []
    
    # Process each column for memory optimization
    for col in df.columns:
        # Skip object (string) columns as they require different handling
        if df[col].dtype != object:

            # Print current column type
            if verbose:
                print("******************************")
                print("Column: ", col)
                print("dtype before: ", df[col].dtype)

            # Initialize variables for integer type detection
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()

            # Handle missing values - integers don't support NaN
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                # Fill NaN with a value below the minimum (safe for integer conversion)
                df[col].fillna(mn - 1, inplace=True)

            # Test if column can be converted to integer type
            if pd.api.types.is_integer_dtype(df[col]):
                IsInt = True
            else:
                # Try converting to integer and check if there's any loss of precision
                asint = df[col].fillna(0).astype(np.int64)
                result = df[col] - asint
                result = result.sum()
                # If the difference is negligible, treat as integer
                if result > -0.01 and result < 0.01:
                    IsInt = True

            # Optimize integer datatypes based on value ranges
            if IsInt:
                if mn >= 0:
                    # Use unsigned integers for non-negative values
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)  # 0-255
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)  # 0-65535
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)  # 0-4294967295
                    else:
                        df[col] = df[col].astype(np.uint64)  # Larger values
                else:
                    # Use signed integers for negative values
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)  # -128 to 127
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)  # -32768 to 32767
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)  # -2147483648 to 2147483647
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  # Larger range

            # Convert float columns to 32-bit for memory efficiency
            else:
                df[col] = df[col].astype(np.float32)  # 32-bit float instead of 64-bit

            # Print new column type
            if verbose:
                print("dtype after: ", df[col].dtype)
                print("******************************")

    # Calculate and report final memory usage
    mem_usg = df.memory_usage().sum() / 1024**2
    if verbose:
        print("___MEMORY USAGE AFTER COMPLETION:___")
        print("Memory usage is: ", mem_usg, " MB")
        print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    
    return df, NAlist


def merge_week_data(
    data: Dict, trans: pd.DataFrame, week_num: int, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Merge transaction, user and item features with week data.
    
    This function prepares the training data by merging:
    1. Retrieval candidates (user-item pairs)
    2. Transaction features for the target week
    3. User demographic and preference features
    4. Item categorical and descriptive features
    5. Labels (for training data only)
    
    The function handles both training (with labels) and test (without labels)
    scenarios automatically based on the week number.

    Parameters
    ----------
    data : Dict
        Dictionary containing 'user', 'item', and 'inter' dataframes
    trans : pd.DataFrame
        Transaction dataframe with week information
    week_num : int
        Target week number (0 for test, >0 for training)
    candidates : pd.DataFrame
        Retrieval candidates dataframe with customer_id and article_id

    Returns
    -------
    pd.DataFrame
        Merged dataframe ready for model training with all features and labels
    """
    tqdm.pandas()

    user = data["user"]
    item = data["item"]

    # Extract transaction features for the target week
    # Use week_num + 1 because we want features from the week before the target week
    trans_info = (
        trans[trans["week"] == week_num + 1]
        .sort_values(by=["t_dat", "customer_id"])
        .reset_index(drop=True)
        .groupby(["week", "article_id"], as_index=False)
        .last()  # Take the latest transaction for each article in the week
        .drop(columns=["customer_id"])  # Remove customer_id as it's in candidates
    )
    trans_info["week"] = week_num  # Align week number with target week
    trans_info, _ = reduce_mem_usage(trans_info)  # Optimize memory usage

    # * ======================================================================================================================
    # LABEL GENERATION (only for training data, not test data)
    # * ======================================================================================================================

    if week_num != 0:  # This is training data (not test data)
        # Calculate the date range for the target week
        start_date, end_date = calc_valid_date(week_num)
        
        # Find actual purchases in the target week (positive labels)
        mask = (start_date <= data["inter"]["t_dat"]) & (
            data["inter"]["t_dat"] < end_date
        )
        label = data["inter"].loc[mask, ["customer_id", "article_id"]]
        label = label.drop_duplicates(["customer_id", "article_id"])  # Remove duplicates
        label["label"] = 1  # Mark as positive

        # Filter candidates to only include users who made purchases in the target week
        label_customers = label["customer_id"].unique()
        print(f"Number of customers with purchases in week {week_num}: {len(label_customers)}")
        candidates = candidates[candidates["customer_id"].isin(label_customers)]

        # Merge labels with candidates (left join to keep all candidates)
        candidates = candidates.merge(
            label, on=["customer_id", "article_id"], how="left"
        )
        candidates["label"] = candidates["label"].fillna(0)  # Fill missing labels as negative

    # * ======================================================================================================================
    # FEATURE MERGING
    # * ======================================================================================================================

    # Merge transaction features with candidates
    candidates = candidates.merge(trans_info, on="article_id", how="left")

    # Merge user demographic and preference features
    user_feats = [
        "FN",  # Fashion News subscription
        "Active",  # Active status
        "club_member_status",  # Club membership status
        "fashion_news_frequency",  # Fashion news frequency
        "age",  # Customer age
        "user_gender",  # Inferred user gender
    ]
    candidates = candidates.merge(
        user[["customer_id", *user_feats]], on="customer_id", how="left"
    )
    candidates[user_feats] = candidates[user_feats].astype("int8")  # Optimize memory

    # Merge item categorical and descriptive features
    item_feats = [
        "product_type_no",  # Product type number
        "product_group_name",  # Product group name
        "graphical_appearance_no",  # Graphical appearance number
        "colour_group_code",  # Color group code
        "perceived_colour_value_id",  # Perceived color value ID
        "perceived_colour_master_id",  # Perceived color master ID
        "article_gender",  # Article gender (0=unisex, 1=male, 2=female)
        "season_type",  # Season type (0=general, 1=summer, 2=winter)
    ]
    candidates = candidates.merge(
        item[["article_id", *item_feats]], on="article_id", how="left"
    )
    candidates[item_feats] = candidates[item_feats].astype("int8")  # Optimize memory

    # Final memory optimization
    candidates, _ = reduce_mem_usage(candidates)

    return candidates


def calc_embd_similarity(
    candidate: pd.DataFrame,
    user_embd: np.ndarray,
    item_embd: np.ndarray,
    sub: bool = True,
    item_id: str = "article_id",
) -> np.ndarray:
    """Calculate user-item embedding similarity.
    
    This function computes cosine similarity between user and item embeddings
    for all candidate user-item pairs. It processes data in batches to handle
    large datasets efficiently and avoid memory issues.
    
    The similarity is calculated using dot product of normalized embeddings,
    which is equivalent to cosine similarity when embeddings are L2-normalized.

    Parameters
    ----------
    candidate : pd.DataFrame
        DataFrame containing candidate user-item pairs
    user_embd : np.ndarray
        Pre-trained user embeddings matrix (shape: [num_users, embedding_dim])
    item_embd : np.ndarray
        Pre-trained item embeddings matrix (shape: [num_items, embedding_dim])
    sub : bool, optional
        Whether to subtract 1 from ID values (for 0-based indexing), by default True
    item_id : str, optional
        Column name for item IDs, by default "article_id"

    Returns
    -------
    np.ndarray
        Array of similarity scores for each candidate pair
        
    Notes
    -----
    - Assumes embeddings are L2-normalized for cosine similarity
    - Uses batch processing to handle large datasets
    - TODO: Consider adding embedding statistics (std, mean, etc.) as features
    """
    
    # Initialize similarity array
    sim = np.zeros(candidate.shape[0])
    batch_size = 10000  # Process in batches to manage memory
    
    # Process candidates in batches
    for batch in tqdm(range(0, candidate.shape[0], batch_size)):
        # Extract user and item IDs for current batch
        if sub:
            # Convert to 0-based indexing for embedding lookup
            tmp_users = (
                candidate.loc[batch : batch + batch_size - 1, "customer_id"].values - 1
            )
            tmp_items = (
                candidate.loc[batch : batch + batch_size - 1, item_id].values - 1
            )
        else:
            # Use IDs as-is (1-based indexing)
            tmp_users = candidate.loc[
                batch : batch + batch_size - 1, "customer_id"
            ].values
            tmp_items = candidate.loc[batch : batch + batch_size - 1, item_id].values
        
        # Get embeddings for current batch
        tmp_user_embd = np.expand_dims(user_embd[tmp_users], 1)  # (batch_size, 1, dim)
        tmp_item_embd = np.expand_dims(item_embd[tmp_items], 2)  # (batch_size, dim, 1)
        
        # Calculate dot product (cosine similarity for normalized embeddings)
        tmp_sim = np.einsum("ijk,ikj->ij", tmp_user_embd, tmp_item_embd)
        sim[batch : batch + batch_size] = tmp_sim.reshape(-1)
    
    return sim
