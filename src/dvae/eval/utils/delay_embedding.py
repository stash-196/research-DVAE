import numpy as np

def compute_delay_embedding(observation, delay, dimensions, handle_nan="remove"):
    """
    Compute the time-delay embedding of a 1D observation array.

    :param observation: 1D NumPy array of observations (may contain NaNs).
    :param delay: Time delay (tau) between dimensions.
    :param dimensions: Embedding dimension (m).
    :param handle_nan: How to handle NaNs ('remove' drops invalid rows; 'interpolate' fills linearly;
                        'zero' replaces with 0; 'mask' keeps but returns mask too).
    :return: Embedded array (N - (m-1)*tau, m). If 'mask', returns (embedded, valid_mask).
    """
    if not isinstance(observation, np.ndarray) or observation.ndim != 1:
        raise ValueError("observation must be a 1D NumPy array.")

    n = len(observation)
    embedding_length = n - (dimensions - 1) * delay
    if embedding_length <= 0:
        raise ValueError("Delay and dimensions too large for observation length.")

    # Handle NaNs in input observation
    if np.any(np.isnan(observation)):
        if handle_nan == "interpolate":
            # Linear interpolation (avoids bias if possible, but use cautiously)
            nan_mask = np.isnan(observation)
            xp = np.arange(n)[~nan_mask]
            fp = observation[~nan_mask]
            observation = np.interp(np.arange(n), xp, fp)
        elif handle_nan == "zero":
            observation = np.nan_to_num(observation, nan=0.0)
        # 'remove' and 'mask' handled after embedding

    # Build embedding
    embedded = np.empty((embedding_length, dimensions))
    for i in range(dimensions):
        embedded[:, i] = observation[i * delay : i * delay + embedding_length]

    # Post-embedding NaN handling
    if handle_nan in ["remove", "mask"]:
        valid_mask = ~np.any(np.isnan(embedded), axis=1)
        if handle_nan == "remove":
            embedded = embedded[valid_mask]
            if len(embedded) == 0:
                raise ValueError("All embeddings invalid after NaN removal.")
            return embedded
        else:  # 'mask'
            return embedded, valid_mask

    return embedded
