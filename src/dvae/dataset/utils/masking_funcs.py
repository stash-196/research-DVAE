import numpy as np


def generate_fixed_burst_mask(length, p_missing, burst_length=50):
    mask = np.zeros(length, dtype=int)
    num_bursts = int(p_missing * length / burst_length)
    starts = np.random.choice(length - burst_length, num_bursts, replace=False)
    for start in starts:
        mask[start : start + burst_length] = 1
    return mask


def generate_markovian_burst_mask(shape, pi_1, exp_burst_length):
    """
    Generate a Markovian burst mask for missing data.

    Parameters:
    - length (int): Length of the mask.
    - pi_1 (float): Desired long-run proportion of missing data (between 0 and 1).
    - p_01 (float): Probability of transitioning from missing (1) to present (0).

    Returns:
    - numpy array: Binary mask where 1 means missing, 0 means present.
    """
    length = shape[0]
    dim = shape[1] if len(shape) > 1 else 1

    exp_burst_length = exp_burst_length - 1

    pi_0 = 1 - pi_1
    p_11 = exp_burst_length / (1 + exp_burst_length)
    p_10 = 1 - p_11
    p_00 = 1 / pi_0 * ((1 + p_10) * pi_0 - p_10)
    p_01 = 1 - p_00

    P = np.array([[p_00, p_01], [p_10, p_11]])

    # print(f"p_00: {p_00}, p_01: {p_01}, p_10: {p_10}, p_11: {p_11}")
    # print(P)

    # Start with data present (state 0)
    masks = []
    # Generate the sequence
    for _ in range(dim):
        state = 0
        mask = []
        for _ in range(length):
            mask.append(state)
            state = np.random.choice([0, 1], p=P[state])
        masks.append(mask)
    return np.array(masks).T
