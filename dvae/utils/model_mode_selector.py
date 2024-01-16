#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def create_autonomous_mode_selector(seq_len, mode='all_true', autonomous_ratio=0.0):
    """
    Creates a mode selector array for RNN training.

    :param seq_len: Length of the sequence.
    :param mode: The mode of operation - 'all_true', 'all_false', 'half_half', or 'gradual'.
    :param autonomous_ratio: Ratio of autonomous steps for 'gradual' mode.
    :return: A list of booleans where False represents teacher-forcing mode and True represents autonomous mode.
    """
    if mode == 'all_true':
        return np.array([True] * seq_len)
    elif mode == 'all_false':
        return np.array([False] * seq_len)
    elif mode == 'half_half':
        half_len = seq_len // 2
        return np.array([False] * half_len + [True] * (seq_len - half_len))
    elif mode == 'scheduled_sampling':
        autonomous_len = int(seq_len * autonomous_ratio)
        return np.array([False] * (seq_len - autonomous_len) + [True] * (autonomous_len))
    elif mode == 'gradual':
        # Gradually increase the ratio of False (autonomous mode) in the sequence
        mode_selector = []
        autonomous_step = int(1 / autonomous_ratio) if autonomous_ratio > 0 else seq_len
        for i in range(seq_len):
            mode_selector.append(i % autonomous_step != 0)
        return np.array(mode_selector)
    else:
        raise ValueError("Invalid mode selected.")