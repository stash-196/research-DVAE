import numpy as np
import torch
from . import forward_kinematics

import pickle


# store l.hist as pickle data for later use in pytorch dataloader
def save_pickle(data, path):

    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved data to {path}")


def readCSVasFloat(filename, with_key=False):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    """
    returnArray = []
    lines = open(filename).readlines()
    if with_key:  # skip first line
        lines = lines[1:]
    for line in lines:
        line = line.strip().split(",")
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray


def writeCSVasFloat(filename, data_array, with_key=False, key=None):
    """
    Args
      filename: string. Path to the csv file
      data_array: data
    """
    with open(filename, "w") as f:
        if with_key:
            f.write(key)
        for vec in data_array:
            line_to_write = ",".join(map(str, vec)) + "\n"
            f.write(line_to_write)


def expmap2xyz_torch(expmap):
    """
    convert expmaps to joint locations
    :param expmap: N*99
    :return: N*32*3
    """
    parent, offset, rotInd, expmapInd = forward_kinematics._some_variables()
    xyz = forward_kinematics.fkl_torch(expmap, parent, offset, rotInd, expmapInd)
    return xyz


def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

    Args
      r: 1x3 exponential map
    Returns
      R: 3x3 rotation matrix
    """
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
    r0x = r0x - r0x.T
    R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x)
    return R


def expmap2rotmat_torch(r):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = (
        torch.eye(3, 3).repeat(n, 1, 1).float().cuda()
        + torch.mul(torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1)
        + torch.mul(
            (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)),
            torch.matmul(r1, r1),
        )
    )
    return R


def rotmat2quat(R):
    """
    Converts a rotation matrix to a quaternion
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

    Args
      R: 3x3 rotation matrix
    Returns
      q: 1x4 quaternion
    """
    rotdiff = R - R.T

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sintheta = np.linalg.norm(r) / 2
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps)

    costheta = (np.trace(R) - 1) / 2

    theta = np.arctan2(sintheta, costheta)

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)
    return q


def quat2expmap(q):
    """
    Converts a quaternion to an exponential map
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

    Args
      q: 1x4 quaternion
    Returns
      r: 1x3 exponential map
    Raises
      ValueError if the l2 norm of the quaternion is not close to 1
    """
    if np.abs(np.linalg.norm(q) - 1) > 1e-3:
        raise (ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]

    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps))
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0

    r = r0 * theta
    return r


def rotmat2expmap(R):
    return quat2expmap(rotmat2quat(R))


def find_indices_srnn(frame_num1, frame_num2, seq_len, input_n=10):
    """
    Adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478

    which originaly from
    In order to find the same action indices as in SRNN.
    https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    T1 = frame_num1 - 150
    T2 = frame_num2 - 150  # seq_len
    idxo1 = None
    idxo2 = None
    for _ in np.arange(0, 4):
        idx_ran1 = rng.randint(16, T1)
        idx_ran2 = rng.randint(16, T2)
        # print("subact1 {}".format(idx_ran1))
        # print("subact2 {}".format(idx_ran2))
        idxs1 = np.arange(idx_ran1 + 50 - input_n, idx_ran1 + 50 - input_n + seq_len)
        idxs2 = np.arange(idx_ran2 + 50 - input_n, idx_ran2 + 50 - input_n + seq_len)
        if idxo1 is None:
            idxo1 = idxs1
            idxo2 = idxs2
        else:
            idxo1 = np.vstack((idxo1, idxs1))
            idxo2 = np.vstack((idxo2, idxs2))
    return idxo1, idxo2


def _boundary_cuts(boundaries, num_frames):
    """Sorted sample indices strictly inside ``[1, num_frames)``.

    ``None`` or an empty list means the series is one piece. A window
    ``[start, start + seq_len)`` crosses a cut ``c`` when
    ``start < c < start + seq_len``.
    """
    if boundaries is None:
        return None
    cuts = np.asarray(boundaries, dtype=np.int64).ravel()
    if cuts.size == 0:
        return None
    n = int(num_frames)
    cuts = np.unique(cuts)
    cuts = cuts[(cuts > 0) & (cuts < n)]
    if cuts.size == 0:
        return None
    return cuts


def longest_unbroken_span(num_frames, boundaries) -> int:
    """Longest run that does not contain a split boundary."""
    n = int(num_frames)
    if n <= 0:
        return 0
    cuts = _boundary_cuts(boundaries, n)
    if cuts is None:
        return n
    edges = np.r_[0, cuts, n]
    return int(np.diff(edges).max())


def _find_indices_span(num_frames, seq_len, num_indices):
    """Historical single-span start picker. Local RNG, never the global one."""
    SEED = 1234567890
    # Local RNG: never reseed NumPy's global generator (it drives the PTF masks).
    rng = np.random.RandomState(SEED)

    T = num_frames - seq_len + 1
    n = int(T / num_indices)
    list0 = np.arange(0, T)
    list1 = np.arange(0, T, (n + 1))
    t = num_indices - len(list1)
    if t == 0:
        listf = list1
    else:
        list2 = np.setdiff1d(list0, list1)
        vis = True
        if not vis:
            rng.shuffle(list2)
        list2 = list2[:t]
        listf = np.concatenate((list1, list2))
    return listf


def _find_indices_strict(num_frames, seq_len, boundaries=None, observed=None):
    """Abutting windows inside each piece. The only uncovered tail is the remainder.

    ``observed`` is set when starts must not land on a missing sample. A
    tile that would start there moves forward to the next observed row, and
    the next window begins where that one ends, so the windows still do not
    overlap. A move that cannot fit inside the piece stops that piece.
    """
    n = int(num_frames)
    length = int(seq_len)
    if length < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}.")
    flags = None
    if observed is not None:
        flags = np.asarray(observed, dtype=bool).reshape(-1)
        if flags.shape[0] != n:
            raise ValueError(
                f"observed mask length {flags.shape[0]} != series length {n}"
            )
    cuts = _boundary_cuts(boundaries, n)
    edges = np.r_[0, cuts, n] if cuts is not None else np.asarray([0, n])
    starts = []
    for left, right in zip(edges[:-1], edges[1:]):
        pos = int(left)
        right = int(right)
        while pos + length <= right:
            start = pos
            if flags is not None and not flags[start]:
                start = pos + 1
                while start < right and not flags[start]:
                    start += 1
                if start + length > right:
                    break
            starts.append(start)
            pos = start + length
    if not starts:
        return np.asarray([], dtype=np.int64)
    return np.asarray(starts, dtype=np.int64)


def find_indices(
    num_frames,
    seq_len,
    num_indices,
    boundaries=None,
    strict_non_overlap=False,
    observed=None,
):
    """Window starts. ``boundaries`` are sample indices a window must not cross.

    No boundaries (``None`` or empty) is the historical single-span call.
    With boundaries, the same picker runs inside each piece, each with its
    own ``RandomState(1234567890)``. Starts are offsets into ``num_frames``.

    ``strict_non_overlap=False`` is that historical picker, including the
    local RNG. ``True`` tiles each piece with abutting windows; ``num_indices``
    is not used. ``observed`` is only read in that strict mode.
    """
    if strict_non_overlap:
        return _find_indices_strict(num_frames, seq_len, boundaries, observed)
    cuts = _boundary_cuts(boundaries, num_frames)
    if cuts is None:
        return _find_indices_span(num_frames, seq_len, num_indices)
    edges = np.r_[0, cuts, int(num_frames)]
    parts = []
    seq_len = int(seq_len)
    for start, end in zip(edges[:-1], edges[1:]):
        span = int(end - start)
        n_idx = span // seq_len
        if n_idx < 1:
            continue
        local = np.asarray(
            _find_indices_span(span, seq_len, n_idx), dtype=np.int64
        ).ravel()
        parts.append(local + int(start))
    if not parts:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(parts)


if __name__ == "__main__":
    # file_name = '/mnt/beegfs/perception/xbie/h3.6m/dataset/S1/walking_1.txt'
    # the_sequence = readCSVasFloat(file_name)
    # print(the_sequence.shape)

    # idxo1, idxo2 = find_indices_srnn(frame_num1=1600, frame_num2=1620, seq_len=10, input_n=10)
    # print(idxo1)
    # print(idxo2)

    num_frames = 150
    seq_len = 50
    num_indices = 16
    find_indices(num_frames, seq_len, num_indices)
