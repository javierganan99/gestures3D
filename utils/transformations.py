import numpy as np


def translate_to_center(kpts):
    """
    Input skeleton and oputput translated to middle-shoulder point skeleton
    """
    steps = 3
    indexes = np.array([5, 6]) * steps
    center = np.array([np.mean(kpts[indexes + i]) for i in range(steps)])
    kpts[0::3] -= center[0]
    kpts[1::3] -= center[1]
    kpts[2::3] -= center[2]
    return kpts


def translate_to_center_batch(data_numpy):
    """
    Input batch of skeletons and oputput translated to middle-shoulder point batch
    """
    C, T, V, M = data_numpy.shape
    for frame in range(T):
        center = np.mean(data_numpy[:-1, frame, :, :], axis=1)
        for i in range(3):
            data_numpy[i, frame, :, :] -= center[i, 0]
    return data_numpy


def random_rotation(data_numpy, angle_ranges={"x": 20.0, "y": 20.0, "z": 20.0}):
    """
    Randomly rotate skeleton batch
    """

    def random_rotation_matrix(angle_ranges):
        # Generate random angles in radians
        rx = np.random.uniform(-angle_ranges["x"] * np.pi / 180, angle_ranges["x"] * np.pi / 180)
        ry = np.random.uniform(-angle_ranges["y"] * np.pi / 180, angle_ranges["y"] * np.pi / 180)
        rz = np.random.uniform(-angle_ranges["z"] * np.pi / 180, angle_ranges["z"] * np.pi / 180)

        # Create rotation matrices for each axis
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])

        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])

        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

        # Multiply rotation matrices in the order of z, y, x to obtain the final rotation matrix
        return np.matmul(Rz, np.matmul(Ry, Rx))

    R = random_rotation_matrix(angle_ranges)
    data_numpy[:-1, :, :, :] = np.einsum("ij,jklm->iklm", R, data_numpy[:-1, :, :, :])
    return data_numpy
