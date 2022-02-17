import os
import numpy as np
from scipy import io
from scipy.signal import convolve
from scipy import sparse
import random
import torch
from torch.utils.data import Dataset


def Y_factory(s, Y_size, A_size, density, SNR: float = 1, lattice_structure=None):
    """
    This function produces a QPI measurement with specified size, defect density, kernel size and number of levels
    """
    n1, n2 = Y_size
    m1, m2 = A_size
    A = kernel_factory(s, m1, m2)
    X = sparse.random(n1, n2, density)
    X = X / np.sum(X)

    signal = np.array([convolve(X.A, A[level], mode='same') for level in range(s)])
    signal_var = np.var(signal, axis=(-2, -1))
    noise_var = signal_var / SNR
    noise = np.random.normal(0, 1, size=(s, n1, n2)) * np.sqrt(noise_var)[:, None, None]

    if lattice_structure is None:
        Y = signal + noise
    else:
        lattice = lattice_background(s, Y_size, A_size, lattice_structure)
        lattice *= np.sqrt(noise_var / np.var(lattice))[:, None, None]  # Makes lattice var the same as noise_var
        Y = signal + (noise + lattice) / np.sqrt(2)  # Makes the total non-signal var: signal_var / SNR
    return Y, A, X


def kernel_factory(s: int, m1: int, m2: int):
    """
    This function produces a set of s random m1 by m2 kernels
    """
    m_max = max(m1, m2)
    A = np.zeros([s, m_max, m_max], dtype=float)
    symmetry = random.choice([2, 3, 4, 6])
    half_sym = np.floor(symmetry / 2).astype('int')
    lowest_k = 0.5
    highest_k = 3
    k = np.random.uniform(lowest_k, highest_k, [s, symmetry])
    x, y = np.meshgrid(np.linspace(-1, 1, m_max), np.linspace(-1, 1, m_max))
    arb_angle = np.random.uniform(0, 2 * np.pi)

    for direction in range(symmetry):
        ang = direction * 180 / symmetry
        ang = arb_angle + ang * np.pi / 180
        r = (x * np.cos(ang) + np.sin(ang) * y)
        phi = np.random.uniform(0, 2 * np.pi)
        for i in range(s):
            # Adding normal decay
            sigma = np.random.uniform(0.2, 0.5)
            decay = gaussian_window(m_max, m_max, sigma)
            A[i, :, :] += np.cos(2 * np.pi * k[i, direction % half_sym] * r) * decay

    # Normalizing:
    # A = np.abs(A)
    A = sphere_norm_by_layer(A)
    return A


def lattice_background(s: int, measurement_size: (int, int), kernel_size: (int, int),
                       structure: str = None) -> np.ndarray:
    """
    Generates a square lattice background in a given size
    """
    m_max = np.max(kernel_size)
    aspect_ratio = measurement_size[0] / measurement_size[1]
    atoms_in_kernel = np.random.uniform(0.9, 3)
    lambda_lattice = (np.max(kernel_size) / np.max(measurement_size)) / (2 * atoms_in_kernel)
    background = np.zeros((s, measurement_size[0], measurement_size[1]))
    x, y = np.meshgrid(np.linspace(-1, 1, measurement_size[1]),
                       np.linspace(-aspect_ratio, aspect_ratio, measurement_size[0]))  # Add nm scale instead of [-1,1]?
    theta = np.arctan(y / x)
    r = np.sqrt(x ** 2 + y ** 2)
    arb_angle = np.random.uniform(0, np.pi)
    d_theta = theta - arb_angle
    if structure is None:
        structure = random.choice(['square', 'honeycomb'])
    if structure == 'square':
        dir1, dir2 = r * np.cos(d_theta), r * np.sin(d_theta)
        k1, k2 = 2 * np.pi / lambda_lattice, 2 * np.pi / lambda_lattice
        background = np.array([np.cos(dir1 * k1) + np.cos(dir2 * k2) for _ in range(s)])
    elif structure == 'honeycomb':
        dir1, dir2, dir3 = r * np.cos(d_theta), r * np.cos(2 * np.pi / 3 - d_theta), r * np.cos(2 * np.pi / 3 + d_theta)
        k1, k2, k3 = 2 * np.pi / lambda_lattice, 2 * np.pi / lambda_lattice, 2 * np.pi / lambda_lattice
        background = np.array([np.cos(dir1 * k1) + np.cos(dir2 * k2) + np.cos(dir3 * k3) for _ in range(s)])
    background = sphere_norm_by_layer(background)
    return background


def gaussian_window(n1: int, n2: int, sig=1, mu=0):
    """
    This function produces a 2D gaussian of size n1 by n2
    """
    x, y = np.meshgrid(np.linspace(-1, 1, n1), np.linspace(-1, 1, n2))
    d = np.sqrt(x * x + y * y)
    return np.exp(-((d - mu) ** 2 / (2.0 * sig ** 2)))


def sphere_norm_by_layer(M: np.ndarray) -> np.ndarray:
    """
    Returns your matrix with each layer normalized to the unit sphere.
    """
    assert len(np.shape(M)) == 3, 'The matrix does not have 3 dim'
    return M / np.linalg.norm(M, axis=(-2, -1))[:, None, None]


def save_data(number_of_samples, measurement_size, kernel_size, SNR=2, training=False, validation=False, testing=False):
    files_in_folder = os.listdir()
    if training and 'training_dataset' not in files_in_folder:
        os.system("mkdir training_dataset")
    if validation and 'validation_dataset' not in files_in_folder:
        os.system("mkdir validation_dataset")
    if testing and 'testing_dataset' not in files_in_folder:
        os.system("mkdir testing_dataset")

    density_exponent = np.random.uniform(low=-3.5, high=-1.5, size=(number_of_samples,))

    E, n1, n2 = measurement_size
    # sample_ker_size = np.random.randint(low=kernel_size[0] / 2, high=kernel_size[0] * 2, size=number_of_samples)
    sample_SNR = np.random.uniform(SNR / 2, 5 * SNR, number_of_samples)

    for i in range(number_of_samples):
        temp_measurement, temp_kernel, temp_activation_map = Y_factory(E, (n1, n2),
                                                                       kernel_size,
                                                                       10 ** density_exponent[i],
                                                                       sample_SNR[i])
        if training:
            np.save(os.getcwd() + '/training_dataset/kernel_%d' % i, temp_kernel)
            np.save(os.getcwd() + '/training_dataset/measurement_%d' % i, temp_measurement)
            io.mmwrite(os.getcwd() + '/training_dataset/activation_%d' % i, temp_activation_map)
        elif validation:
            np.save(os.getcwd() + '/validation_dataset/kernel_%d' % i, temp_kernel)
            np.save(os.getcwd() + '/validation_dataset/measurement_%d' % i, temp_measurement)
            io.mmwrite(os.getcwd() + '/validation_dataset/activation_%d' % i, temp_activation_map)
        elif testing:
            np.save(os.getcwd() + '/testing_dataset/kernel_%d' % i, temp_kernel)
            np.save(os.getcwd() + '/testing_dataset/measurement_%d' % i, temp_measurement)
            io.mmwrite(os.getcwd() + '/testing_dataset/activation_%d' % i, temp_activation_map)
    if not training and not validation and not testing:
        print("Specify validation or training to save files.")


class QPIDataSet(Dataset):
    """
    Simple class that takes the data dir path and returns the data in (measurement, kernel, activation) triplets
    """

    def __init__(self, dataset_path):
        self.path2dataset = dataset_path
        self.files_in_folder = os.listdir(dataset_path)  # Listing all the files in the path
        self.length = len(self.files_in_folder) // 3
        self.files_in_folder.sort()  # Sorting the list

        # The first third are the activation maps
        self.activation_map = self.files_in_folder[:self.length]
        # The 2nd third are the kernels
        self.kernel = self.files_in_folder[self.length:2 * self.length]
        # The last third are the measurements
        self.measurement = self.files_in_folder[2 * self.length:3 * self.length]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        measurement = np.load(f'{self.path2dataset}/{self.measurement[idx]}')
        kernel = np.load(f'{self.path2dataset}/{self.kernel[idx]}')
        activation = io.mmread(f'{self.path2dataset}/{self.activation_map[idx]}').tolil()

        measurement = torch.FloatTensor(measurement)
        kernel = torch.FloatTensor(kernel)
        activation = torch.FloatTensor(activation.A)

        if torch.cuda.is_available():
            return measurement.cuda(), kernel.cuda(), activation.cuda()
        return measurement, kernel, activation
