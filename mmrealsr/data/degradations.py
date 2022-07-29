import math
import numpy as np
import random
import torch

from basicsr.data import degradations as basic_deg
from basicsr.data.degradations import mesh_grid, sigma_matrix2, pdf2

# -------------------------------------------------------------------- #
# --------------------------- blur kernels --------------------------- #
# -------------------------------------------------------------------- #


def sample_two_sigma(sigma_x_range, sigma_y_range, isotropic=True, diff_lower_bound=0.1):
    """Randomly sample two difference sigma_x(y).

    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` is ignored
        sigma_x_large - sigma_x_small >= diff_lower_bound
    In the anisotropic mode
        (sigma_x_large + sigma_y_large) - (sigma_x_small + sigma_y_small) >= diff_lower_bound

    Returns:
        sigma_x_small (float)
        sigma_y_small (float)
        sigma_x_large (float)
        sigma_y_large (float)
    """
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        x_tot_length = sigma_x_range[1] - sigma_x_range[0]
        y_tot_length = sigma_y_range[1] - sigma_y_range[0]
        # sample the interval between two sigmas
        x_sample_length = np.random.uniform(0, x_tot_length)
        y_sample_length = np.random.uniform(max(diff_lower_bound - x_sample_length, 0), y_tot_length)
        # sample the start point, derivative end point
        sigma_x_small = np.random.uniform(sigma_x_range[0], sigma_x_range[1] - x_sample_length)
        sigma_x_large = sigma_x_small + x_sample_length
        sigma_y_small = np.random.uniform(sigma_y_range[0], sigma_y_range[1] - y_sample_length)
        sigma_y_large = sigma_y_small + y_sample_length
    else:
        x_tot_length = sigma_x_range[1] - sigma_x_range[0]
        # sample the interval between two sigmas
        x_sample_length = np.random.uniform(diff_lower_bound, x_tot_length)
        # sample the start point, derivative end point
        sigma_x_small = np.random.uniform(sigma_x_range[0], sigma_x_range[1] - x_sample_length)
        sigma_x_large = sigma_x_small + x_sample_length
        sigma_y_small = sigma_x_small
        sigma_y_large = sigma_x_large

    return sigma_x_small, sigma_y_small, sigma_x_large, sigma_y_large


def two_random_bivariate_Gaussian(kernel_size,
                                  sigma_x_range,
                                  sigma_y_range,
                                  rotation_range,
                                  noise_range=None,
                                  isotropic=True,
                                  diff_lower_bound=0.1):
    """Randomly generate two bivariate isotropic or anisotropic Gaussian kernels.

    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.
        sigma_x_large - sigma_x_small >= diff_lower_bound
    In the anisotropic mode
        (sigma_x_large + sigma_y_large) - (sigma_x_small + sigma_y_small) >= diff_lower_bound

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel_small (ndarray):
        kernel_large (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    sigma_x_small, sigma_y_small, sigma_x_large, sigma_y_large = sample_two_sigma(
        sigma_x_range, sigma_y_range, isotropic=isotropic, diff_lower_bound=diff_lower_bound)
    if isotropic is False:
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        rotation = 0

    kernel_small = basic_deg.bivariate_Gaussian(
        kernel_size, sigma_x_small, sigma_y_small, rotation, isotropic=isotropic)
    kernel_large = basic_deg.bivariate_Gaussian(
        kernel_size, sigma_x_large, sigma_y_large, rotation, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel_small.shape)
        kernel_small = kernel_small * noise
        kernel_large = kernel_large * noise
    kernel_small = kernel_small / np.sum(kernel_small)
    kernel_large = kernel_large / np.sum(kernel_large)

    return kernel_small, kernel_large

# def max_bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
#     """Generate a bivariate isotropic or anisotropic Gaussian kernel.
#
#     In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.
#
#     Args:
#         kernel_size (int):
#         sig_x (float):
#         sig_y (float):
#         theta (float): Radian measurement.
#         grid (ndarray, optional): generated by :func:`mesh_grid`,
#             with the shape (K, K, 2), K is the kernel size. Default: None
#         isotropic (bool):
#
#     Returns:
#         kernel (ndarray): normalized kernel.
#     """
#     if grid is None:
#         grid, _, _ = mesh_grid(kernel_size)
#     if isotropic:
#         sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
#     else:
#         sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
#     kernel = pdf2(sigma_matrix, grid)
#     kernel = kernel / np.sum(kernel)
#     return kernel

def three_random_bivariate_Gaussian(kernel_size,
                                  sigma_x_range,
                                  sigma_y_range,
                                  rotation_range,
                                  noise_range=None,
                                  isotropic=True,
                                  diff_lower_bound=0.1):
    """Randomly generate two bivariate isotropic or anisotropic Gaussian kernels.

    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.
        sigma_x_large - sigma_x_small >= diff_lower_bound
    In the anisotropic mode
        (sigma_x_large + sigma_y_large) - (sigma_x_small + sigma_y_small) >= diff_lower_bound

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel_small (ndarray):
        kernel_large (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    sigma_x_small, sigma_y_small, sigma_x_large, sigma_y_large = sample_two_sigma(
        sigma_x_range, sigma_y_range, isotropic=isotropic, diff_lower_bound=diff_lower_bound)
    if isotropic is False:
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        rotation = 0

    kernel_small = basic_deg.bivariate_Gaussian(
        kernel_size, sigma_x_small, sigma_y_small, rotation, isotropic=isotropic)
    kernel_large = basic_deg.bivariate_Gaussian(
        kernel_size, sigma_x_large, sigma_y_large, rotation, isotropic=isotropic)
    kernel_max = basic_deg.bivariate_Gaussian(
        kernel_size, sigma_x_range[1], sigma_y_range[1], rotation, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel_small.shape)
        kernel_small = kernel_small * noise
        kernel_large = kernel_large * noise
        kernel_max = kernel_max * noise
    kernel_small = kernel_small / np.sum(kernel_small)
    kernel_large = kernel_large / np.sum(kernel_large)
    kernel_max = kernel_max / np.sum(kernel_max)

    return kernel_small, kernel_large, kernel_max

def two_random_bivariate_generalized_Gaussian(kernel_size,
                                              sigma_x_range,
                                              sigma_y_range,
                                              rotation_range,
                                              beta_range,
                                              noise_range=None,
                                              isotropic=True,
                                              diff_lower_bound=0.1):
    """Randomly generate two different bivariate generalized Gaussian kernels.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        beta_range (tuple): [0.5, 8]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel_small (ndarray):
        kernel_large (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    sigma_x_small, sigma_y_small, sigma_x_large, sigma_y_large = sample_two_sigma(
        sigma_x_range, sigma_y_range, isotropic=isotropic, diff_lower_bound=diff_lower_bound)
    if isotropic is False:
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        rotation = 0

    # assume beta_range[0] < 1 < beta_range[1]
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel_small = basic_deg.bivariate_generalized_Gaussian(
        kernel_size, sigma_x_small, sigma_y_small, rotation, beta, isotropic=isotropic)
    kernel_large = basic_deg.bivariate_generalized_Gaussian(
        kernel_size, sigma_x_large, sigma_y_large, rotation, beta, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel_small.shape)
        kernel_small = kernel_small * noise
        kernel_large = kernel_large * noise
    kernel_small = kernel_small / np.sum(kernel_small)
    kernel_large = kernel_large / np.sum(kernel_large)

    return kernel_small, kernel_large

def three_random_bivariate_generalized_Gaussian(kernel_size,
                                              sigma_x_range,
                                              sigma_y_range,
                                              rotation_range,
                                              beta_range,
                                              noise_range=None,
                                              isotropic=True,
                                              diff_lower_bound=0.1):
    """Randomly generate two different bivariate generalized Gaussian kernels.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        beta_range (tuple): [0.5, 8]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel_small (ndarray):
        kernel_large (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    sigma_x_small, sigma_y_small, sigma_x_large, sigma_y_large = sample_two_sigma(
        sigma_x_range, sigma_y_range, isotropic=isotropic, diff_lower_bound=diff_lower_bound)
    if isotropic is False:
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        rotation = 0

    # assume beta_range[0] < 1 < beta_range[1]
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel_small = basic_deg.bivariate_generalized_Gaussian(
        kernel_size, sigma_x_small, sigma_y_small, rotation, beta, isotropic=isotropic)
    kernel_large = basic_deg.bivariate_generalized_Gaussian(
        kernel_size, sigma_x_large, sigma_y_large, rotation, beta, isotropic=isotropic)
    kernel_max = basic_deg.bivariate_generalized_Gaussian(
        kernel_size, sigma_x_range[1], sigma_y_range[1], rotation, beta, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel_small.shape)
        kernel_small = kernel_small * noise
        kernel_large = kernel_large * noise
        kernel_max = kernel_max * noise
    kernel_small = kernel_small / np.sum(kernel_small)
    kernel_large = kernel_large / np.sum(kernel_large)
    kernel_max = kernel_max / np.sum(kernel_max)

    return kernel_small, kernel_large, kernel_max

def two_random_bivariate_plateau(kernel_size,
                                 sigma_x_range,
                                 sigma_y_range,
                                 rotation_range,
                                 beta_range,
                                 noise_range=None,
                                 isotropic=True,
                                 diff_lower_bound=0.1):
    """Randomly generate two bivariate plateau kernels.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi/2, math.pi/2]
        beta_range (tuple): [1, 4]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel_small (ndarray):
        kernel_large (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    sigma_x_small, sigma_y_small, sigma_x_large, sigma_y_large = sample_two_sigma(
        sigma_x_range, sigma_y_range, isotropic=isotropic, diff_lower_bound=diff_lower_bound)
    if isotropic is False:
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        rotation = 0

    # TODO: this may be not proper
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel_small = basic_deg.bivariate_plateau(
        kernel_size, sigma_x_small, sigma_y_small, rotation, beta, isotropic=isotropic)
    kernel_large = basic_deg.bivariate_plateau(
        kernel_size, sigma_x_large, sigma_y_large, rotation, beta, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel_small.shape)
        kernel_small = kernel_small * noise
        kernel_large = kernel_large * noise
    kernel_small = kernel_small / np.sum(kernel_small)
    kernel_large = kernel_large / np.sum(kernel_large)

    return kernel_small, kernel_large

def three_random_bivariate_plateau(kernel_size,
                                 sigma_x_range,
                                 sigma_y_range,
                                 rotation_range,
                                 beta_range,
                                 noise_range=None,
                                 isotropic=True,
                                 diff_lower_bound=0.1):
    """Randomly generate two bivariate plateau kernels.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi/2, math.pi/2]
        beta_range (tuple): [1, 4]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel_small (ndarray):
        kernel_large (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    sigma_x_small, sigma_y_small, sigma_x_large, sigma_y_large = sample_two_sigma(
        sigma_x_range, sigma_y_range, isotropic=isotropic, diff_lower_bound=diff_lower_bound)
    if isotropic is False:
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        rotation = 0

    # TODO: this may be not proper
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel_small = basic_deg.bivariate_plateau(
        kernel_size, sigma_x_small, sigma_y_small, rotation, beta, isotropic=isotropic)
    kernel_large = basic_deg.bivariate_plateau(
        kernel_size, sigma_x_large, sigma_y_large, rotation, beta, isotropic=isotropic)
    kernel_max = basic_deg.bivariate_plateau(
        kernel_size, sigma_x_range[1], sigma_y_range[1], rotation, beta, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel_small.shape)
        kernel_small = kernel_small * noise
        kernel_large = kernel_large * noise
        kernel_max = kernel_max * noise
    kernel_small = kernel_small / np.sum(kernel_small)
    kernel_large = kernel_large / np.sum(kernel_large)
    kernel_max = kernel_max / np.sum(kernel_max)

    return kernel_small, kernel_large, kernel_max

def two_random_mixed_kernels(kernel_list,
                             kernel_prob,
                             kernel_size=21,
                             sigma_x_range=(0.6, 5),
                             sigma_y_range=(0.6, 5),
                             rotation_range=(-math.pi, math.pi),
                             betag_range=(0.5, 8),
                             betap_range=(0.5, 8),
                             noise_range=None,
                             diff_lower_bound=0.1):
    """Randomly generate two different mixed kernels, the only difference between two kernels is the sigma.

    Args:
        kernel_list (tuple): a list name of kernel types,
            support ['iso', 'aniso', 'skew', 'generalized', 'plateau_iso',
            'plateau_aniso']
        kernel_prob (tuple): corresponding kernel probability for each
            kernel type
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        beta_range (tuple): [0.5, 8]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel_small (ndarray):
        kernel_large (ndarray):
    """
    kernel_type = random.choices(kernel_list, kernel_prob)[0]
    if kernel_type == 'iso':
        kernel_small, kernel_large = two_random_bivariate_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            noise_range=noise_range,
            isotropic=True,
            diff_lower_bound=diff_lower_bound)
    elif kernel_type == 'aniso':
        kernel_small, kernel_large = two_random_bivariate_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            noise_range=noise_range,
            isotropic=False,
            diff_lower_bound=diff_lower_bound)
    elif kernel_type == 'generalized_iso':
        kernel_small, kernel_large = two_random_bivariate_generalized_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betag_range,
            noise_range=noise_range,
            isotropic=True,
            diff_lower_bound=diff_lower_bound)
    elif kernel_type == 'generalized_aniso':
        kernel_small, kernel_large = two_random_bivariate_generalized_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betag_range,
            noise_range=noise_range,
            isotropic=False,
            diff_lower_bound=diff_lower_bound)
    elif kernel_type == 'plateau_iso':
        kernel_small, kernel_large = two_random_bivariate_plateau(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betap_range,
            noise_range=None,
            isotropic=True,
            diff_lower_bound=diff_lower_bound)
    elif kernel_type == 'plateau_aniso':
        kernel_small, kernel_large = two_random_bivariate_plateau(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betap_range,
            noise_range=None,
            isotropic=False,
            diff_lower_bound=diff_lower_bound)
    else:
        raise NotImplementedError(f'kernel type {kernel_type} is not supported yet.')

    return kernel_small, kernel_large

def three_random_mixed_kernels(kernel_list,
                             kernel_prob,
                             kernel_size=21,
                             sigma_x_range=(0.6, 5),
                             sigma_y_range=(0.6, 5),
                             rotation_range=(-math.pi, math.pi),
                             betag_range=(0.5, 8),
                             betap_range=(0.5, 8),
                             noise_range=None,
                             diff_lower_bound=0.1):
    """Randomly generate two different mixed kernels, the only difference between two kernels is the sigma.

    Args:
        kernel_list (tuple): a list name of kernel types,
            support ['iso', 'aniso', 'skew', 'generalized', 'plateau_iso',
            'plateau_aniso']
        kernel_prob (tuple): corresponding kernel probability for each
            kernel type
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        beta_range (tuple): [0.5, 8]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel_small (ndarray):
        kernel_large (ndarray):
    """
    kernel_type = random.choices(kernel_list, kernel_prob)[0]
    if kernel_type == 'iso':
        kernel_small, kernel_large, kernel_max = three_random_bivariate_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            noise_range=noise_range,
            isotropic=True,
            diff_lower_bound=diff_lower_bound)
    elif kernel_type == 'aniso':
        kernel_small, kernel_large, kernel_max = three_random_bivariate_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            noise_range=noise_range,
            isotropic=False,
            diff_lower_bound=diff_lower_bound)
    elif kernel_type == 'generalized_iso':
        kernel_small, kernel_large, kernel_max = three_random_bivariate_generalized_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betag_range,
            noise_range=noise_range,
            isotropic=True,
            diff_lower_bound=diff_lower_bound)
    elif kernel_type == 'generalized_aniso':
        kernel_small, kernel_large, kernel_max = three_random_bivariate_generalized_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betag_range,
            noise_range=noise_range,
            isotropic=False,
            diff_lower_bound=diff_lower_bound)
    elif kernel_type == 'plateau_iso':
        kernel_small, kernel_large, kernel_max = three_random_bivariate_plateau(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betap_range,
            noise_range=None,
            isotropic=True,
            diff_lower_bound=diff_lower_bound)
    elif kernel_type == 'plateau_aniso':
        kernel_small, kernel_large, kernel_max = three_random_bivariate_plateau(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betap_range,
            noise_range=None,
            isotropic=False,
            diff_lower_bound=diff_lower_bound)
    else:
        raise NotImplementedError(f'kernel type {kernel_type} is not supported yet.')

    return kernel_small, kernel_large, kernel_max

def custom_random_add_gaussian_noise_pt(imgs,
                                        sigma_range=(0, 1.0),
                                        gray_prob=0,
                                        clip=True,
                                        rounds=False,
                                        use_diff_noise=False,
                                        gaussian_noise_diff_range=(1, 21)):
    if use_diff_noise:
        sample_diff_abs = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device) * (
                gaussian_noise_diff_range[1] - gaussian_noise_diff_range[0]) + gaussian_noise_diff_range[0]
        sample_sigma_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (sigma_range[1] - sample_diff_abs) + sigma_range[0]
        sample_sigma_large = sample_sigma_small + sample_diff_abs
    else:
        sample_sigma_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
        sample_sigma_large = sample_sigma_small.clone().detach()
    gray_noise = torch.rand(imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device)
    gray_noise = (gray_noise < gray_prob).float()
    noise_small = basic_deg.generate_gaussian_noise_pt(imgs[0], sample_sigma_small, gray_noise)
    noise_large = basic_deg.generate_gaussian_noise_pt(imgs[0], sample_sigma_large, gray_noise)

    outs = list()
    outs.append(imgs[0] + noise_small)
    outs.append(imgs[1] + noise_small)
    outs.append(imgs[2] + noise_large)

    if clip and rounds:
        outs = [torch.clamp((out * 255.0).round(), 0, 255) / 255. for out in outs]
    elif clip:
        outs = [torch.clamp(out, 0, 1) for out in outs]
    elif rounds:
        outs = [(out * 255.0).round() / 255. for out in outs]
    return outs

def custom_random_add_three_gaussian_noise_pt(imgs,
                                        sigma_range=(0, 1.0),
                                        gray_prob=0,
                                        clip=True,
                                        rounds=False,
                                        use_diff_noise=False,
                                        gaussian_noise_diff_range=(1, 21)):
    if use_diff_noise:
        sample_diff_abs = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device) * (
                gaussian_noise_diff_range[1] - gaussian_noise_diff_range[0]) + gaussian_noise_diff_range[0]
        sample_sigma_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (sigma_range[1] - sample_diff_abs) + sigma_range[0]
        sample_sigma_large = sample_sigma_small + sample_diff_abs
    else:
        sample_sigma_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
        sample_sigma_large = sample_sigma_small.clone().detach()
    gray_noise = torch.rand(imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device)
    gray_noise = (gray_noise < gray_prob).float()
    noise_small = basic_deg.generate_gaussian_noise_pt(imgs[0], sample_sigma_small, gray_noise)
    noise_large = basic_deg.generate_gaussian_noise_pt(imgs[0], sample_sigma_large, gray_noise)

    outs = list()
    outs.append(imgs[0] + noise_small)
    outs.append(imgs[1] + noise_small)
    outs.append(imgs[2] + noise_small)
    outs.append(imgs[3] + noise_large)

    if clip and rounds:
        outs = [torch.clamp((out * 255.0).round(), 0, 255) / 255. for out in outs]
    elif clip:
        outs = [torch.clamp(out, 0, 1) for out in outs]
    elif rounds:
        outs = [(out * 255.0).round() / 255. for out in outs]
    return outs

def custom_random_add_gaussian_noise_pt_anchor(imgs,
                                        sigma_range=(0, 1.0),
                                        gray_prob=0,
                                        clip=True,
                                        rounds=False,
                                        use_diff_noise=False,
                                        gaussian_noise_diff_range=(1, 21)):
    if use_diff_noise:
        sample_diff_abs = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device) * (
                gaussian_noise_diff_range[1] - gaussian_noise_diff_range[0]) + gaussian_noise_diff_range[0]
        sample_sigma_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (sigma_range[1] - sample_diff_abs) + sigma_range[0]
        sample_sigma_large = sample_sigma_small + sample_diff_abs
    else:
        sample_sigma_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
        sample_sigma_large = sample_sigma_small.clone().detach()
    gray_noise = torch.rand(imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device)
    gray_noise = (gray_noise < gray_prob).float()
    noise_small0 = basic_deg.generate_gaussian_noise_pt(imgs[0], sample_sigma_small, gray_noise)
    noise_small1 = basic_deg.generate_gaussian_noise_pt(imgs[1], sample_sigma_small, gray_noise)
    noise_large = basic_deg.generate_gaussian_noise_pt(imgs[3], sample_sigma_large, gray_noise)
    noise_max = basic_deg.generate_gaussian_noise_pt(imgs[2], (sigma_range[1]*torch.ones(imgs[2].size(0))).type_as(imgs[2]), gray_noise)

    outs = list()
    outs.append(imgs[0] + noise_small0)
    outs.append(imgs[1] + noise_small1)
    outs.append(imgs[2] + noise_max)
    outs.append(imgs[3] + noise_large)

    if clip and rounds:
        outs = [torch.clamp((out * 255.0).round(), 0, 255) / 255. for out in outs]
    elif clip:
        outs = [torch.clamp(out, 0, 1) for out in outs]
    elif rounds:
        outs = [(out * 255.0).round() / 255. for out in outs]
    return outs

def custom_random_add_gaussian_noise_pt_anchor_enh(imgs, anchor_enh,
                                        sigma_range=(0, 1.0),
                                        gray_prob=0,
                                        clip=True,
                                        rounds=False,
                                        use_diff_noise=False,
                                        gaussian_noise_diff_range=(1, 21)):
    if use_diff_noise:
        sample_diff_abs = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device) * (
                gaussian_noise_diff_range[1] - gaussian_noise_diff_range[0]) + gaussian_noise_diff_range[0]
        sample_sigma_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (sigma_range[1] - sample_diff_abs) + sigma_range[0]
        sample_sigma_large = sample_sigma_small + sample_diff_abs
    else:
        sample_sigma_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
        sample_sigma_large = sample_sigma_small.clone().detach()
    gray_noise = torch.rand(imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device)
    gray_noise = (gray_noise < gray_prob).float()
    noise_small0 = basic_deg.generate_gaussian_noise_pt(imgs[0], sample_sigma_small, gray_noise)
    noise_small1 = basic_deg.generate_gaussian_noise_pt(imgs[1], sample_sigma_small, gray_noise)
    noise_large = basic_deg.generate_gaussian_noise_pt(imgs[3], sample_sigma_large, gray_noise)
    noise_max = basic_deg.generate_gaussian_noise_pt(imgs[2], (sigma_range[1]*torch.ones(imgs[2].size(0))).type_as(imgs[2]), gray_noise)

    noise_anchor_enh = basic_deg.generate_gaussian_noise_pt(anchor_enh[1], sample_sigma_small, gray_noise)

    outs = list()
    outs.append(imgs[0] + noise_small0)
    outs.append(imgs[1] + noise_small1)
    outs.append(imgs[2] + noise_max)
    outs.append(imgs[3] + noise_large)

    anchor_enh[1] = anchor_enh[1] + noise_anchor_enh

    if clip and rounds:
        outs = [torch.clamp((out * 255.0).round(), 0, 255) / 255. for out in outs]
        anchor_enh[1] = torch.clamp((anchor_enh[1] * 255.0).round(), 0, 255) / 255.
    elif clip:
        outs = [torch.clamp(out, 0, 1) for out in outs]
        anchor_enh[1] = torch.clamp(anchor_enh[1], 0, 1)
    elif rounds:
        outs = [(out * 255.0).round() / 255. for out in outs]
        anchor_enh[1] = (anchor_enh[1] * 255.0).round() / 255.
    return outs, anchor_enh

def custom_random_add_gaussian_noise_pt_anchor_single(imgs,
                                        sigma_range=(0, 1.0),
                                        gray_prob=0,
                                        clip=True,
                                        rounds=False,
                                        use_diff_noise=False,
                                        gaussian_noise_diff_range=(1, 21)):
    if use_diff_noise:
        sample_diff_abs = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device) * (
                gaussian_noise_diff_range[1] - gaussian_noise_diff_range[0]) + gaussian_noise_diff_range[0]
        sample_sigma_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (sigma_range[1] - sample_diff_abs) + sigma_range[0]
        sample_sigma_large = sample_sigma_small + sample_diff_abs
    else:
        sample_sigma_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
        sample_sigma_large = sample_sigma_small.clone().detach()
    gray_noise = torch.rand(imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device)
    gray_noise = (gray_noise < gray_prob).float()
    noise_small = basic_deg.generate_gaussian_noise_pt(imgs[0], sample_sigma_small, gray_noise)
    noise_large = basic_deg.generate_gaussian_noise_pt(imgs[1], sample_sigma_large, gray_noise)
    noise_max = basic_deg.generate_gaussian_noise_pt(imgs[2],
                                                     (sigma_range[1] * torch.ones(imgs[2].size(0))).type_as(imgs[2]),
                                                     gray_noise)

    outs = list()
    outs.append(imgs[0] + noise_small)
    outs.append(imgs[1] + noise_large)
    outs.append(imgs[2] + noise_max)

    if clip and rounds:
        outs = [torch.clamp((out * 255.0).round(), 0, 255) / 255. for out in outs]
    elif clip:
        outs = [torch.clamp(out, 0, 1) for out in outs]
    elif rounds:
        outs = [(out * 255.0).round() / 255. for out in outs]
    return outs


def custom_random_add_gaussian_noise_pt_anchor_single_blur(imgs,
                                        sigma_range=(0, 1.0),
                                        gray_prob=0,
                                        clip=True,
                                        rounds=False,
                                        use_diff_noise=False,
                                        gaussian_noise_diff_range=(1, 21)):
    sample_sigma = torch.rand(
        imgs[0].size(0), dtype=imgs[0].dtype,
        device=imgs[0].device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    gray_noise = torch.rand(imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device)
    gray_noise = (gray_noise < gray_prob).float()
    noise = basic_deg.generate_gaussian_noise_pt(imgs[0], sample_sigma, gray_noise)

    outs = list()
    outs.append(imgs[0] + noise)
    outs.append(imgs[1] + noise)
    outs.append(imgs[2] + noise)

    if clip and rounds:
        outs = [torch.clamp((out * 255.0).round(), 0, 255) / 255. for out in outs]
    elif clip:
        outs = [torch.clamp(out, 0, 1) for out in outs]
    elif rounds:
        outs = [(out * 255.0).round() / 255. for out in outs]
    return outs


def custom_random_add_four_gaussian_noise_pt_anchor(imgs,
                                        sigma_range=(0, 1.0),
                                        gray_prob=0,
                                        clip=True,
                                        rounds=False,
                                        use_diff_noise=False,
                                        gaussian_noise_diff_range=(1, 21)):
    if use_diff_noise:
        sample_diff_abs = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device) * (
                gaussian_noise_diff_range[1] - gaussian_noise_diff_range[0]) + gaussian_noise_diff_range[0]
        sample_sigma_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (sigma_range[1] - sample_diff_abs) + sigma_range[0]
        sample_sigma_large = sample_sigma_small + sample_diff_abs
    else:
        sample_sigma_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
        sample_sigma_large = sample_sigma_small.clone().detach()
    gray_noise = torch.rand(imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device)
    gray_noise = (gray_noise < gray_prob).float()
    noise_small = basic_deg.generate_gaussian_noise_pt(imgs[0], sample_sigma_small, gray_noise)
    noise_large = basic_deg.generate_gaussian_noise_pt(imgs[0], sample_sigma_large, gray_noise)
    noise_max = basic_deg.generate_gaussian_noise_pt(imgs[0], (sigma_range[1]*torch.ones(imgs[0].size(0))).type_as(imgs[0]), gray_noise)

    outs = list()
    outs.append(imgs[0] + noise_small)
    outs.append(imgs[1] + noise_small)
    outs.append(imgs[2] + noise_small)
    outs.append(imgs[3] + noise_max)
    outs.append(imgs[4] + noise_large)

    if clip and rounds:
        outs = [torch.clamp((out * 255.0).round(), 0, 255) / 255. for out in outs]
    elif clip:
        outs = [torch.clamp(out, 0, 1) for out in outs]
    elif rounds:
        outs = [(out * 255.0).round() / 255. for out in outs]
    return outs


def custom_random_add_poisson_noise_pt(imgs,
                                       scale_range=(0, 1.0),
                                       gray_prob=0,
                                       clip=True,
                                       rounds=False,
                                       use_diff_noise=False,
                                       poisson_noise_diff_range=(0.05, 2.05)):
    if use_diff_noise:
        sample_diff_abs = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device) * (
                poisson_noise_diff_range[1] - poisson_noise_diff_range[0]) + poisson_noise_diff_range[0]
        sample_scale_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (scale_range[1] - sample_diff_abs) + scale_range[0]
        sample_scale_large = sample_scale_small + sample_diff_abs
    else:
        sample_scale_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        sample_scale_large = sample_scale_small.clone().detach()

    gray_noise = torch.rand(imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device)
    gray_noise = (gray_noise < gray_prob).float()

    noise_small = basic_deg.generate_poisson_noise_pt(imgs[0], sample_scale_small, gray_noise)
    noise_large = basic_deg.generate_poisson_noise_pt(imgs[0], sample_scale_large, gray_noise)

    outs = list()
    outs.append(imgs[0] + noise_small)
    outs.append(imgs[1] + noise_small)
    outs.append(imgs[2] + noise_large)

    if clip and rounds:
        outs = [torch.clamp((out * 255.0).round(), 0, 255) / 255. for out in outs]
    elif clip:
        outs = [torch.clamp(out, 0, 1) for out in outs]
    elif rounds:
        outs = [(out * 255.0).round() / 255. for out in outs]
    return outs

def custom_random_add_three_poisson_noise_pt(imgs,
                                       scale_range=(0, 1.0),
                                       gray_prob=0,
                                       clip=True,
                                       rounds=False,
                                       use_diff_noise=False,
                                       poisson_noise_diff_range=(0.05, 2.05)):
    if use_diff_noise:
        sample_diff_abs = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device) * (
                poisson_noise_diff_range[1] - poisson_noise_diff_range[0]) + poisson_noise_diff_range[0]
        sample_scale_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (scale_range[1] - sample_diff_abs) + scale_range[0]
        sample_scale_large = sample_scale_small + sample_diff_abs
    else:
        sample_scale_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        sample_scale_large = sample_scale_small.clone().detach()

    gray_noise = torch.rand(imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device)
    gray_noise = (gray_noise < gray_prob).float()

    noise_small = basic_deg.generate_poisson_noise_pt(imgs[0], sample_scale_small, gray_noise)
    noise_large = basic_deg.generate_poisson_noise_pt(imgs[0], sample_scale_large, gray_noise)

    outs = list()
    outs.append(imgs[0] + noise_small)
    outs.append(imgs[1] + noise_small)
    outs.append(imgs[2] + noise_small)
    outs.append(imgs[3] + noise_large)

    if clip and rounds:
        outs = [torch.clamp((out * 255.0).round(), 0, 255) / 255. for out in outs]
    elif clip:
        outs = [torch.clamp(out, 0, 1) for out in outs]
    elif rounds:
        outs = [(out * 255.0).round() / 255. for out in outs]
    return outs

def custom_random_add_poisson_noise_pt_anchor(imgs,
                                       scale_range=(0, 1.0),
                                       gray_prob=0,
                                       clip=True,
                                       rounds=False,
                                       use_diff_noise=False,
                                       poisson_noise_diff_range=(0.05, 2.05)):
    if use_diff_noise:
        sample_diff_abs = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device) * (
                poisson_noise_diff_range[1] - poisson_noise_diff_range[0]) + poisson_noise_diff_range[0]
        sample_scale_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (scale_range[1] - sample_diff_abs) + scale_range[0]
        sample_scale_large = sample_scale_small + sample_diff_abs
    else:
        sample_scale_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        sample_scale_large = sample_scale_small.clone().detach()

    gray_noise = torch.rand(imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device)
    gray_noise = (gray_noise < gray_prob).float()

    noise_small0 = basic_deg.generate_poisson_noise_pt(imgs[0], sample_scale_small, gray_noise)
    noise_small1 = basic_deg.generate_poisson_noise_pt(imgs[1], sample_scale_small, gray_noise)
    noise_large = basic_deg.generate_poisson_noise_pt(imgs[3], sample_scale_large, gray_noise)
    noise_max = basic_deg.generate_poisson_noise_pt(imgs[2], (scale_range[1]*torch.ones(imgs[2].size(0))).type_as(imgs[2]), gray_noise)

    outs = list()
    outs.append(imgs[0] + noise_small0)
    outs.append(imgs[1] + noise_small1)
    outs.append(imgs[2] + noise_max)
    outs.append(imgs[3] + noise_large)

    if clip and rounds:
        outs = [torch.clamp((out * 255.0).round(), 0, 255) / 255. for out in outs]
    elif clip:
        outs = [torch.clamp(out, 0, 1) for out in outs]
    elif rounds:
        outs = [(out * 255.0).round() / 255. for out in outs]
    return outs

def custom_random_add_poisson_noise_pt_anchor_enh(imgs, anchor_enh,
                                       scale_range=(0, 1.0),
                                       gray_prob=0,
                                       clip=True,
                                       rounds=False,
                                       use_diff_noise=False,
                                       poisson_noise_diff_range=(0.05, 2.05)):
    if use_diff_noise:
        sample_diff_abs = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device) * (
                poisson_noise_diff_range[1] - poisson_noise_diff_range[0]) + poisson_noise_diff_range[0]
        sample_scale_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (scale_range[1] - sample_diff_abs) + scale_range[0]
        sample_scale_large = sample_scale_small + sample_diff_abs
    else:
        sample_scale_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        sample_scale_large = sample_scale_small.clone().detach()

    gray_noise = torch.rand(imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device)
    gray_noise = (gray_noise < gray_prob).float()

    noise_small0 = basic_deg.generate_poisson_noise_pt(imgs[0], sample_scale_small, gray_noise)
    noise_small1 = basic_deg.generate_poisson_noise_pt(imgs[1], sample_scale_small, gray_noise)
    noise_large = basic_deg.generate_poisson_noise_pt(imgs[3], sample_scale_large, gray_noise)
    noise_max = basic_deg.generate_poisson_noise_pt(imgs[2], (scale_range[1]*torch.ones(imgs[2].size(0))).type_as(imgs[2]), gray_noise)

    noise_anchor = basic_deg.generate_poisson_noise_pt(anchor_enh[1], sample_scale_small, gray_noise)

    outs = list()
    outs.append(imgs[0] + noise_small0)
    outs.append(imgs[1] + noise_small1)
    outs.append(imgs[2] + noise_max)
    outs.append(imgs[3] + noise_large)
    anchor_enh[1] = anchor_enh[1]+noise_anchor

    if clip and rounds:
        outs = [torch.clamp((out * 255.0).round(), 0, 255) / 255. for out in outs]
        anchor_enh[1] = torch.clamp((anchor_enh[1] * 255.0).round(), 0, 255) / 255.
    elif clip:
        outs = [torch.clamp(out, 0, 1) for out in outs]
        anchor_enh[1] = torch.clamp(anchor_enh[1], 0, 1)
    elif rounds:
        outs = [(out * 255.0).round() / 255. for out in outs]
        anchor_enh[1] = (anchor_enh[1] * 255.0).round() / 255.
    return outs, anchor_enh

def custom_random_add_poisson_noise_pt_anchor_single(imgs,
                                       scale_range=(0, 1.0),
                                       gray_prob=0,
                                       clip=True,
                                       rounds=False,
                                       use_diff_noise=False,
                                       poisson_noise_diff_range=(0.05, 2.05)):
    if use_diff_noise:
        sample_diff_abs = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device) * (
                poisson_noise_diff_range[1] - poisson_noise_diff_range[0]) + poisson_noise_diff_range[0]
        sample_scale_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (scale_range[1] - sample_diff_abs) + scale_range[0]
        sample_scale_large = sample_scale_small + sample_diff_abs
    else:
        sample_scale_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        sample_scale_large = sample_scale_small.clone().detach()

    gray_noise = torch.rand(imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device)
    gray_noise = (gray_noise < gray_prob).float()

    noise_small = basic_deg.generate_poisson_noise_pt(imgs[0], sample_scale_small, gray_noise)
    noise_large = basic_deg.generate_poisson_noise_pt(imgs[1], sample_scale_large, gray_noise)
    noise_max = basic_deg.generate_poisson_noise_pt(imgs[2],
                                                    (scale_range[1] * torch.ones(imgs[2].size(0))).type_as(imgs[2]),
                                                    gray_noise)

    outs = list()
    outs.append(imgs[0] + noise_small)
    outs.append(imgs[1] + noise_large)
    outs.append(imgs[2] + noise_max)

    if clip and rounds:
        outs = [torch.clamp((out * 255.0).round(), 0, 255) / 255. for out in outs]
    elif clip:
        outs = [torch.clamp(out, 0, 1) for out in outs]
    elif rounds:
        outs = [(out * 255.0).round() / 255. for out in outs]
    return outs


def custom_random_add_poisson_noise_pt_anchor_single_blur(imgs,
                                       scale_range=(0, 1.0),
                                       gray_prob=0,
                                       clip=True,
                                       rounds=False,
                                       use_diff_noise=False,
                                       poisson_noise_diff_range=(0.05, 2.05)):
    sample_scale = torch.rand(
        imgs[0].size(0), dtype=imgs[0].dtype,
        device=imgs[0].device) * (scale_range[1] - scale_range[0]) + scale_range[0]

    gray_noise = torch.rand(imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device)
    gray_noise = (gray_noise < gray_prob).float()

    noise = basic_deg.generate_poisson_noise_pt(imgs[0], sample_scale, gray_noise)

    outs = list()
    outs.append(imgs[0] + noise)
    outs.append(imgs[1] + noise)
    outs.append(imgs[2] + noise)

    if clip and rounds:
        outs = [torch.clamp((out * 255.0).round(), 0, 255) / 255. for out in outs]
    elif clip:
        outs = [torch.clamp(out, 0, 1) for out in outs]
    elif rounds:
        outs = [(out * 255.0).round() / 255. for out in outs]
    return outs


def custom_random_add_four_poisson_noise_pt_anchor(imgs,
                                       scale_range=(0, 1.0),
                                       gray_prob=0,
                                       clip=True,
                                       rounds=False,
                                       use_diff_noise=False,
                                       poisson_noise_diff_range=(0.05, 2.05)):
    if use_diff_noise:
        sample_diff_abs = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device) * (
                poisson_noise_diff_range[1] - poisson_noise_diff_range[0]) + poisson_noise_diff_range[0]
        sample_scale_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (scale_range[1] - sample_diff_abs) + scale_range[0]
        sample_scale_large = sample_scale_small + sample_diff_abs
    else:
        sample_scale_small = torch.rand(
            imgs[0].size(0), dtype=imgs[0].dtype,
            device=imgs[0].device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        sample_scale_large = sample_scale_small.clone().detach()

    gray_noise = torch.rand(imgs[0].size(0), dtype=imgs[0].dtype, device=imgs[0].device)
    gray_noise = (gray_noise < gray_prob).float()

    noise_small = basic_deg.generate_poisson_noise_pt(imgs[0], sample_scale_small, gray_noise)
    noise_large = basic_deg.generate_poisson_noise_pt(imgs[0], sample_scale_large, gray_noise)
    noise_max = basic_deg.generate_poisson_noise_pt(imgs[0], (scale_range[1]*torch.ones(imgs[0].size(0))).type_as(imgs[0]), gray_noise)

    outs = list()
    outs.append(imgs[0] + noise_small)
    outs.append(imgs[1] + noise_small)
    outs.append(imgs[2] + noise_small)
    outs.append(imgs[3] + noise_max)
    outs.append(imgs[4] + noise_large)

    if clip and rounds:
        outs = [torch.clamp((out * 255.0).round(), 0, 255) / 255. for out in outs]
    elif clip:
        outs = [torch.clamp(out, 0, 1) for out in outs]
    elif rounds:
        outs = [(out * 255.0).round() / 255. for out in outs]
    return outs