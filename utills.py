from scipy.signal import convolve2d
import os.path
import cv2
import numpy as np


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))
    
    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)
    
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(np.mean(ssim_map))


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def psnr(im1, im2):
    diff = np.float64(im1[:]) - np.float64(im2[:])
    rmse = np.sqrt(np.mean(diff ** 2))
    psnr = 20 * np.log10(255 / rmse)
    return psnr, rmse


def get_filenames(paths):
    filenames = []
    for path in paths:
        for root, dirs, files in os.walk(path):
            for f in files:
                filenames.append(os.path.join(root, f))
    return filenames


def get_files(paths):
    filenames = []
    for path in paths:
        
        for root, dirs, files in os.walk(path):
            for f in dirs:
                filenames.append(os.path.join(root, f))
    return filenames


def rgb2y(rgb):
    h, w, d = rgb.shape
    rgb = np.float32(rgb) / 255.0
    y = rgb * (np.reshape([65.481, 128.553, 24.966], [1, 1, 3]) / 255.0)
    y = y[:, :, 0] + y[:, :, 1] + y[:, :, 2]
    y = np.reshape(y, [h, w]) + 16 / 255.0
    return np.uint8(y * 255 + 0.5)


def img_to_uint8(img):
    img = np.clip(img, 0, 255)
    return np.round(img).astype(np.uint8)


rgb_to_ycbcr = np.array([[65.481, 128.553, 24.966],
                         [-37.797, -74.203, 112.0],
                         [112.0, -93.786, -18.214]])

ycbcr_to_rgb = np.linalg.inv(rgb_to_ycbcr)


def rgb2ycbcr(img):
    """ img value must be between 0 and 255"""
    img = np.float64(img)
    img = np.dot(img, rgb_to_ycbcr.T) / 255.0
    img = img + np.array([16, 128, 128])
    return img


def ycbcr2rgb(img):
    """ img value must be between 0 and 255"""
    img = np.float64(img)
    img = img - np.array([16, 128, 128])
    img = np.dot(img, ycbcr_to_rgb.T) * 255.0
    return img


def gaussian_kernel_2d_opencv(sigma=0.0):
    kx = cv2.getGaussianKernel(15, sigma)
    ky = cv2.getGaussianKernel(15, sigma)
    return np.multiply(kx, np.transpose(ky))
