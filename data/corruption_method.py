"""
Code for generating common image corruptions.

Adapted from the ImageNet-C corruption implementations used in:
Benchmarking Neural Network Robustness to Common Corruptions and Perturbations
"""

import warnings
from io import BytesIO

import cv2
import numpy as np
import skimage as sk
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, zoom as scipy_zoom

from utils import set_seed


set_seed(42)
warnings.simplefilter("ignore", UserWarning)


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        values = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        values = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    x, y = np.meshgrid(values, values)
    aliased_disk = np.array((x ** 2 + y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255., mode="s&p", amount=c)
    return np.clip(x, 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for channel in range(3):
        channels.append(cv2.filter2D(x[:, :, channel], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))
    return np.clip(channels, 0, 1) * 255


def glass_blur(x, severity=1):
    c = [(0.7, 1, 2), (0.9, 2, 1), (1.1, 2, 3), (1.3, 3, 2), (1.5, 4, 2)][severity - 1]
    arr = cv2.GaussianBlur(np.array(x) / 255.0, ksize=(0, 0), sigmaX=c[0])
    h, w = arr.shape[:2]
    max_delta = c[1]
    for _ in range(c[2]):
        for yy in range(max_delta, h - max_delta):
            for xx in range(max_delta, w - max_delta):
                dx, dy = np.random.randint(-max_delta, max_delta + 1, size=2)
                arr[yy, xx], arr[yy + dy, xx + dx] = arr[yy + dy, xx + dx].copy(), arr[yy, xx].copy()
    arr = cv2.GaussianBlur(arr, ksize=(0, 0), sigmaX=c[0])
    return np.clip(arr, 0, 1) * 255


def motion_blur(x, severity=1):
    radius = [7, 9, 13, 17, 21][severity - 1]
    angle = [0, 15, 30, 45, 60][severity - 1]
    kernel = np.zeros((radius, radius), dtype=np.float32)
    kernel[radius // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((radius / 2 - 0.5, radius / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (radius, radius))
    kernel /= max(kernel.sum(), 1e-6)
    arr = np.array(x) / 255.0
    return np.clip(cv2.filter2D(arr, -1, kernel), 0, 1) * 255


def _clipped_zoom(channel, zoom_factor):
    h, w = channel.shape[:2]
    zoomed = scipy_zoom(channel, zoom_factor, order=1)
    zh, zw = zoomed.shape[:2]
    top = max((zh - h) // 2, 0)
    left = max((zw - w) // 2, 0)
    return zoomed[top:top + h, left:left + w]


def zoom_blur(x, severity=1):
    factors = [
        np.arange(1.00, 1.11, 0.02),
        np.arange(1.00, 1.16, 0.03),
        np.arange(1.00, 1.21, 0.03),
        np.arange(1.00, 1.26, 0.04),
        np.arange(1.00, 1.31, 0.05),
    ][severity - 1]
    arr = np.array(x) / 255.0
    out = arr.copy()
    for factor in factors[1:]:
        out += _clipped_zoom(arr, factor)
    out /= len(factors)
    return np.clip(out, 0, 1) * 255


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def snow(x, severity=1):
    c = [(0.10, 0.20), (0.15, 0.25), (0.20, 0.30), (0.25, 0.35), (0.30, 0.40)][severity - 1]
    arr = np.array(x) / 255.0
    layer = np.random.normal(size=arr.shape[:2], loc=c[0], scale=c[1]).clip(0, 1)
    layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=1.2 + severity * 0.2)
    layer = np.repeat(layer[:, :, None], 3, axis=2)
    return np.clip(arr * (1.0 - 0.18 * severity / 5.0) + layer * (0.35 + 0.08 * severity), 0, 1) * 255


def frost(x, severity=1):
    c = [0.18, 0.24, 0.30, 0.36, 0.44][severity - 1]
    arr = np.array(x) / 255.0
    noise = np.random.rand(*arr.shape[:2])
    frost_layer = gaussian_filter(noise, sigma=1.5 + severity)
    frost_layer = (frost_layer - frost_layer.min()) / (np.ptp(frost_layer) + 1e-6)
    frost_layer = np.stack([frost_layer * 0.85, frost_layer * 0.95, frost_layer], axis=2)
    return np.clip((1 - c) * arr + c * frost_layer, 0, 1) * 255


def fog(x, severity=1):
    c = [0.18, 0.25, 0.32, 0.40, 0.50][severity - 1]
    arr = np.array(x) / 255.0
    h, w = arr.shape[:2]
    noise = np.random.rand(h, w)
    fog_map = gaussian_filter(noise, sigma=max(h, w) / (8.0 - severity * 0.6))
    fog_map = (fog_map - fog_map.min()) / (np.ptp(fog_map) + 1e-6)
    fog_map = np.repeat(fog_map[:, :, None], 3, axis=2)
    return np.clip(arr * (1.0 - c) + fog_map * c + c * 0.15, 0, 1) * 255


def elastic_transform(x, severity=1):
    alpha, sigma = [(8, 2), (12, 3), (16, 4), (20, 5), (24, 6)][severity - 1]
    arr = np.array(x) / 255.0
    h, w = arr.shape[:2]
    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma=sigma, mode="reflect") * alpha
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma=sigma, mode="reflect") * alpha
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = (yy + dy, xx + dx)
    warped = np.empty_like(arr)
    for channel in range(arr.shape[2]):
        warped[:, :, channel] = map_coordinates(arr[:, :, channel], coords, order=1, mode="reflect")
    return np.clip(warped, 0, 1) * 255


def pixelate(x, severity=1):
    c = [0.70, 0.60, 0.50, 0.40, 0.32][severity - 1]
    image = x.convert("RGB")
    w, h = image.size
    small = image.resize((max(1, int(w * c)), max(1, int(h * c))), Image.BOX)
    return np.asarray(small.resize((w, h), Image.NEAREST))


def jpeg_compression(x, severity=1):
    quality = [80, 65, 50, 35, 25][severity - 1]
    buffer = BytesIO()
    x.convert("RGB").save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"))
