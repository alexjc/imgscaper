import numpy
import colorsys

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt


def rgb2hls(img):
    copy = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    converted = numpy.array([colorsys.rgb_to_hls(v[0]/255.0,v[1]/255.0,v[2]/255.0) for v in copy], dtype=numpy.float32)
    return converted.reshape(img.shape[0], img.shape[1], img.shape[2])


def _convertHls2Rgb(hls):
    rgb = colorsys.hls_to_rgb(*hls)
    return tuple(int(v * 255.0) for v in rgb)

def hls2rgb(img):
    copy = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    output = numpy.array([_convertHls2Rgb(col) for col in copy], dtype=numpy.uint8)
    return output.reshape(img.shape[0], img.shape[1], img.shape[2])


def blur(img, sigma):
    output = img.copy()
    output[:,:,0] = gaussian_filter(img[:,:,0], sigma=sigma)
    output[:,:,1] = gaussian_filter(img[:,:,1], sigma=sigma)
    output[:,:,2] = gaussian_filter(img[:,:,2], sigma=sigma)
    return output


def details(img):
    smooth = blur(img, sigma=24.0)
    diff = (img - smooth)
    return diff


def distance(img):
    return distance_transform_edt(img)


def normalized(array):
    mn, mx = min(array.flatten()), max(array.flatten())
    return (array - mn) / (mx - mn)
