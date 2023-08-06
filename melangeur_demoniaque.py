import numpy as np
import images
from perlin_noise import perlin_noise

def blend(im1, im2, map_):
    return im1 * map_ + im2 * (1 - map_)

def get_size(im):
    size = max(im.shape[2], im.shape[3])
    #make size the closest higher power of 2
    size = np.ceil(np.log2(size))
    size = int(2**size)
    return size

def get_noise(im):
    size = get_size(im)
    noise = perlin_noise(size, 2, 1.2, starting_freq=3)
    noise = images.bw_np_to_torch(noise)
    noise = noise[:, :, :im.shape[2], :im.shape[3]]
    return noise

### Type 1 : Basic noise serving as a mask for the blending ###

def create_map(image, lamda_):
    noise = get_noise(image)
    noise = images.to_map(noise, lambda_ = lamda_)
    return noise

### Type 2 : One image's intensity is used as a mask for the blending ###

def create_image_map(image, lambda_):
    image = images.BlackAndWhite(image)
    image = images.to_map(image, lambda_ = lambda_)
    return image

### Type 3 : Edge detection ###

def create_outline_map(image, lambda_):
    image = images.edge(image)
    image = images.to_map(image, lambda_ = lambda_)
    return image

def create_smooth_outline_map(image, lambda_):
    image = images.smooth_edge_detector(image, 71)
    image = images.to_map(image, lambda_ = lambda_)
    return image

########
# Mixing types
########

### Type 1 and 2 ###

def create_perlin_image_map(im1, im2, lambda_, t = 0.7):
    noise1 = get_noise(im1)
    noise2 = get_noise(im2)

    map_ = create_image_map(im1, lambda_)
    
    noise = t * map_ * noise1 + (1 - t) * noise2
    noise = images.to_map(noise, lambda_ = lambda_)

    return noise


### Type 1 and 3 ###

def create_perlin_edge_map(im1, im2, lambda_, t = 0.7):
    noise1 = get_noise(im1)
    noise2 = get_noise(im2)

    edge1 = images.smooth_edge_detector(im1, 71)
    edge2 = images.smooth_edge_detector(im2, 71)

    noise = t * noise1 * edge1 * (1 - edge2) + (1 - t) * noise2
    noise = images.to_map(noise, lambda_ = lambda_)
    return noise

### Type 2 and 3 ###

def create_image_edge_map(im1, im2, lambda_):
    map_ = create_image_map(im1, lambda_)
    edge2 = images.smooth_edge_detector(im2, 71)

    map_ = map_ - edge2
    map_ = map_.clamp(0, 1)
    map_ = images.to_map(map_, lambda_ = lambda_)

    return map_