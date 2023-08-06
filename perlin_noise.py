import numpy as np

#Perlin Noise

def smooth(t):
    return t*t*(3 - 2*t)

def perlin_only_one_freq(size, dim, decay, freq):
    if dim == 0:
        return np.random.rand()
    
    perlin_freq = np.zeros([size]*dim)

    n = 2**freq
    scale = decay ** freq
    N = size // n

    seed = np.zeros([n] + [size]*(dim - 1))
    for i in range(n):
        seed[i] = perlin_only_one_freq(size, dim - 1, decay, freq)

    for i in range(n):
        for j in range(N):
            h = smooth(j/N)
            y = (1 - h) * seed[i] + h * seed[(i + 1) % n]
            perlin_freq[i*N + j] = y / scale
    
    return perlin_freq

def perlin_noise(size, dim, decay, starting_freq = 1):
    #randomly generate a grid of numbers
    if dim == 0:
        #return a single number
        return np.random.rand()
    
    perlin = np.zeros([size]*dim)

    for freq in range(starting_freq, int(np.log(size)/np.log(2)) + 1):
        perlin += perlin_only_one_freq(size, dim, decay, freq)
    
    perlin -= np.min(perlin)
    perlin /= np.max(perlin)

    return perlin