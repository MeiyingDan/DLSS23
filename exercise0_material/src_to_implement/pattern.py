import numpy as np
import matplotlib.pyplot as plt


class Checker():
    def __init__(self, resolution, tile_size):
        self.tile_size = tile_size   
        self.resolution = resolution  


    def draw(self):
        if self.resolution % (2*self.tile_size) == 0:
            black_cell = np.zeros((self.tile_size, self.tile_size))
            white_cell = np.ones((self.tile_size, self.tile_size))
            twoBW_firstrow = np.concatenate((black_cell, white_cell), axis=1)
            twoBW_secondrow = np.concatenate((white_cell, black_cell), axis=1)
            fourBW = np.concatenate((twoBW_firstrow, twoBW_secondrow), axis=0)
            self.output = np.tile(fourBW, (self.resolution // (2*self.tile_size), self.resolution // (2*self.tile_size)))

        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()








class Circle():
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        (x, y) = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
        distance = np.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)
        self.output = np.zeros((self.resolution, self.resolution))
        self.output[distance < self.radius] = 1
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()
class Spectrum():
    def __init__(self, resolution=100):
        self.resolution = resolution

    def draw(self):
        spectrum = np.zeros([self.resolution, self.resolution, 3])
        spectrum[:, :, 0] = np.linspace(0, 1, self.resolution)
        spectrum[:, :, 1] = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)
        'why reshape'
        spectrum[:, :, 2] = np.linspace(1, 0, self.resolution)
        self.output = spectrum
        return self.output.copy()

    def show(self):
        # plt.imshow(self.draw())
        plt.imshow(self.output)
        plt.show()







