#imports
import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):

        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.current_index = 0
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.current_epoch = 0
        self.index = 0

        with open(self.label_path, 'r') as f:
            self.label = json.load(f)

        self.image_set = []      # create an empty list to hold the loaded data
        for file_name in os.listdir(self.file_path):
            image = np.load(os.path.join(self.file_path, file_name))         # load data from file
            image = skimage.transform.resize(image, (self.image_size))       # Das Bild wird geforderten Dimension umgewandelt
            self.image_set.append((image, self.label[file_name[:-4]]))       # jedes Bild wird zu dem List hinzugefügt 



    def next(self):
        num_image = len(self.image_set)
        num_batch = num_image // self.batch_size
        batch_datas = []

        if self.current_index == 0:
            self.current_epoch += 1

            for i in range(num_batch):
                batch_data = self.image_set[i * self.batch_size: (i + 1) * self.batch_size]
                batch_datas.append(batch_data)
            if num_image % self.batch_size != 0:
                num_batch = num_image // self.batch_size + 1
                num_repeat = self.batch_size - num_image % self.batch_size
                last_batch = np.concatenate((self.image_set[-self.batch_size + num_repeat:], self.image_set[:num_repeat]))
                batch_datas.append(last_batch)

            self.index += self.batch_size
            if self.index >= num_image:
                self.index = 0

        return tuple(batch_datas)

    def augment(self, img):
        shuffled_image = []
        if self.shuffle:
            permutation = np.random.permutation(len(self.img))
            shuffled_image.append([self.img[i] for i in permutation])
            # image_set = image_set[permutation, :, :]
            self.img = shuffled_image

        if self.mirroring:
            if np.random.random() < 0.5:
                self.img = np.flip(self.img, axis=0)
            if np.random.random() < 0.5:
                self.img = np.flip(self.img, axis=1)

        if self.rotation:
            k = np.random.choice([1, 2, 3])
            self.img = np.rot90(self.img, k, axes=(0, 1))
        return self.img

    def current_epoch(self):
        return self.current_epoch

    def class_name(self, x):
        class_name = self.class_dict[self.x]
        return class_name

    def show(self):
        images, labels = self.next()
        class_names = self.class_names
        fig, axs = plt.subplots(1, len(images), figsize=(10, 10))     #creating multiple subplots in a single figure
        for i, ax in enumerate(axs):
            ax.imshow(images[i])
            ax.set_title(class_names[labels[i]])
            ax.axis('off')
        plt.show()





