import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        # TODO: implement constructor
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
            image = skimage.transform.resize(image, (self.image_size))       # add data to the list
            self.image_set.append((image, self.label[file_name[:-4]]))       # convert list to numpy array



    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method


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

                # batch_data = np.concatenate((batch_data, last_batch))
                batch_datas.append(last_batch)

            self.index += self.batch_size
            if self.index >= num_image:
                self.index = 0

        return tuple(batch_datas)




    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

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
        # return the current epoch number
        return self.current_epoch



    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        class_name = self.class_dict[self.x]
        return class_name


    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images, labels = self.next()
        class_names = self.class_names
        fig, axs = plt.subplots(1, len(images), figsize=(10, 10))     #creating multiple subplots in a single figure
        for i, ax in enumerate(axs):
            ax.imshow(images[i])
            ax.set_title(class_names[labels[i]])
            ax.axis('off')
        plt.show()





