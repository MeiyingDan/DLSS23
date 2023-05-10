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
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, c=False):
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
        self.c = c
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        with open(self.label_path, 'r') as f:
            self.label = json.load(f)

        # self.data_set = []       # create an empty list to hold the loaded data
        # for file_name in os.listdir(self.file_path):
        #     data = np.load(os.path.join(self.file_path, file_name))    # load data from file
        #     self.data_set.append(data)       # add data to the list
        # self.loaded_data = np.array(self.data_set)    # convert list to numpy array

        self.image_set = []
        for file_name in os.listdir(self.file_path):
            image = np.load(os.path.join(self.file_path, file_name))
            image = skimage.transform.resize(image, (self.image_size))
            self.image_set.append((image, self.label[file_name[:-4]]))



    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method


        return (images,labels)


        #return images, labels


    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        return img




    def current_epoch(self):
        # return the current epoch number
        return 0

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        pass

