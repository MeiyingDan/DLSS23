import numpy as np
import matplotlib.pyplot as plt
import json
from skimage import transform
import os

class ImageGenerator(object):
    def __init__(self, file_path, label_path, batch_size, image_size,
                 rotation = False, mirroring = False, shuffle = False,
                 in_epoch_flag=False):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.height = image_size[0]
        self.width = image_size[1]
        self.channel = image_size[2]
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        self.dic = {0: 'airplane',
               1: 'automobile',
               2: 'bird',
               3: 'cat',
               4: 'deer',
               5: 'dog',
               6: 'frog',
               7: 'horse',
               8: 'ship',
               9: 'truck'}

        with open(label_path) as f:
            self.label = json.load(f)

        self.labels = []
        for doc_name, lab in self.label.items():
            self.labels.append((int(doc_name), int(lab)))
        # (0 = ’airplane’; 1 = ’automobile’; 2 = ’bird’; 3 = ’cat’; 4 = ’deer’; 5 = ’dog’; 6= ’frog’; 7 = ’horse’; 8 = ’ship’; 9 = ’truck’ )
        self.labels = sorted(self.labels)

        self.imgs = []
        for filename in os.listdir(self.file_path):
            if filename.endswith('.npy'):
                file_path = os.path.join(self.file_path, filename)
                self.imgs.append((np.load(file_path), int(filename[:-4])))
        self.imgs = sorted(self.imgs, key=lambda x: x[1])

        self.imgs = [t[0] for t in self.imgs]

        self.num_pictures = len(self.imgs)

        self.in_epoch_flag = False
        self.epoch = 0
        self.count = 0

    def get_num_picture(self):
        return self.num_pictures

    def next(self):
        if self.in_epoch_flag == False:
            self.epoch_index = 0

            index_orig = np.arange(0, self.num_pictures, 1)
            index = index_orig

            # self.num_batch = self.num_pictures // self.batch_size
            self.remainder = self.num_pictures % self.batch_size

            if self.shuffle == True:
                index = np.random.permutation(index)

            self.batchs_index = []

            index = list(index)
            if self.remainder != 0:
                for i in range(self.batch_size - self.remainder):
                    if self.shuffle == True:
                        # randomly choose
                        k = np.random.choice(index)
                        index.append(k)
                    else:
                        # choose from the beginngen
                        index.append(index[i])

            self.num_batch = len(index) // self.batch_size


            for i in range(self.num_batch):
                sub_batch = []
                for j in range(self.batch_size):
                    sub_batch.append(index[j])

                self.batchs_index.append(sub_batch)
                index = index[self.batch_size:]

                # print("batch =", i)
                # print(self.batchs_index[i])
                # print('len=', len(self.batchs_index[i]))
                # print()

            # risizing
            for i in range(self.num_pictures):
                self.imgs[i] = transform.resize(self.imgs[i], (self.height, self.width, self.channel))

            # mirroring
            mirr = ['h', 'v', 'False']
            if self.mirroring == True:
                for i in range(self.num_pictures):
                    mirr_index = np.random.choice(mirr)
                    if mirr_index == 'h':
                        self.imgs[i] = np.flip(self.imgs[i], axis=1)
                    if mirr_index == 'v':
                        self.imgs[i] = np.flip(self.imgs[i], axis=0)
                    if mirr_index == 'False':
                        pass

            # rotation
            if self.rotation == True:
                rot = ['90', '180', '270']
                for i in range(self.num_pictures):
                    rot_index = np.random.choice(rot)
                    if rot_index == '90':
                        self.imgs[i] = np.rot90(self.imgs[i], k=1)
                    if rot_index == '180':
                        self.imgs[i] = np.rot90(self.imgs[i], k=2)
                    if rot_index == '270':
                        self.imgs[i] = np.rot90(self.imgs[i], k=3)

            self.imgs_batch = []
            self.labels_batch = []
            for i in range(self.num_batch):
                a = []
                b = []
                for k in self.batchs_index[i]:
                    a.append(self.imgs[k])  # array of image
                    b.append(self.labels[k][1])  # label of image
                self.imgs_batch.append([np.array(a),np.array(b)])

            self.next_batch = self.imgs_batch[self.epoch_index]
            self.count += 1

        if self.in_epoch_flag == True:
            if self.num_batch != self.epoch_index+1:
                self.epoch_index += 1
                self.next_batch = self.imgs_batch[self.epoch_index]

            else:
                self.epoch_index = 0
                self.epoch += 1
                self.in_epoch_flag == False
                self.count = 0

        if self.count != 0:
            self.in_epoch_flag = True

        return self.next_batch


    def current_epoch(self):
        # return self.epoch, self.epoch_index
        return self.epoch # for test

    def class_name(self, int_label):
        self.name = self.dic[int_label]
        return self.name

    def show(self):
        k = 1
        r = 5
        h = (self.batch_size // r) + (self.batch_size % r)
        for i in range(r):
            for j in range(h):
                plt.subplot(r, h, k)
                plt.imshow( self.next_batch[ 0 ][ k-1 ] )
                # name = self.class_name[ self.imgs_batch[self.choose_batch][k - 1][1] ]
                titl = self.dic[ self.next_batch[ 1 ][ k-1 ] ]
                plt.title(titl)
                if k >= self.batch_size:
                    break
                k += 1
        plt.show()


