from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
#erweiterte importen 
import pandas as pd

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data: pd.DataFrame, mode: str):
        #Aufgabe 3/2 Konstruktur
        self.data = data
        self.mode = mode
        self._transform: tv.transform.Compose = tv.transforms.Compose([
                                                tv.transforms.ToPILImage(),
                                                tv.transforms.ToTensor(),
                                                tv.transforms.Normalize(mean=train_mean, std=train_std)])
        
    #Aufgabe 3/2 - __len__ funktion umschreibung
    def __len__(self):
        return len(self.data)
        
    #Aufgabe 3/2 - __getitem__(self, index)  umschreiben so, dass (Bild, entsprechend Label) zuruckgibt
    def __getitem__(self,index) -> tuple[torch.tensor,torch.tensor]:
        zeile = self.data.iloc[index]
        bild = imread(zeile['filename'], as_gray=True)
        bild = gray2rgb(bild) 
        bild = torch.tensor(self.transform(bild))
        label = torch.tensor([zeile['crack'], zeile['inactive']]) 
        return (bild,label)

    @property
    def transform(self):
        return self._transform
    
    @transform.setter
    def transform(self, transform):
        self._transform = transform