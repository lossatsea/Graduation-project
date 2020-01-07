import glob
import random
import os
import numpy as np
import csv
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

year_list = ["0001", "1996", "1997", "1998", "1999", "2000",
             "2001", "2002", "2003", "2004", "2005", "2006",
             "2007", "2008", "2009", "2010", "2011", "2012",
             "2013", "2014", "2015", "2017", "2018", "2019",
             "2020", "unknow"]

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train", start_year = 0):
        self.transform = transforms_
        # read image
        self.files = []
        image_folder = os.path.join(root, 'dataset')
        mode_folder = os.path.join(image_folder, mode)
        for i in range(start_year, len(year_list)):
            year = year_list[i]
            year_folder = os.path.join(mode_folder, year)
            #print(year_folder)
            img_files = glob.glob( os.path.join(year_folder, '*.jpg'))
            for file in img_files:
                self.files.append(file)
        print(len(self.files))

        # read label file
        self.labels = dict()
        with open(os.path.join(root, "attr_tag_14_binary.csv"), "r") as csvfile:
            reader = csv.reader(csvfile)
            index = 0
            for line in reader:
                if index == 0:
                    pass
                else:
                    self.labels[line[1]] = np.array(line[2:len(line)]).astype(np.float).tolist()
                index += 1
        print(len(self.labels))

    def __getitem__(self, index):
        #load image
        file_path = self.files[index % len(self.files)]
        img = Image.open(file_path)
        #get image name
        filename = os.path.basename(file_path)
        img = self.transform(img)
        #get label from dictionary according to image name
        label = self.labels[filename]
        label = torch.tensor(label)

        return {"image": img, "label": label}

    def __len__(self):
        return len(self.files)

    def getLen(self):
        return self.__len__()
