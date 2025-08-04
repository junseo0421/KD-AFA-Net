from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
from os.path import join, splitext, basename

def rand_crop(img, target_height, target_width):
    # reshape image to an appropriate size, and random crop to target size

    width = img.size[0]
    height = img.size[1]

    width_scale = target_width / width
    height_scale = target_height / height
    
    if height_scale >= 0.5:
        starting_x = random.randint(0, width - target_width)
        starting_y = random.randint(0, height - target_height)
    else:
        down_sample_ratio = height_scale / 0.5
        if round(down_sample_ratio*width) < target_width:
            down_sample_ratio = width_scale
        new_width = round(down_sample_ratio * width)
        new_height = round(down_sample_ratio * height)
        img = img.resize((new_width, new_height)) 
        starting_x = random.randint(0, new_width - target_width)
        starting_y = random.randint(0, new_height - target_height)
        
    img = img.crop((starting_x, starting_y, starting_x+target_width, starting_y+target_height))
    
    return img


def center_crop(img, target_height, target_width):
    # reshape image to an appropriate size, and center crop to target size
    
    width = img.size[0]
    height = img.size[1]

    width_scale = target_width / width
    height_scale = target_height / height
    
    if height_scale >= 0.5:
        starting_x = (width - target_width) / 2
        starting_y = (height - target_height) / 2
    else:
        down_sample_ratio = height_scale / 0.5
        if round(down_sample_ratio*width) < target_width:
            down_sample_ratio = width_scale
        new_width = round(down_sample_ratio * width)
        new_height = round(down_sample_ratio * height)
        img = img.resize((new_width, new_height)) 
        starting_x = (new_width - target_width) / 2
        starting_y = (new_height - target_height) / 2
        
    img = img.crop((starting_x, starting_y, starting_x+target_width, starting_y+target_height))
    
    return img

class dataset_norm(Dataset):
    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, imglist1=[], imglist2=[], imglist3=[]):
        self.transforms = transforms
        self.imgSize = imgSize
        self.inputsize = inputsize

        self.img_list1 = imglist1
        self.img_list2 = imglist2
        self.img_list3 = imglist3

        self.size = len(self.img_list1)

    def __getitem__(self, index):
        index = index % self.size

        img1 = Image.open(self.img_list1[index]).convert("L")
        img2 = Image.open(self.img_list2[index]).convert("L")
        img3 = Image.open(self.img_list3[index]).convert("L")

        img1 = np.array(img1)
        img2 = np.array(img2)
        img3 = np.array(img3)

        img_cat = np.stack([img1, img2, img3], axis=-1)
        img_cat = Image.fromarray(img_cat)
        img_cat = self.transforms(img_cat)

        c0 = img_cat[0].unsqueeze(0)
        gt = c0.repeat(3, 1, 1)

        i = (self.imgSize - self.inputsize) // 2

        iner_img = img_cat[:, :, i:i + self.inputsize]
        mask_img = np.ones((3, self.imgSize, self.imgSize))
        mask_img[:, :, i:i + self.inputsize] = iner_img

        return gt, mask_img

    def __len__(self):
        return self.size


class dataset_test4(Dataset):
    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1, imglist=[]):
        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        self.preSize = imgSize + 64 * (pred_step - 1)
        self.inputsize = inputsize
        self.inputsize2 = inputsize + 64 * (pred_step - 1)

        self.img_list = imglist
        self.size = len(self.img_list)

    def __getitem__(self, index):
        index = index % self.size
        name = self.img_list[index]
        img = Image.open(name).convert('RGB')

        i = (self.imgSize - self.inputsize) // 2

        if self.transforms is not None:
            img = self.transforms(img)

        iner_img = img

        mask_img = np.ones((3, self.preSize, self.preSize))

        if self.pred_step > 1:
            mask_img[:, i:i + self.inputsize2, i:i+self.inputsize2] = img
        else:
            mask_img[:, :, i:i + self.inputsize2] = iner_img

        return img, iner_img, mask_img, splitext(basename(name))[0], name.replace('\\', '/').split('/')[-2]

    def __len__(self):
        return self.size
