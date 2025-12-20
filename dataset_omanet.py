import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import random
from PIL import Image


def transform(size=352):
    transform = {"image": transforms.Compose([
                        transforms.Resize((size, size)),
                        transforms.ToTensor()]),
                "binary" : transforms.Compose([
                        transforms.Resize((size, size)),
                        transforms.ToTensor()])
                }
    return transform
    

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def denoise(image, kernel_size=5):

    return cv2.blur(image, (kernel_size, kernel_size))

def calculate_snr_map(image, eps=1e-8):
    img_gray = np.array(image.convert('L')).astype(np.float32)
    img_denoised = denoise(img_gray)
    noise = np.abs(img_gray - img_denoised)
    noise = np.maximum(noise, eps)

    snr = img_denoised / noise  
    
    snr_min, snr_max = snr.min(), snr.max()
    snr_norm = (snr - snr_min) / (snr_max - snr_min + eps)
    snr_norm = np.clip(snr_norm, 0.0, 1.0)

    return Image.fromarray((snr_norm * 255).astype(np.uint8))


class Customtransform: 
    def __init__(self):
        pass

    def cv_random_flip(self, img, label, snr):
        if random.randint(0, 1) == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            snr = snr.transpose(Image.FLIP_LEFT_RIGHT)
        return img, label, snr

    def randomCrop(self, image, label, snr):
        border = 30
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_width = np.random.randint(image_width - border, image_width)
        crop_win_height = np.random.randint(image_height - border, image_height)
        random_region = (
            (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
            (image_height + crop_win_height) >> 1)
        return image.crop(random_region), label.crop(random_region), snr.crop(random_region)


    def randomRotation(self, image, label, snr):
        mode = Image.BICUBIC
        if random.random() > 0.8:
            random_angle = np.random.randint(-15, 15)
            image = image.rotate(random_angle, mode)
            label = label.rotate(random_angle, mode)
            snr = snr.rotate(random_angle, mode)
        return image, label, snr

    def __call__(self, img, label, snr):
        img, label, snr = self.cv_random_flip(img, label, snr)
        img, label, snr = self.randomCrop(img, label, snr)
        img, label, snr = self.randomRotation(img, label, snr)
        return img, label, snr



class CamObjDataset_snr(Dataset):
    def __init__(self, data_root, size=352):

        self.aug = Customtransform()
        self.transform = transform(size)
                
        images = sorted(os.listdir(os.path.join(data_root, 'low')))
        gts = sorted(os.listdir(os.path.join(data_root, 'Mask')))
        
        self.images = [os.path.join(data_root, 'low', x) for x in images if is_image_file(x)]
        self.gts = [os.path.join(data_root, 'Mask', x) for x in gts if is_image_file(x)]
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image =Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        snr = calculate_snr_map(image)

        # data augumentation
        image, gt, snr = self.aug(image, gt, snr)

        image = self.transform['image'](image)
        gt = self.transform['binary'](gt)
        snr = self.transform['binary'](snr)

        return image, gt, snr


class test_dataset_snr(Dataset):
    """load test dataset (batchsize=1)"""
    def __init__(self, data_root, size=352):

        self.transform = transform(size)
                
        images = sorted(os.listdir(os.path.join(data_root, 'low')))
        gts = sorted(os.listdir(os.path.join(data_root, 'Mask')))
        
        self.images = [os.path.join(data_root, 'low', x) for x in images if is_image_file(x)]
        self.gts = [os.path.join(data_root, 'Mask', x) for x in gts if is_image_file(x)]
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image =Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        snr = calculate_snr_map(image)

        image = self.transform['image'](image)
        gt = self.transform['binary'](gt)
        snr = self.transform['binary'](snr)

        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        
        return image, gt, snr, name
    

class CamObjDataset(Dataset):
    def __init__(self, data_root, size=352):

        self.aug = Customtransform()
        self.transform = transform(size)
                
        images = sorted(os.listdir(os.path.join(data_root, 'low')))
        gts = sorted(os.listdir(os.path.join(data_root, 'Mask')))
        
        self.images = [os.path.join(data_root, 'low', x) for x in images if is_image_file(x)]
        self.gts = [os.path.join(data_root, 'Mask', x) for x in gts if is_image_file(x)]
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image =Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        
        # data augumentation
        image, gt = self.aug(image, gt)

        image = self.transform['image'](image)
        gt = self.transform['binary'](gt)

        return image, gt


class test_dataset(Dataset):
    """load test dataset (batchsize=1)"""
    def __init__(self, data_root, size=352):

        self.transform = transform(size)
                
        images = sorted(os.listdir(os.path.join(data_root, 'low')))
        gts = sorted(os.listdir(os.path.join(data_root, 'Mask')))

        self.images = [os.path.join(data_root, 'low', x) for x in images if is_image_file(x)]
        self.gts = [os.path.join(data_root, 'Mask', x) for x in gts if is_image_file(x)]
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image =Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')

        image = self.transform['image'](image)
        gt = self.transform['binary'](gt)

        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        
        return image, gt, name


def get_loader(data_root, batchsize, size, snr_opt=None, shuffle=None, drop_last=False):
        
    if snr_opt :
            dataset = CamObjDataset_snr(data_root, size)
    else : 
        dataset = CamObjDataset(data_root, size)
    
    data_loader = DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  drop_last=drop_last)
    return data_loader

