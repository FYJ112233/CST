from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import random
import math
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
import torchvision
import torchvision.transforms as transforms

from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True


class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, gray=2):
        self.gray = gray

    def __call__(self, img):

        idx = random.randint(0, self.gray)

        if idx == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
            img[0, :, :] = tmp_img
            img[1, :, :] = tmp_img
            img[2, :, :] = tmp_img
        return img


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            # img = Image.open(img_path)
            img = img.resize((144, 288), Image.ANTIALIAS)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class VideoDataset_train(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir,dataset_rgb, seq_len=12, sample='evenly', transform=None, index1=[], index2=[]):
        self.dataset_ir = dataset_ir
        self.dataset_rgb = dataset_rgb
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.index1 = index1
        self.index2 = index2

        size_h = 288
        size_w = 144

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.transform_thermal = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((size_h, size_w)),
            transforms.Pad(10),
            transforms.RandomCrop((size_h, size_w)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)])  

        

        self.transform_color = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((size_h, size_w)),
            transforms.Pad(10),
            transforms.RandomCrop((size_h, size_w)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            normalize,
            ChannelExchange(gray=2),  ### CA
            ChannelRandomErasing(probability=0.5)]) 
            # ChannelAdapGray(probability=0.5)]) 

        self.transform_color1 = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((size_h, size_w)),
            transforms.Pad(10),
            transforms.RandomCrop((size_h, size_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelExchange(gray=2)])

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):

        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[self.index2[index]]

        num_ir = len(img_ir_paths)

        img_rgb_paths,pid_rgb,camid_rgb = self.dataset_rgb[self.index1[index]]
        num_rgb = len(img_rgb_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num_ir))
        if num_ir < S: 
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num_ir / S)
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        sample_clip_rgb = []
        frame_indices_rgb = list(range(num_rgb))
        if num_rgb < S: 
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[s * 1:(s + 1) * 1]
                sample_clip_rgb.append(list(pool_rgb))
        else:
            inter_val_rgb = math.ceil(num_rgb / S)
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (inter_val_rgb * S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[inter_val_rgb * s:inter_val_rgb * (s + 1)]
                sample_clip_rgb.append(list(pool_rgb))

        sample_clip_rgb = np.array(sample_clip_rgb)


        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
            imgs_ir = []
            for index in indices:
                index=int(index)
                img_path = img_ir_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None:  
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)


            frame_indices = range(num_rgb)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]

            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break

                indices.append(index)
            indices=np.array(indices)
            imgs_rgb = []
            for index in indices:
                index=int(index)
                img_path = img_rgb_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_rgb.append(img)
            imgs_rgb = torch.cat(imgs_rgb, dim=0)
            return imgs_ir, pid_ir, camid_ir, imgs_rgb, pid_rgb, camid_rgb
        elif self.sample == 'video_train':
            idx1 = np.random.choice(sample_clip_ir.shape[1], sample_clip_ir.shape[0])
            number_ir = sample_clip_ir[np.arange(len(sample_clip_ir)), idx1]

            imgs_ir = []
            for index in number_ir:
                index = int(index)
                img_path = img_ir_paths[index]
                img = read_image(img_path)


                img = np.array(img) ### [3, 288 ,144]

                #### --------------------------------------------------------------
                #### self.transform_thermal： 随机擦除 + 灰度化。
                img = self.transform_thermal(img)

                # img = self.transform(img) # [3, 288 ,144]

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0) # [18, 288 ,144]

            idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
            number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]
            imgs_rgb = []
            imgs_rgb_aug = []
            for index in number_rgb:
                index = int(index)
                img_path = img_rgb_paths[index]
                img = read_image(img_path)


                img = np.array(img)

                

                # img = self.transform(img)
                img = self.transform_color(img)
                img1 = self.transform_color1(img)

                imgs_rgb.append(img)
                imgs_rgb_aug.append(img1)
            imgs_rgb = torch.cat(imgs_rgb, dim=0)
            imgs_rgb_aug = torch.cat(imgs_rgb_aug, dim=0)

            return imgs_ir, pid_ir, camid_ir, imgs_rgb, imgs_rgb_aug, pid_rgb, camid_rgb

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


class VideoDataset_test(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=12, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths) ##24

        S = self.seq_len #10#
        sample_clip_ir = []
        frame_indices_ir = list(range(num))
        if num < S:  
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (S - num)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num / S) ### 3
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir) ###ndarry(10, 3);(0 1 2) (3 4 5),...,(21 22 23), (23 23 23),(23 23 23)

        if self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index=0
            frame_indices = range(num)
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            last_seq = list(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list=[]
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
 
                    img = np.array(img)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
              
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list) 
            return imgs_array, pid, camid

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            num_ir = len(img_paths)
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs_ir = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid

        if self.sample == 'video_test':
            number = sample_clip_ir[:, 0] ##[0 3 6 9,...., 21, 23, 23]
            imgs_ir = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None: 
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0) ###[30, 288, 144]
            return imgs_ir, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

class VideoDataset_train_evaluation(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir, seq_len=12, sample='evenly', transform=None):
        self.dataset_ir = dataset_ir
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset_ir)

    def __getitem__(self, index):

        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[index]

        num_ir = len(img_ir_paths)

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
            imgs_ir = []
            for index in indices:
                index=int(index)
                img_path = img_ir_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None:  
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)


            return imgs_ir, pid_ir, camid_ir
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

