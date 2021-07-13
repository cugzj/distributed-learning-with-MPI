import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

class Rotation(object):
    def __init__(self, degree, resample=False, expand=False, center=None):
        self.degree = degree

        self.resample = resample
        self.expand = expand
        self.center = center

    def __call__(self, img):
        
        def rotate(img, angle, resample=False, expand=False, center=None):
            """Rotate the image by angle and then (optionally) translate it by (n_columns, n_rows)
            Args:
            img (PIL Image): PIL Image to be rotated.
            angle ({float, int}): In degrees degrees counter clockwise order.
            resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
            expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
            center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
            """
                
            return img.rotate(angle, resample, expand, center)

        angle = self.degree

        return rotate(img, angle, self.resample, self.expand, self.center)

def get_dataset(bsz, num_task):
    train_loader, test_loader, labels = {}, {}, {}
    for i in range(num_task):
        transform = transforms.Compose([Rotation(180/num_task*i), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        trainset = MNIST(root='./dataset/mnist_data', train=True, download=True, transform=transform)
        testset = MNIST(root='./dataset/mnist_data', train=False, download=True, transform=transform)
        train_loader[i] = DataLoader(trainset, batch_size=bsz, shuffle=False)
        test_loader[i] = DataLoader(testset, batch_size=bsz, shuffle=False)
        labels[i] = list(range(10))
    return train_loader, test_loader, labels