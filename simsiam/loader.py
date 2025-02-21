# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import ImageFilter
import random




class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, train_transform, transform_aug):
        self.train_transform = train_transform
        self.transform_aug = transform_aug

    def __call__(self, x):
        q = self.train_transform(x)
        k = self.transform_aug(x)
        v = self.transform_aug(x)
        return [q, k,v]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
