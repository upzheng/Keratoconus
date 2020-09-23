import numpy as np
import cv2
from PIL import Image
from PIL import ImageOps
import numpy.random as random
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


class ConvertFromInts(object):
    def __call__(self, image, label=None):
        return image.astype(np.float32), label


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32).flatten()

    def __call__(self, image, label=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), label


class kera_SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32).flatten()

    def __call__(self, image, label=None):
        image = image.astype(np.float32)
        ori_image = image + 0
        image -= self.mean
        image[ori_image == 0] = 0
        return image.astype(np.float32), label


class Rescale(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, image, label=None):
        image /= self.scale
        return image.astype(np.float32), label


class DivStd(object):

    def __init__(self, std):
        self.std = np.array(std, dtype=np.float32).flatten()

    def __call__(self, image, labels=None):
        image = image.astype(np.float32)
        image /= self.std
        return image.astype(np.float32), labels


class RandomMirror(object):
    def __call__(self, image, label):
        _, width, _ = image.shape
        # random.seed(10)
        if random.randint(2):  # 2
            image = image[:, ::-1]
        return image, label


class RandomFlip(object):
    def __call__(self, image, label):
        _, width, _ = image.shape
        if random.randint(2):  # 2
            image = image[::-1, :]
        return image, label


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, labels=None):
        if random.randint(2):  # 2
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, labels


class RandomRotate(object):

    def __call__(self, image, labels=None):
        if random.randint(2):  # 2
            image = Image.fromarray(image)
            alpha = np.random.randint(360)
            image = np.array(image.rotate(alpha))
        return image, labels


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels=None):
        for t in self.transforms:
            img, labels = t(img, labels)
        return img, labels


class BaseAugmentation(object):

    def __init__(self, size=300, rescale=255., mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        self.mean = mean
        self.size = size
        self.std = std
        self.rescale = rescale

        self.augment = Compose([
            Resize(self.size),
            ConvertFromInts(),
            Rescale(self.rescale),
            SubtractMeans(self.mean),
            DivStd(self.std)
        ])

    def __call__(self, img, label):
        return self.augment(img, label)


class Augmentation:

    def __init__(self, size=300, rescale=255., mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        self.mean = mean
        self.size = size
        self.std = std
        self.rescale = rescale

        self.augment = Compose([
            Resize(self.size),
            RandomMirror(),
            RandomFlip(),
            RandomRotate(),
            ConvertFromInts(),
            RandomBrightness(),
            Rescale(self.rescale),
            SubtractMeans(self.mean),
            DivStd(self.std)
        ])

    def __call__(self, img, label):
        return self.augment(img, label)


class Kerat_Augmentation:

    def __init__(self, size=300, rescale=255., mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), mode='train'):

        self.mean = mean
        self.size = size
        self.std = std
        self.rescale = rescale
        if mode == 'train':
            self.augment = Compose([
                ConvertFromInts(),
                Rescale(self.rescale),
                kera_SubtractMeans(self.mean),  # SubtractMeans(self.mean),
                DivStd(self.std)
            ])
        else:
            self.augment = Compose([
                ConvertFromInts(),
                Rescale(self.rescale),
                kera_SubtractMeans(self.mean),  # SubtractMeans(self.mean),
                DivStd(self.std)
            ])

    def __call__(self, img, label):
        return self.augment(img, label)
