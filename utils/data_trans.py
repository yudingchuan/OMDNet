import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairCompose(transforms.Compose):
    def __call__(self, imgs):
        for t in self.transforms:
            imgs = t(imgs)
        return imgs


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:  # parent default p=0.5
            return [F.hflip(img) for img in imgs]
        return imgs


class PairToTensor(transforms.ToTensor):
    def __call__(self, imgs):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return [F.to_tensor(pic) for pic in imgs]
