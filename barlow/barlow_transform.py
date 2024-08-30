# ==================================== IMPORTS ==================================== #

# Libraries
import torchvision.transforms as transforms

# Custom files
from utils.functions import *
from utils.constants import *
from utils.logger import _app_logger

# ===================================== CLASS ===================================== #

class BarlowTwinsTransform:
    def __init__(
        self,
        train=True,
        input_height=224,
        gaussian_blur=True,
        jitter_strength=1.0,
        normalize=None,
    ):
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.normalize = normalize
        self.train = train

        color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        color_transform = [
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            color_transform.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5
                )
            )

        self.color_transform = transforms.Compose(color_transform)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose(
                [transforms.ToTensor(), normalize]
            )

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((self.input_height, self.input_height)),
                transforms.RandomHorizontalFlip(p=0.5),
                self.color_transform,
                self.final_transform,
            ]
        )

        self.finetune_transform = None
        if self.train:
            self.finetune_transform = transforms.Compose(
                [
                    transforms.RandomCrop((32, 32), padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.finetune_transform = transforms.Compose(
                [
                    transforms.Resize((self.input_height, self.input_height)),
                    transforms.ToTensor(),
                ]
            )

    def __call__(self, sample):
        return (
            self.transform(sample),
            self.transform(sample),
            self.finetune_transform(sample),
        )