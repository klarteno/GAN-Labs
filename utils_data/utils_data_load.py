import torch

from torchvision import transforms

from utils_os_path import DataFolderTrain
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader


data_folder_train = DataFolderTrain()
folder_train = data_folder_train.folder_train
folder_test = data_folder_train.folder_test
labels_attributes_path = data_folder_train.labels_attributes_path


def check_image_path(path_str: str):
    if path_str.lower().endswith(".jpg"):
        return True
    else:
        return False


################
####      CelebA
################


def __get_celeba_dataset(image_size=128):
    # For reference, ImageNet uses:
    ms = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    custom_transforms = transforms.Compose(
        [
            transforms.CenterCrop((160, 160)),
            transforms.Resize([image_size, image_size]),
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.ToTensor(),
            # https://github.com/soumith/ganhacks : normalize input images
            transforms.Normalize(*ms),
        ]
    )
    return ImageFolder(
        folder_train, transform=custom_transforms, is_valid_file=check_image_path
    )


def get_celeba_dataloaader_test(batch_size=1, image_size=64):
    # For reference, ImageNet uses:
    ms = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    custom_transforms = transforms.Compose(
        [
            transforms.CenterCrop((160, 160)),
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            # https://github.com/soumith/ganhacks : normalize input images
            transforms.Normalize(*ms),
        ]
    )

    dataset = ImageFolder(
        folder_test, transform=custom_transforms, is_valid_file=check_image_path
    )

    test_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, drop_last=False, pin_memory=True
    )

    return test_loader


def __get_celeba_dataloader(dataset, batch_size=32, workers=2):
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True )
        #num_workers=workers,
        #persistent_workers=True,
        #pin_memory=True,
   

    return train_loader


def get_celeba_datasets(batch_size=32):
    ##########################
    # Dataset
    ##########################
    train_dataset = __get_celeba_dataset(image_size=128)
    train_loader = __get_celeba_dataloader(dataset=train_dataset, batch_size=batch_size)

    return train_loader


def transform(idx, attr_names, desired_attr, file):
    attr = torch.tensor([int(entry) for entry in file[idx].split(",")[1:]])
    mask = [attr_names[1:][i] in desired_attr for i in range(len(attr))]
    masked = attr[mask]
    return torch.relu(masked).float()


from itertools import compress

# ImageFolder is used as a dataset which is a subclass of Dataset and reads the files from the folder in a sorted order(and tested on Celeba dataset) which means the csv atributes can be indexed , if smaller random datasets are used then use something like: file_path:str=train_dataset.samples.__getitem__(7)[0]   to get the file path of the 7th image in the dataset and then use the file path to get the attributes from the csv file
class CelebDataSset(Dataset):
    def __init__(self, train_dataset, desired_attr):
        self.desired_attr = desired_attr
        self.ds = train_dataset

        self.file = open(labels_attributes_path).read().split()
        self.attr_names = self.file[0].split(",")
        self.file = self.file[1:]

    def __getitem__(self, idx):
        """'
        attr = torch.tensor([int(entry) for entry in self.file[idx].split(',')[1:]])
        mask = [self.attr_names[1:][i] in self.desired_attr for i in range(len(attr))]
        masked = attr[mask]

        label_idx=torch.relu(masked).float()


        return self.ds[idx][0], label_idx
        """

        attr = [
            1.0 if int(entry) == 1 else 0.0 for entry in self.file[idx].split(",")[1:]
        ]
        mask = [self.attr_names[1:][i] in self.desired_attr for i in range(len(attr))]
        masked = list(compress(attr, mask))

        label_masked = (
            torch.tensor(masked, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)
        )

        return self.ds[idx][0], label_masked

    def __len__(self):
        return len(self.ds)


def get_celeba_datasets_with_labels(
    image_size=64, batch_size=32, desired_attr=["Male", "Young"]
):
    ##########################
    # Dataset
    ##########################

    train_dataset = __get_celeba_dataset(image_size=image_size)
    dataset = CelebDataSset(train_dataset, desired_attr=desired_attr)
    train_loader = __get_celeba_dataloader(dataset=dataset, batch_size=batch_size)

    return train_loader


def unnormalize_celeba(image):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    _mean = (
        -mean[0] / std[0],
        -mean[1] / std[1],
        -mean[2] / std[2],
    )  # equivalent to  -mean / std
    _std = (1.0 / std[0], 1.0 / std[1], 1.0 / std[2])  # equivalent to 1.0 / std

    unnormalize = transforms.Normalize(_mean, _std)

    return unnormalize(image)


################
#### FashionMnist
################

from torchvision import datasets


def get_fashion_mnist_dataset(batch_size=32, download=False, workers=3):
    ms = (0.5, 0.5)

    custom_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(*ms) # totensor normalizes between [0,1] and [-1,1] is subject to testing
        ]
    )

    train_dataset = datasets.FashionMNIST(
        root="Data", train=True, transform=custom_transforms, download=download
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_loader


def get_fashion_mnist_dataset_test(batch_size=1, download=False, workers=3):

    ms = (0.5, 0.5)

    custom_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(*ms),
        ]
    )

    test_dataset = datasets.FashionMNIST(
        root="Data", train=False, transform=custom_transforms, download=download
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # num_workers=workers,
    # persistent_workers=True,
    # pin_memory=True

    return test_loader


def unnormalize_FashionMNIST(image):

    ms = (0.5, 0.5)
    mean = ms[0]
    std = ms[1]

    # unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    unnormalize = transforms.Normalize(-mean / std, 1.0 / std)

    return unnormalize(image)


def normalize_FashionMNIST(image):
    ms = (0.5, 0.5)
    normalize = transforms.Normalize(*ms)
    return normalize(image)


def get_fashion_mnist_label(label):

    output_mapping = {
        0: "T-shirt/Top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    input = label.item() if type(label) == torch.Tensor else label

    return output_mapping[input]


""" 
if __name__ == '__main__':
    train_loader = get_celeba_datasets()

    image, _ = next(iter(train_loader))
    print(image.shape[0])
"""
