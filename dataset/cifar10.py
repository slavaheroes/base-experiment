import torchvision


def cifar10(data_dir, train_transforms, test_transforms):
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transforms)

    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transforms)

    return train_dataset, test_dataset
