import importlib

import yaml
import torchvision.transforms as transforms


def make_dataset(cfg):
    # adapted only for CIFAR10
    # TODO: generalize for other datasets
    data_module = importlib.import_module(cfg['dataset']['lib'])

    train_transforms, test_transforms = make_transform(cfg)

    train_ds, test_ds = getattr(data_module, cfg['dataset']['name'])(
        data_dir=cfg['dataset']['data_dir'], train_transforms=train_transforms, test_transforms=test_transforms
    )

    return train_ds, test_ds


def make_transform(cfg):
    if cfg['dataset']['transforms'] == None:
        return None, None
    elif cfg['dataset']['transforms'] == 'basic':
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(size=32, padding=4),
                transforms.ToTensor(),  # between 0 and 1
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # between -1 and 1
            ]
        )

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # between 0 and 1  # between -1 and 1
        )

        return train_transform, test_transform
    else:
        raise NotImplementedError


def make_model(cfg):
    model_module = importlib.import_module(cfg['model']['lib'])

    model = getattr(model_module, cfg['model']['name'])(**cfg['model']['args'])

    return model


def make_optimizer(cfg, model):
    optim_module = importlib.import_module(cfg['optimizer']['lib'])
    optimizer = getattr(optim_module, cfg['optimizer']['name'])(model.parameters(), **cfg['optimizer']['args'])
    return optimizer


def make_scheduler(cfg, optimizer):
    sched_module = importlib.import_module(cfg['scheduler']['lib'])
    scheduler = getattr(sched_module, cfg['scheduler']['name'])(optimizer, **cfg['scheduler']['args'])
    return scheduler


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
