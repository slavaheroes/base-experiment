import os
import argparse

import torch
import lightning as L

import utils
from learners.classification_learner import ClassificationLearner


def define_logger(cfg):

    logger = L.pytorch.loggers.WandbLogger(
        entity='slavaheroes',
        project=cfg['project'],
        name=cfg['name'],
    )

    return logger


def train_func(cfg):

    logger = define_logger(cfg)

    # make dataset + dataloaders
    train_ds, test_ds = utils.make_dataset(cfg)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg['dataset']['batch_size'], shuffle=True, num_workers=cfg['dataset']['num_workers']
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg['dataset']['batch_size'], shuffle=False, num_workers=cfg['dataset']['num_workers']
    )

    print(f'Train batches: {len(train_loader)}')
    print(f'Test batches: {len(test_loader)}')

    # make models
    model = utils.make_model(cfg)

    # make optimizer
    optimizer = utils.make_optimizer(cfg, model)

    # make scheduler
    scheduler = utils.make_scheduler(cfg, optimizer)

    # make learner
    learner = ClassificationLearner(model=model, optimizer=optimizer, scheduler=scheduler, config=cfg)

    # define callbacks
    callbacks = [L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')]

    if cfg['EarlyStopping']['enable']:
        early_stop = L.pytorch.callbacks.EarlyStopping(
            **cfg['EarlyStopping']['args'],
        )
        callbacks.append(early_stop)

    if cfg['ModelCheckpoint']['enable']:
        model_checkpoint = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=os.path.join('/mnt', cfg['ModelCheckpoint']['dirpath'] + "_" + cfg['project'], cfg['name']),
            filename=cfg['ModelCheckpoint']['filename'],
            **cfg['ModelCheckpoint']['args'],
        )
        callbacks.append(model_checkpoint)

    # training strategy
    if torch.cuda.device_count() == 1:
        strategy = 'auto'
    elif cfg['train_strategy'] == 'ddp':
        strategy = 'ddp'
    else:
        raise NotImplementedError

    # lightning trainer
    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg['max_epochs'],
        devices='auto',
        accelerator='auto',
        strategy=strategy,
        accumulate_grad_batches=cfg['trainer']['accumulate_grad_batches'],
        log_every_n_steps=cfg['trainer']['log_every_n_steps'],
        val_check_interval=cfg['trainer']['val_check_interval'],
        num_sanity_val_steps=cfg['trainer']['num_sanity_val_steps'],
        gradient_clip_val=cfg['trainer']['gradient_clip_val'],
        limit_train_batches=cfg['trainer']['limit_train_batches'],
        limit_val_batches=cfg['trainer']['limit_val_batches'],
        overfit_batches=cfg['trainer']['overfit_batches'],
        deterministic=False,  # set to True for reproducibility
    )

    trainer.fit(learner, train_loader, test_loader)

    # validate with the best model
    trainer.validate(learner, test_loader, ckpt_path='best', verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config.yaml')
    parser.add_argument('--devices', type=str, help='Comma separated list of devices', default='0,1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set seed
    # sets seeds for numpy, torch and python.random
    L.seed_everything(args.seed, workers=True)

    # Set GPU env variable
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    # Read config file
    cfg = utils.read_yaml_file(args.config_path)

    train_func(cfg)
