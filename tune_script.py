import os
import argparse

import torch
import lightning as L
from ray import tune
from ray.tune import CLIReporter
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import RayDDPStrategy, RayTrainReportCallback, RayLightningEnvironment, prepare_trainer
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

import utils
from learners.classification_learner import ClassificationLearner


def train_func(cfg):
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

    # define report metrics
    callbacks = [
        RayTrainReportCallback(),
        TuneReportCheckpointCallback(
            metrics={
                "loss": "avg_valid_loss",
                "accuracy": "avg_valid_accuracy",
            },
            on="validation_end",
        ),
    ]

    # training strategy
    if torch.cuda.device_count() == 1 or cfg['train_strategy'] == 'ddp':
        strategy = RayDDPStrategy()
    else:
        raise NotImplementedError

    # lightning trainer
    trainer = L.Trainer(
        callbacks=callbacks,
        max_epochs=cfg['max_epochs'],
        devices='auto',
        accelerator='auto',
        strategy=strategy,
        plugins=[RayLightningEnvironment()],
        accumulate_grad_batches=cfg['trainer']['accumulate_grad_batches'],
        gradient_clip_val=cfg['trainer']['gradient_clip_val'],
        log_every_n_steps=cfg['trainer']['log_every_n_steps'],
        val_check_interval=cfg['trainer']['val_check_interval'],
        limit_train_batches=cfg['trainer']['limit_train_batches'],
        limit_val_batches=cfg['trainer']['limit_val_batches'],
        enable_progress_bar=False,
    )

    trainer = prepare_trainer(trainer)
    trainer.fit(learner, train_loader, test_loader)


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

    # Ray Trainer definition
    scaling_config = ScalingConfig(
        num_workers=torch.cuda.device_count(), use_gpu=True, resources_per_worker={"CPU": 10, "GPU": 1}
    )

    run_config = RunConfig(
        name="tune_experiments",
        progress_reporter=CLIReporter(
            metric_columns=["loss", "accuracy"],
        ),
        callbacks=[WandbLoggerCallback(project=cfg["project"] + '_tune', log_config=True, entity='slavaheroes')],
        storage_path='/SSD/slava/ray-tune-experiments/',
    )

    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    # Ray Tune definition

    scheduler = ASHAScheduler(max_t=cfg['max_epochs'], grace_period=5, reduction_factor=2)

    # define search space
    # TODO: load from config file

    cfg['optimizer']['args']['lr'] = tune.grid_search([1e-4, 1e-3])
    cfg['dataset']['batch_size'] = tune.grid_search([16, 32, 64])

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": cfg},
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=2,  # number of random samples per grid search combination
            scheduler=scheduler,
        ),
    )

    results = tuner.fit()
    best_trial = results.get_best_result(metric="loss", mode='min', scope='last-5-avg')
    print("Best Trial: ", best_trial)

    print(f"Best trial config: {best_trial.config}")
