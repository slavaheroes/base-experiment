# Vanilla Classification Learner with Cross-Entropy Loss
# https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks

import torch
import lightning as L
import torchmetrics


class ClassificationLearner(L.LightningModule):
    def __init__(self, model, optimizer, scheduler, config) -> None:
        super(ClassificationLearner, self).__init__()
        self.save_hyperparameters(ignore=['model', 'optimizer', 'scheduler'])

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self.criterion = torch.nn.CrossEntropyLoss()
        self.validation_step_outputs = []

        # define metrics
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=config['model']['args']['num_classes']
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=config['model']['args']['num_classes'])

    def forward(self, x):
        self.model(x)

    def loss_fn(self, x, y):
        return self.criterion(x, y)

    def on_train_start(self):
        pass

    def on_train_epoch_start(self):
        pass

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, labels)

        train_acc = self.accuracy(outputs, labels)
        train_f1 = self.f1_score(outputs, labels)

        self.log('loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', train_acc, prog_bar=False, on_epoch=True, sync_dist=True)
        self.log('train_f1', train_f1, prog_bar=False, on_epoch=True, sync_dist=True)

        return {'loss': loss}

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, labels)

        accuracy = self.accuracy(outputs, labels)
        f1 = self.f1_score(outputs, labels)
        return self.validation_step_outputs.append({'valid_loss': loss, 'valid_accuracy': accuracy, 'valid_f1': f1})

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        mean_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        mean_acc = torch.stack([x['valid_accuracy'] for x in outputs]).mean()
        mean_f1 = torch.stack([x['valid_f1'] for x in outputs]).mean()

        self.log('avg_valid_loss', mean_loss, sync_dist=True)
        self.log('avg_valid_accuracy', mean_acc, sync_dist=True)
        self.log('avg_valid_f1', mean_f1, sync_dist=True)

        self.validation_step_outputs.clear()

    def on_fit_end(self):
        pass

    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]
