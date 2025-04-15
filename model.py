import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from decoder import Decoder
from encoder import Encoder


class UNet(pl.LightningModule):
    def __init__(self, input_channels, n_filters=32, n_classes=8, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.n_filters = n_filters
        
        self.encoder1 = Encoder(input_channels, n_filters)
        self.encoder2 = Encoder(n_filters, n_filters * 2)
        self.encoder3 = Encoder(n_filters * 2, n_filters * 4)
        self.encoder4 = Encoder(n_filters * 4, n_filters * 8, dropout_rate=0.3)
        self.encoder5 = Encoder(n_filters * 8, n_filters * 16, dropout_rate=0.3, maxpooling=False)

        self.decoder1 = Decoder(n_filters * 16, n_filters * 8, n_filters *8)
        self.decoder2 = Decoder(n_filters * 8, n_filters * 4, n_filters*4)
        self.decoder3 = Decoder(n_filters * 4, n_filters * 2, n_filters*2)
        self.decoder4 = Decoder(n_filters * 2, n_filters)

        self.final_conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(n_filters, n_classes, kernel_size=1)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes, average='micro')

        self.lr = lr

    def forward(self, x):
        cblock1 = self.encoder1(x)
        cblock2 = self.encoder2(cblock1[0])
        cblock3 = self.encoder3(cblock2[0])
        cblock4 = self.encoder4(cblock3[0])
        cblock5 = self.encoder5(cblock4[0])

        ublock1 = self.decoder1(cblock5[0], cblock4[1])
        ublock2 = self.decoder2(ublock1, cblock3[1])
        ublock3 = self.decoder3(ublock2, cblock2[1])
        ublock4 = self.decoder4(ublock3, cblock1[1])

        x = self.final_conv1(ublock4)
        x = self.final_conv2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(preds.argmax(dim=1), y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(preds.argmax(dim=1), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        y =y.long()
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)  
        acc = self.accuracy(preds.argmax(dim=1), y)  
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", acc, prog_bar=True)  
        return loss  


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def on_epoch_end(self):
        # Optionally log additional metrics here, such as learning rate or custom metrics
        pass
