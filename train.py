import torch
from PIL import Image
# from dataloader import ImageDataModule
from model import UNet
from torchsummary import summary
from datas import Dataloader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os
def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE.type)
    # Define hyperparameters
    n_filters = 32
    batch_size = 16
    input_shape = (batch_size, 3, 512, 512)  # Batch_size x Channels x Height x Width
    num_workers = 4
    val_split = 0.2
    test_split = 0.1
    epochs = 10
    class_map = {(i, 0, 0): i for i in range(13)}
    n_classes = len(class_map)

    # Initialize the Dataloader
    data_dir = '/kaggle/input/lyft-udacity-challenge/dataA/dataA'
    dm = Dataloader(
        data_dir=data_dir,
        class_map=class_map,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        test_split=test_split
    )
    # Setup the datasets
    dm.setup()

    # Initialize the LightningModule
    model = UNet(input_shape[1], n_filters= n_filters, n_classes= n_classes)
    model.to(DEVICE)
    #print model summary
    summary(model, input_shape[1:])

    #Initialize the logger
    logger = TensorBoardLogger("/kaggle/working/UNet-SUIM/lightning_logs", name=None, version=0)

    # Load the best checkpoint
    checkpoint = ModelCheckpoint(monitor="val_loss", save_top_k=1,  mode="min")
    filename="best_model-{epoch:02d}-{val_loss:.2f}"
    trainer = pl.Trainer(callbacks=[checkpoint], max_epochs=epochs, accelerator='cuda', devices=1)
    trainer.fit(model, dm)

    # Plot the loss and save it in the plots folder
    log_dir = logger.log_dir
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Extract loss and accuracy
    train_loss = event_acc.Scalars("train_loss")
    val_loss = event_acc.Scalars("val_loss")

    # Create plots folder if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot([x.step for x in train_loss], [x.value for x in train_loss], label="Train Loss", color="blue")
    plt.plot([x.step for x in val_loss], [x.value for x in val_loss], label="Validation Loss", color="red")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("plots/loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
