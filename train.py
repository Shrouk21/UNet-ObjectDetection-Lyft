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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define hyperparameters
    num_classes = 7
    batch_size = 32
    input_shape = (batch_size, 3, 128, 128)  # Batch_size x Channels x Height x Width
    num_workers = 4
    val_split = 0.2
    data_dir = "Emotion Detection"
    epochs = 10

    # Initialize the LightningDataModule
    dm = dataloader(data_dir, batch_size, num_workers, val_split)
    dm.setup()
    # input_shape = dm.size() #Image shape (batch_size, 3, 128, 128)

    # Initialize the LightningModule
    model = UNet(input_shape, num_classes=num_classes)
    model.to(DEVICE)
    #print model summary
    summary(model, input_shape[1:])

    #Initialize the logger
    logger = TensorBoardLogger("lightning_logs", name="emodetector")

    # Load the best checkpoint
    checkpoint = ModelCheckpoint(monitor="val_loss", save_top_k=1,  mode="min")
    filename="best_model-{epoch:02d}-{val_loss:.2f}"
    trainer = pl.Trainer(callbacks=[checkpoint], max_epochs=epochs, accelerator=DEVICE.type)
    trainer.fit(model, dm)

    # Plot the loss and accuracy and save them in the plots folder
    log_dir = logger.log_dir
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Extract loss and accuracy
    train_loss = event_acc.Scalars("train_loss")
    val_loss = event_acc.Scalars("val_loss")
    train_acc = event_acc.Scalars("train_acc")
    val_acc = event_acc.Scalars("val_acc")

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

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot([x.step for x in train_acc], [x.value for x in train_acc], label="Train Accuracy", color="blue")
    plt.plot([x.step for x in val_acc], [x.value for x in val_acc], label="Validation Accuracy", color="red")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig("plots/accuracy_curve.png")
    plt.close()

if __name__ == "__main__":
    main()
