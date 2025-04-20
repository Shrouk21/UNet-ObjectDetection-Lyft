
# UNet Model for Image Segmentation

## Overview
This project implements a U-Net model for image segmentation tasks using PyTorch and PyTorch Lightning. The U-Net architecture is widely used for tasks such as semantic segmentation, where the goal is to predict class labels for each pixel in an image. The model is designed to work with a multi-class segmentation problem and leverages deep convolutional layers for both encoding and decoding.

## Features
- U-Net architecture with both encoder and decoder blocks
- PyTorch Lightning integration for training and validation
- Model checkpointing and TensorBoard logging
- Cross-entropy loss for multi-class segmentation
- Learning rate scheduling with ReduceLROnPlateau
- Dataset used Lyft for self driving cars in Kaggle

## Requirements
- Python >= 3.7
- PyTorch >= 1.8
- PyTorch Lightning >= 1.3
- torchmetrics >= 0.3
- matplotlib >= 3.0
- PIL (Pillow) >= 8.0

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Installation
1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/yourrepository.git
    cd yourrepository
    ```

2. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Model Architecture

### Overview
The U-Net model consists of an encoder-decoder structure. The encoder progressively reduces spatial dimensions, extracting higher-level features, while the decoder reconstructs the spatial dimensions to generate the segmentation output. The model uses convolutional layers for feature extraction, batch normalization for stabilizing training, and skip connections to retain spatial information.

### Parameters
- `input_channels`: Number of input channels (e.g., 3 for RGB images).
- `n_filters`: Number of filters for the first convolution layer (default 32).
- `n_classes`: Number of output classes for segmentation (default 8).
- `lr`: Learning rate for the Adam optimizer (default 1e-3).

## Data Preparation

### Dataset Structure

Your dataset should be organized as follows:

```
dataA/
    ├── images/
    ├── masks/
```

The `images/` folder contains the input images, while the `masks/` folder contains the corresponding segmentation masks.

### Data Preprocessing
The data is preprocessed with standard transformations such as resizing and normalization. You can adjust the data pipeline as needed in `datas.py`.

## Training the Model

### Run Training

To train the model, run the following command:

```bash
python main.py
```

### Training Workflow
- The dataset is split into training, validation, and test sets.
- The model uses a U-Net architecture for segmentation.
- Training logs are saved using TensorBoard for monitoring.
- Model checkpoints are saved based on validation loss for later use.

### TensorBoard Logging
You can monitor the training process using TensorBoard. After training, start the TensorBoard server:

```bash
tensorboard --logdir lightning_logs
```

### Example Output
Loss curves (train and validation losses) will be plotted and saved as:

```bash
plots/loss_curve.png
```

## Results Visualization
Once training completes, the results (e.g., loss curves) are saved in the `plots` directory. Here’s an example of what’s generated:

```bash
plots/
    └── loss_curve.png
```

## Model Checkpoints
The best model, based on the lowest validation loss, is saved during training. You can load this model for inference or further training. The checkpoint is saved in the `lightning_logs` folder.

## Inference
To perform inference, load the trained model and pass input images for prediction. Modify the `main.py` script to suit your needs for inference.

## File Descriptions
- `model.py`: Contains the U-Net model definition.
- `main.py`: Script to initialize and train the model.
- `decoder.py`: Decoder block for U-Net architecture.
- `encoder.py`: Encoder block for U-Net architecture.
- `datas.py`: Handles dataset loading and preprocessing.
- `requirements.txt`: List of dependencies for the project.


