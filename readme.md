# MNIST Handwritten Digit Classification with PyTorch

This project implements a simple Artificial Neural Network (ANN) to classify handwritten digits from the MNIST dataset using PyTorch.

## Project Overview

The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9) and is widely used as a benchmark for image classification algorithms. This project builds a fully-connected neural network to classify these digits with high accuracy.

![Sample MNIST Digits](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

## Features

- Data loading and preprocessing using PyTorch's DataLoader
- Implementation of a multi-layer feed-forward neural network
- Training and evaluation pipeline
- Performance visualization (accuracy and loss curves)
- Model prediction visualization
- Trained model saving and loading

## Model Architecture

The neural network consists of:
- Input layer: 784 neurons (28×28 flattened images)
- Hidden layers: 512 → 256 → 128 neurons with ReLU activation
- Dropout layers (20%) for regularization
- Output layer: 10 neurons (one for each digit 0-9)
- Log softmax activation for classification

## Requirements

All required packages are listed in the `environment.yml` file. The main dependencies are:
- Python 3.10
- PyTorch 2.1.0
- torchvision 0.16.0
- matplotlib
- numpy

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/mnist-pytorch-classification.git
   cd mnist-pytorch-classification
   ```

2. Create the Conda environment:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate mnist-pytorch
   ```

## Usage

### Training the Model

Run the main script to train the model:

```bash
python mnist_classifier.py
```

This will:
1. Download the MNIST dataset (if not already downloaded)
2. Train the model for the specified number of epochs
3. Evaluate the model on the test set
4. Save the trained model to `mnist_ann_model.pth`
5. Generate visualizations of the training process and model predictions

### Customizing Training

You can modify the following hyperparameters in the script:
- `batch_size`: Number of samples per batch
- `learning_rate`: Learning rate for the optimizer
- `num_epochs`: Number of training epochs
- `device`: Whether to use CPU or GPU for training

## Results

With the default settings, the model should achieve around 97-98% accuracy on the test set after 10 epochs of training. The training script generates two visualization files:

1. `mnist_training_results.png`: Shows the training/testing loss and accuracy curves
2. `mnist_predictions.png`: Displays sample predictions with correct/incorrect highlighting

## Project Structure

```
mnist-pytorch-classification/
├── mnist_classifier.py     # Main training script
├── environment.yml         # Conda environment file
├── README.md               # This file
├── data/                   # Downloaded MNIST data (created automatically)
├── mnist_ann_model.pth     # Saved model weights (after training)
├── mnist_training_results.png  # Training visualization
└── mnist_predictions.png   # Prediction visualization
```

## Extending the Project

Some ideas for extending this project:
- Implement a Convolutional Neural Network (CNN) for improved accuracy
- Add data augmentation to improve model generalization
- Create a web interface for live digit recognition
- Experiment with different architectures and hyperparameters
- Add early stopping to prevent overfitting

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The MNIST dataset was created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- PyTorch framework and tutorials