# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Image Classifier Project

## Project Overview

This project is an image classification model built using PyTorch. The model is trained on a dataset of flower images and can classify flowers into various categories. The project includes scripts for training the model (`train.py`) and for making predictions on new images using the trained model (`predict.py`).

The primary goal of this project is to develop a command-line application that can train a deep learning model on a dataset and then use that model to classify images.

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Predicting Image Classes](#predicting-image-classes)
- [Project Structure](#project-structure)
- [Important Notes](#important-notes)
- [License](#license)

## Getting Started

To get started with this project, you'll need to clone the repository and ensure you have the necessary dependencies installed. You can run the training script to train a new model or use the prediction script to classify images using a pre-trained model.

### Prerequisites

Before running the scripts, make sure you have the following installed:

- Python 3.x
- PyTorch
- torchvision
- argparse (usually included with Python)

You can install the necessary Python packages using `pip`:

```bash
pip install torch torchvision
```
## Usage

### Training the Model

The `train.py` script is used to train a new image classifier model using a dataset. You can specify various hyperparameters and options through command-line arguments.

#### Basic Usage

```bash
python train.py data_directory --save_dir save_directory --arch "resnet18" --learning_rate 0.001 --hidden_units 512 --epochs 5 --gpu
```

#### Command-Line Arguments

- `data_directory`: Directory containing the training and validation data.
- `--save_dir`: Directory where the model checkpoint will be saved (default is the current directory).
- `--arch`: Model architecture to use for training (e.g., "resnet18", "vgg16").
- `--learning_rate`: Learning rate for the optimizer (default is 0.001).
- `--hidden_units`: Number of hidden units in the classifier (default is 512).
- `--epochs`: Number of training epochs (default is 5).
- `--gpu`: Use GPU for training if available.

### Predicting Image Classes

The `predict.py` script is used to make predictions on new images using a trained model. You can predict the class of a single image or all images within a directory, with the results saved to a specified output file.

#### Single Image Prediction

To predict the class of a single image:

```bash
python predict.py /path/to/image checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu --output_file image_prediction.txt
```

#### Directory Prediction

To predict the classes for all images in a directory:

```bash
python predict.py /path/to/directory checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu --output_file directory_predictions.txt
```


#### Command-Line Arguments

- `/path/to/image`: Path to the image file you want to classify.
- `checkpoint.pth`: Path to the saved model checkpoint.
- `--top_k`: Number of top probable classes to display (default is 1).
- `--category_names`: Path to a JSON file mapping categories to names.
- `--gpu`: Use GPU for inference if available.
- `--output_file`: File to save the prediction results (default is predictions.txt).

#### Project Structure
├── assets/                  # Additional assets like images, etc.  
├── flowers_data/            # Dataset directory (copied into the repository)  
├── checkpoints/             # Directory where model checkpoints are saved  
├── Image Classifier Project.ipynb  # Jupyter Notebook for exploration and development  
├── train.py                 # Script to train the model  
├── predict.py               # Script to make predictions using the trained model  
├── cat_to_name.json         # JSON file mapping category numbers to flower names  
├── README.md                # This README file  
├── LICENSE                  # License for the project  


## Important Notes

- **Dataset:** Ensure that your dataset is correctly placed in the `flowers_data/` directory, with subdirectories for training (`train/`) and validation (`valid/`) images.
- **GPU Usage:** Training and inference can be significantly accelerated using a GPU. Make sure to enable the `--gpu` flag if you have a GPU available.
- **Model Architecture:** The scripts currently support common architectures like ResNet and VGG. Ensure that the architecture you specify is compatible with your needs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
