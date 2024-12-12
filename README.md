# Linear Hourglass Transformers

This repository contains code for experimenting with **Linear Hourglass Transformers**. It is designed to help you understand and explore how these models work. The code is modular and easy to modify for your own experiments.

## Features

- Implementation of Linear Hourglass Transformers.
- Easy-to-follow code structure.
- Customizable parameters for training and evaluation.
- Sample datasets for quick testing.

## Requirements

You can install the required libraries via poetry by running:

```bash
poetry install
```

I encountered issues with the fast-transformers package while running GPU code on the computing cluster. If you face similar challenges with CUDA files, try running the `setup_env.sh` file located in the directory.

## Getting Started

### Clone the Repository

Clone this repository to your local machine using:

```bash
git clone https://github.com/yourusername/linear-hourglass-transformers.git
```

### Setting neptune environment variables

To properly log training metrics into neptune set the `default_project` and `default_token` with your credentials if you don't want to pass them each time running the training.

### Running the Code

1. Navigate to the project directory:
   
   ```bash
   cd linear-hourglass-transformers
   ```

2. Train the model using the default settings:

   ```bash
   python train_single_gpu.py
   ```

### Configurations

You can modify the configurations in the `config.json` file. This includes:

- Model architecture settings
- Training parameters
- Dataset paths

## Repository Structure

- `train_single_gpu.py`: Script to train the model.
- `models/`: Contains the implementation of Vanilla Transformer, Hourglass Transformer and Linear Hourglass Transformer along with its components.
- `data/`: Scripts for data loading and preprocessing.
- `config.json`: Configuration file for the model and training parameters.

## How It Works

The **Linear Hourglass Transformer** is designed to efficiently process sequential data. It uses a unique structure to compress and expand the input data while applying linear attention mechanisms. This makes it lightweight and suitable for various tasks like time series analysis, natural language processing, and more.

## Example Usage

Here is an example of how you can use this repository:

```python
from models.hourglass_transformer import HourglassTransformer

# Initialize model
model = HourglassTransformer(num_layers=4, num_heads=8, d_model=512)

# Train or evaluate your model
# (Refer to train.py or evaluate.py for detailed usage)
```

## Example usage

```python
train_single_gpu.py \                                                                                                    
 --task sequence_modelling \
 --dataset enwik9  \   
 --criterion cross_entropy \
 --model decoder_only \         
 --device cpu \    
 --mha_type cosformer \
 --act_fun relu \                 
 --structure 1x512,1x256,1x512 \
 --max_length 512 \
 --batch_size 1 \               
 --gradient_accumulation_steps 1 \       
 --epochs 1 \                          
 --init_lr 0.0005 \                       
 --tokenizer google/byt5-small \        
```
