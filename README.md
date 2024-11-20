# InceptionV4 Model for ShapeStacks Height Prediction

## Overview

This project trains an **InceptionV4** model to predict the height of objects using the **ShapeStacks** dataset. The model is implemented in **PyTorch** and requires GPU support for efficient training.

## Requirements

- Python 3.x
- PyTorch >= 1.8.0
- CUDA-enabled GPU

## How to Run

1. **Prepare the Dataset**
   Download and place the ShapeStacks dataset in the correct directory. (Inside COMP90086_2024_Project_train and COMP90086_2024_Project_test)

2. **Download the Pretrained model InceptionV4**

   Download model from website: https://huggingface.co/timm/inception_v4.tf_in1k

3. **Train the Model**
   Open the Jupyter Notebook:

   ```t
   jupyter notebook InceptionV4_best_performing_model_1468664_1443427.ipynb
   ```

   Train the model:

   ```python
   trainer = TaskTrainer(
       csv_file='COMP90086_2024_Project_train/train.csv',
       img_dir='COMP90086_2024_Project_train/train',
       model=inceptionv4_mlp(),
       batch_size=128,
       num_epochs=30,
       learning_rate=0.001
   )
   
   trainer.train()
   ```

4. **Monitor with TensorBoard**
   Start TensorBoard to visualize training metrics:

   ```
   tensorboard --logdir=./runs tensorboard --logdir=<class_dir_location>/class --port=<choose_a_port_eg_8000> 
   ```

   Access it at http://localhost:8000.

## Results

The notebook includes visualizations such as loss and accuracy curves to analyze the model's performance.