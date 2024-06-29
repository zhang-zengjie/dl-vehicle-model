# Modeling and Motion Prediction of Autonomous Vehicles Using Deep Learning

**Author:** Zengjie Zhang (z.zhang3@tue.nl)

A Python project demonstrating how to use recursive neural network (RNN) to approximate the dynamic model of an autonomous vehicle and predict its motion.

## Information

### Associated Research Work

This library is associated with the Arxiv article in [https://arxiv.org/abs/2310.02843](https://arxiv.org/abs/2310.02843).

## Installation

### System Requirements

**Operating system**
 - *Windows* (compatible in general, succeed on 11)

**Python Environment**
 - Python version: test passed on `python=3.12`
 - **Recommended**: IDE ([VS code](https://code.visualstudio.com/) or [Pycharm](https://www.jetbrains.com/pycharm/)) and [Conda](https://www.anaconda.com/)
 - Required additional packages: `bagpy`, `torch`, `scipy`. Follow the `Quick Installation` for detailed configurations.
 
 
**Required Dataset**

This demo requires the ROS bag files in the [*Driving Data of a Real F1tenth Car*](https://zenodo.org/records/12536536) dataset. Download the `*.bag` files in this dataset and save them in a subfolder in this directory. The program will automatically locate these data files.

### Quick Installation
 
1. Install conda following this [instruction](https://conda.io/projects/conda/en/latest/user-guide/install/index.html);

2. Open the conda shell, and create an independent project environment;
```
conda create -n dl-vehicle-model python=3.12
```

3. In the same shell, activate the created environment
```
conda activate dl-vehicle-model
```

4. In the same shell, within the `dl-vehicle-model` environment, install the dependencies `bagpy`, `scipy`, and `torch`:
 ```
pip install bagpy
pip install scipy
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Running Instructions

- Open the main script `main.py` and choose the mode between `fit_model` and `predict_trajectory`:
    - `fit_model`: generating a dynamic model for the vehicle
    - `predict_trajectory`: predict the motion of the vehicle given historical trajectories

- Run the main script `main.py`;
- Watch the terminal for runtime information;
- The figures will show up at the end of running; They are also automatically saved in the root directory.