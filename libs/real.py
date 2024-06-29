import random
from libs.common import split_data
import rosbag
import math, os
import torch
from scipy import interpolate as inter
import matplotlib.pyplot as plt
import numpy as np


cmd_topic = '/cmd_vel'
twist_topic = '/vrpn_client_node/Car_2_Tracking/twist'
dataset_dir = 'datasets'

def read_bag(bag_file):

    try:
        bag = rosbag.Bag(bag_file)
    except:
        print('Could not find ROS bag files. Please make sure data have been downloaded from https://zenodo.org/records/12536536.')

    twist = np.array([[t.to_nsec(), math.sqrt(msg.twist.linear.x**2 + msg.twist.linear.y**2)] for  _, msg, t in bag.read_messages(topics=[twist_topic])]).T
    cmd = np.array([[t.to_nsec(), msg.linear.x] for _, msg, t in bag.read_messages(topics=[cmd_topic])]).T

    f = inter.interp1d(twist[0], twist[1], kind='linear')

    v_out = np.append(np.zeros((100, )), f(cmd[0]))
    v_in = np.append(np.zeros((100, )), cmd[1] + 0.3)

    return v_in, v_out

def gen_samples(history_len, pred_horizon, v_in, v_inc):
    return [{'x': torch.tensor(np.array([v_in[i:i+history_len]])).detach(), 
             'y': torch.tensor(np.array([v_inc[i+history_len:i+history_len+pred_horizon]])).detach()} 
                for i in range(len(v_in)-history_len)]

def gen_zeros(history_len, pred_horizon, N):
    return [{'x': torch.tensor(np.zeros((1, history_len))), 
             'y': torch.tensor(np.zeros((1, pred_horizon))).float()} 
                for i in range(N)]


def find_bag_files(root_folder):
    bag_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.bag'):
                bag_files.append(os.path.join(root, file))
    return bag_files


def extract(history_len, pred_horizon, data_dir, draw=True):

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    try:
        # Try to open the file in read mode
        dataset = torch.load(os.path.join(dataset_dir, 'dataset.pt'))

    except FileNotFoundError:

        dataset = []

        for file in find_bag_files(data_dir):

            v_in, v_out = read_bag(os.path.join(data_dir, file))
            dataset += gen_samples(history_len, pred_horizon, v_in, v_out)

            if draw:
                ts = np.arange(len(v_in))
                plt.figure(figsize=(7.5, 5))
                plt.plot(ts, v_in, label="Commanded velocity")
                plt.plot(ts, v_out, linestyle='dotted', label="Actual velocity")
                plt.grid(True, linestyle='--', linewidth=0.5)
                plt.legend()
                plt.show()
            
        dataset += gen_zeros(history_len, pred_horizon, len(dataset))
        torch.save(dataset, os.path.join(dataset_dir, 'dataset.pt'))
    
    return dataset


def data_generate(history_len, pred_horizon, split_ratio, data_dir, visualize=True):

    print("Loading data ... May take some time ...")
    dataset = extract(history_len, pred_horizon, data_dir, draw=visualize)

    print("Shuffling data ...")
    random.shuffle(dataset)
    train_data, valid_data, test_data = split_data(dataset, split_ratio)

    return train_data, valid_data, test_data