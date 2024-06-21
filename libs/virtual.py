import numpy as np
import random
import torch
from common import split_data


def data_interpolate(x, yf, y0):
    # f(x) = a_0 + a_1*x + a_2*(x^2) + a_3*(x^3);
    # Cubic spline to connect the strait lines.
    a_0 = y0
    a_1 = 0
    a_2 = 3 * (yf - y0) / (x[-1] ** 2)
    a_3 = -2 * (yf - y0) / (x[-1] ** 3)
    return a_0 + a_1 * x + a_2 * (x ** 2) + a_3 * (x ** 3)


def data_generate(history_len, pred_horizon, split_ratio):

    print("Generating trajectories ...")
    lane_width = 5.25
    lanes = np.array([1.5, 0.5]) * lane_width
    horizons = [history_len, pred_horizon]
    t_stage = (0, 2, 6, 8)
    sampling_time = 0.1
    num_velocities = 150

    source_lane = lanes[0]
    target_lane = lanes[1]
    history_len = horizons[0]
    predict_len = horizons[1]

    sample_length = history_len + predict_len
    dataset = []
    t = np.arange(t_stage[0], t_stage[3], sampling_time)
    n_stage_1 = int(t_stage[1] / sampling_time)
    n_stage_2 = int(t_stage[2] / sampling_time)
    n_stage_3 = len(t)

    for v in np.linspace(10, 40, num_velocities):

        x = t * v
        y = np.full(x.shape, source_lane)
        y[n_stage_1:n_stage_2] = data_interpolate(x[n_stage_1:n_stage_2] - x[n_stage_1], target_lane, source_lane)
        y[n_stage_2:] = np.full(y[n_stage_2:].shape, target_lane)

        for i in range(n_stage_3 - sample_length):

            sample_x = x[i:i + sample_length] - x[i + history_len - 1]
            sample_y = y[i:i + sample_length]
            sample_xy = np.vstack((sample_x, sample_y))
            sample_item = {'x': torch.tensor(sample_xy[:, :history_len]).detach(), 'y': torch.tensor(sample_xy[:, history_len:]).detach()}
            dataset.append(sample_item)

    print("Shuffling data ...")
    random.shuffle(dataset)

    train_data, validation_data, test_data = split_data(dataset, split_ratio)

    return train_data, validation_data, test_data


def data_split(data, horizons):
    return data[:, :, :horizons[0]], data[:, :, horizons[0]:]


def data_convert(split, data_frame):
    train_split, validation_split, test_split = split

    total_num_data = data_frame.shape[0]

    data_tensor = torch.tensor(data_frame)

    train_index = int(total_num_data * train_split)
    train_frame = data_tensor[:train_index]

    validation_index = int(total_num_data * validation_split) + train_index
    validation_frame = data_tensor[train_index:validation_index]

    test_frame = data_tensor[validation_index:]

    return train_frame, validation_frame, test_frame
