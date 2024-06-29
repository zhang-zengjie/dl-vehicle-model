import torch, os
torch.manual_seed(42)


def train_model(mode):

    if mode == 'predict_trajectory':

        from libs.virtual import data_generate
        train_data, valid_data, test_data = data_generate(history_len=30,
                                                          pred_horizon=20,
                                                          split_ratio=(0.6, 0.2, 0.2))
        
        samples, labels, predictions = get_policy(train_data, valid_data, test_data, 
                                       data_dim=2,
                                       pred_horizon=20, 
                                       batch_size=30, 
                                       num_epochs=30, 
                                       learning_rate=1e-2, 
                                       policy_name='policy.pth', 
                                       retrain=False)
        from libs.visualize import visualize_test_2D_trajectory
        row, column = 10, 10
        visualize_test_2D_trajectory(samples[row][column], labels[row][column], predictions[row][column])
        
    elif mode == 'fit_model':

        from libs.real import data_generate
        train_data, valid_data, test_data = data_generate(history_len=50,
                                                          pred_horizon=1,
                                                          split_ratio=(0.6, 0.2, 0.2),
                                                          data_dir=os.getcwd(),
                                                          visualize=False)
        
        _, labels, predictions = get_policy(train_data, valid_data, test_data, 
                                       data_dim=1,
                                       pred_horizon=1, 
                                       batch_size=30, 
                                       num_epochs=30, 
                                       learning_rate=1e-5, 
                                       policy_name='net_state_dict.pth', 
                                       retrain=False)
        
        from libs.visualize import visualize_test_1D_distribution
        visualize_test_1D_distribution(labels, predictions)

    else:

        print("Invalid mode! Please specify between 'predict_trajectory' and 'fit_model'.")


if __name__ == "__main__":

    from libs.common import get_policy
    mode = 'fit_model'
    ## Choose in which mode the vehicle model is to be generated:
    ## 1. 'fit_model': generating a dynamic model for the vehicle
    ## 2. 'predict_trajectory': predict the motion of the vehicle given historical trajectories
    ## Otherwise, the program gives an error information

    train_model(mode)
    