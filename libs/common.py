import torch, os
from torch.utils.data import DataLoader
from libs.model import PredictionNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
mse_loss = torch.nn.MSELoss()
model_dir = 'models'


def split_data(dataset, split):

    train_split, validation_split, _ = split
    num = len(dataset)

    train_index = int(num * train_split)
    valid_index = int(num * validation_split) + train_index

    train_data = dataset[:train_index]
    valid_data = dataset[train_index:valid_index]
    test_data = dataset[valid_index:]

    return train_data, valid_data, test_data


def trainer(net, dataloader, n_epochs, lr=1e-3, verbose=True, device=torch.device("cpu")):
    optimizer = torch.optim.Adam(params=list(net.parameters()), lr=lr)

    for j in range(n_epochs):
        
        net.train_flg = True

        L = len(dataloader)

        for i, data in enumerate(dataloader):

            sample = data['x'].to(device)
            label = data['y'].to(device)

            optimizer.zero_grad()
            loss = mse_loss(net(sample.float()), label.float())
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimizer.step()
            
            if verbose:
                print("Epoch {:d}".format(j) + ", {:.2f}".format(i/L*100) + '% completed. Loss:' + str(loss))



def validator(net, dataloader, verbose=True):

    net.train_flag = False
    acc_loss, num_samples = 0, 0

    for data in dataloader:
        
        sample = data['x']
        label = data['y']

        loss = mse_loss(net(sample.float()), label.float())

        acc_loss += loss.detach().numpy()
        num_samples += len(sample)

    if verbose:
        print("Validation loss is {:.2f} m^2 (Mean Squared Error)".format(acc_loss / num_samples))


def tester(net, dataloader):
    
    samples, label, prediction = [], [], []

    for data in dataloader:
        samples.append(data['x'].detach().numpy())
        label.append(data['y'].detach().numpy())
        prediction.append(net(data['x'].float()).detach().numpy())

    return samples, label, prediction


def get_policy(train_data, valid_data, test_data, data_dim, pred_horizon, batch_size, num_epochs, learning_rate, policy_name, retrain=False):

    print("Building dataloaders ...")
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if retrain | (not os.path.exists(os.path.join(model_dir, policy_name))):

        print("Creating neural network ...")
        net = PredictionNet(data_dim=data_dim, out_len=pred_horizon, batch_size=batch_size, device=device).to(device)

        print("Training ...")
        trainer(net, train_dataloader, num_epochs, verbose=True, lr=learning_rate, device=device)

        print("Training complete! Saving policy ...")
        torch.save(net.cpu().state_dict(), os.path.join(model_dir, policy_name), _use_new_zipfile_serialization=False, pickle_protocol=2)
        print("Policy saved!")
    
    print("Loading existing policy ...")
    net = PredictionNet(data_dim=data_dim, out_len=pred_horizon, batch_size=batch_size, device=cpu).to(cpu)
    net_state_dict= torch.load(os.path.join(model_dir, policy_name))
    net.load_state_dict(net_state_dict)
    print("Policy loaded!")

    print('Validating ...')
    validator(net, valid_dataloader)
    print('Validation complete!')

    print('Testing ...')
    samples, labels, predictions = tester(net, test_dataloader)
    print('Test complete! Figure is to be shown.')
    return samples, labels, predictions