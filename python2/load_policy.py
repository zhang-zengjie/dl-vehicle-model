import torch, os, sys
import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)
from libs.model import PredictionNet

device = torch.device("cpu")

hist = np.zeros((1, 1, 50))
hist[0, 0, -1] = 1

net = PredictionNet(data_dim=1, out_len=1, batch_size=30, device=device).to(device)
net_state_dict = torch.load(os.path.join('models', 'net_state_dict.pth'))
net.load_state_dict(net_state_dict)

out = net(torch.tensor(hist).float())[0, 0, 0].detach().numpy()

print(out)