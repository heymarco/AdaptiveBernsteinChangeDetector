import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class AutoEncoder(nn.Module):

    def __init__(self, input_size: int, eta: float):
        super(AutoEncoder, self).__init__()
        self.eta = eta
        self.input_size = input_size
        self.bottleneck_size = int(eta * input_size)
        self.encoder = nn.Linear(in_features=self.input_size, out_features=self.bottleneck_size)
        self.decoder = nn.Linear(in_features=self.bottleneck_size, out_features=self.input_size)
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))
        return x

    def update(self, window, epochs: int = 1):
        if len(window) == 0:
            return
        self.train()
        tensor = torch.from_numpy(window).float()
        for ep in range(epochs):
            self.optimizer.zero_grad()
            pred = self.forward(tensor)
            loss = F.mse_loss(pred, tensor)
            loss.backward()
            self.optimizer.step()

    def new_tuple(self, x):
        tensor = torch.from_numpy(x).float()
        self.eval()
        with torch.no_grad():
            pred = self.forward(tensor)
            loss = F.mse_loss(pred, tensor)
            return np.sqrt(loss.item()), pred.numpy()[0], x

