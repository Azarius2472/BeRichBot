import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from tqdm import tqdm


class TsCNNDataset(Dataset):
    def __init__(self, raw_data_df, window_size):
        self.raw_data_df = raw_data_df[['open', 'high', 'low', 'volume', 'close']]  # , 'unix_time']]
        self.window_size = window_size

    def __len__(self):
        return self.raw_data_df.shape[0] - self.window_size

    def __getitem__(self, idx):
        start = idx
        end = idx + self.window_size
        # return self.raw_data_df.iloc[start:end].values, self.raw_data_df.iloc[end]['close']
        return torch.tensor(np.swapaxes(self.raw_data_df.iloc[start:end].diff().dropna().values, 0, 1)), torch.tensor(
            self.raw_data_df.iloc[end].values), torch.tensor(self.raw_data_df.iloc[end - 1].values)
        # return [self.raw_data_df.iloc[start:end]['open'].values, self.raw_data_df.iloc[start:end]['high'].values, self.raw_data_df.iloc[start:end]['low'].values, self.raw_dat


class TsLSTMDataset(Dataset):
    def __init__(self, raw_data_df, window_size):
        self.raw_data_df = raw_data_df[['open', 'high', 'low', 'volume', 'close']]
        self.window_size = window_size

    def __len__(self):
        return self.raw_data_df.shape[0] - self.window_size

    def __getitem__(self, idx):
        start = idx
        end = idx + self.window_size
        return self.raw_data_df.iloc[start:end].diff().dropna().values, self.raw_data_df.iloc[end].values, \
               self.raw_data_df.iloc[end - 1].values


class tsLSTM(nn.Module):

    def __init__(self, n_features, n_hidden=128, n_layers=2):
        super().__init__()

        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=n_layers  # ,
            #             dropout=0.2
        )

        self.regressor = nn.Linear(n_hidden, 5)

    def forward(self, x):
        #         self.lstm.flattern_parameters()

        _, (hidden, _) = self.lstm(x)

        return self.regressor(hidden[-1])


class tsCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(5, 15, 6)
        self.pool = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(15, 30, 6)
        self.conv3 = nn.Conv1d(30, 100, 3)
        #         self.conv4 = nn.Conv1d(30, 40, 3)
        self.fc1 = nn.Linear(200, 60)
        self.fc2 = nn.Linear(60, 10)
        self.fc3 = nn.Linear(10, 5)

    def forward(self, x):
        # bias = x[:-1:-1]
        # x = torch.diff(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        #         x = self.pool(x)

        #         x = self.conv4(x)
        #         x = F.relu(x)
        #         x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x  # + bias

        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x


class TSPredictor():

    def __init__(self, needLoad=False, path_to_model=None, model_type=None, window_size=60):
        self.window_size = window_size
        self.isModelrained = False
        self.model_type = model_type
        self.y_true = None
        self.y_pred = None

        self.max_plt = None
        self.min_plt = None

        if needLoad:

            if model_type == 'CNN':
                self.model = tsCNN()

            if model_type == 'LSTM':
                self.model = tsLSTM(n_features=5)

            self.model.load_state_dict(torch.load(path_to_model))
            self.model.eval()
            self.isModelrained = True

        else:
            if model_type == 'CNN':
                self.model = tsCNN()
            if model_type == 'LSTM':
                self.model = tsLSTM(n_features=5)

    def train(self, num_epochs=5, learning_rate=0.01, data_df=None, test_split=0.15, BATCH_SIZE=512):

        test_start_idx = data_df.shape[0] - int(data_df.shape[0] * test_split)
        self.min_plt = data_df.close.min()
        self.max_plt = data_df.close.max()

        if self.model_type == 'CNN':
            dataset = TsCNNDataset(data_df, self.window_size)
        if self.model_type == 'LSTM':
            dataset = TsLSTMDataset(data_df, self.window_size)

        train_split = Subset(dataset, range(test_start_idx))
        test_split = Subset(dataset, range(test_start_idx, len(dataset)))
        train_batches = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=False)
        test_batches = DataLoader(test_split, batch_size=BATCH_SIZE, shuffle=False)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.to('cuda');

        for epoch in range(num_epochs):
            self.y_true = data_df.close.values.copy()
            self.y_pred = data_df.close.values.copy()
            i = self.window_size
            tepoch = tqdm(train_batches, unit="batch")
            self.model.train()
            for inputs, targets, bias in tepoch:
                # break
                inputs = inputs.to('cuda')
                targets = targets.to('cuda')
                #                 bias = torch.unsqueeze(bias, 1)
                bias = bias.to('cuda')
                optimizer.zero_grad()
                pred = self.model(inputs)

                pred = pred + bias
                loss = criterion(pred, targets)

                pred = pred.detach().cpu().numpy()
                # assert False
                self.y_pred[i:i + pred.shape[0]] = pred[:, -1].flatten().copy()

                i += pred.shape[0]
                # assert False

                tepoch.set_description("loss: « %s »" % str(loss.detach().cpu().numpy()))
                loss.backward()
                optimizer.step()

            tepoch = tqdm(test_batches, unit="batch")
            self.model.eval()
            optimizer.zero_grad()
            for inputs, targets, bias in tepoch:
                inputs = inputs.to('cuda')
                targets = targets.to('cuda')
                #                 bias = torch.unsqueeze(bias, 1)
                bias = bias.to('cuda')
                pred = self.model(inputs)
                pred = pred + bias

                pred = pred.detach().cpu().numpy()
                self.y_pred[i:i + pred.shape[0]] = pred[:, -1].flatten().copy()
                i += pred.shape[0]

            plt.axvline(x=test_start_idx, c='r', linestyle='--')
            plt.ylim(self.min_plt, self.max_plt)

            plt.plot(self.y_pred, color='orange')
            plt.plot(self.y_true, color='blue')

            plt.suptitle('Time-Series Prediction')
            plt.show()

        self.model.eval()
        self.model.to('cpu')
        self.isModelrained = True

    def save_model(self, path):
        self.model.eval()
        self.model.to('cpu')
        torch.save(self.model.state_dict(), path)
        self.isModelrained = True

    def inference_df(self, df, device_str):
        device = torch.device('cuda:0') if device_str == 'gpu' else torch.device('cpu')
        self.model.to(device)

        if not self.isModelrained:
            return None
        else:
            if df.shape[0] != self.window_size:
                return None
            else:
                bias = df.values[-1]
                data = df[['open', 'high', 'low', 'volume', 'close']].diff().dropna()
                vals = data.values

                if self.model_type == 'CNN':
                    vals = np.swapaxes(vals, 0, 1)
                vals = np.expand_dims(vals, axis=0)

                pred = self.model(torch.tensor(vals).to(device)).detach().cpu().numpy()[0]
                return pred + bias

    def inference_far_period(self, df, ticks, device_str):
        new_df = df.copy()

        predicted_values = []
        for i in tqdm(range(ticks)):
            pred = self.inference_df(new_df, device_str)
            predicted_values.append(pred[0])
            new_df = pd.concat([new_df[1:], pd.DataFrame.from_dict({'open': [pred[0]],
                                                                    'high': [pred[1]],
                                                                    'low': [pred[2]],
                                                                    'volume': [pred[3]],
                                                                    'close': [pred[4]]})], ignore_index=True)

        predicted_values = np.asarray(predicted_values)
        return np.mean(predicted_values.reshape(-1, 60), axis=1)
