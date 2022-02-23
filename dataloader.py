import torch
import pandas as pd
import numpy as np

class MNIST_Dataset(torch.utils.data.Dataset):
	def __init__(self, path, mode):
		self.mode = mode
		self.path = path
		self.data = self.read_data()

	def read_data(self):
		df = pd.read_csv(self.path)
		df = df.values
		return df

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return torch.tensor(self.data[idx][1:]).float(), torch.tensor(self.data[idx][0])