from ReportDisc import ReportDisc, TorchReportDisc, TFReportDisc
import torch
from torch_model import FullyConnectedModel
from keras_model import get_model
import tensorflow as tf
from dataloader import MNIST_Dataset
import torch
from tqdm import tqdm
import pandas as pd

def train(webhook_url, epochs=5, mode="torch", embed=False):
	if mode=="torch":
		dataset = MNIST_Dataset("./sample_data/mnist_train.csv", mode)
		train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
		trd = TorchReportDisc(webhook_url, embed_reports=embed)
		trd.report("Starting PyTorch...")
		criterion = torch.nn.CrossEntropyLoss()
		model = FullyConnectedModel()
		optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)
		for epoch in range(epochs):
			running_loss, running_acc = 0, 0
			bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
			for batch_idx, (batch_x, batch_y) in bar:
				optimizer.zero_grad()
				pred_y = model(batch_x)
				loss = criterion(pred_y, batch_y)
				loss.backward()
				optimizer.step()
				running_loss += loss.item()
				pred = torch.argmax(pred_y, axis=1)
				acc = torch.mean((pred==batch_y).float())
				running_acc += acc.item()
				bar.set_description("TRAINING:- "+str({"epoch": epoch+1, "loss": round(running_loss/(batch_idx+1), 4), "acc": round(running_acc/(batch_idx+1), 4)}))
			bar.close()
			trd.report_stats({"loss": running_loss/(batch_idx+1), "acc": running_acc/(batch_idx+1)})
	if mode=="tf":
		data = pd.read_csv("./sample_data/mnist_train.csv")
		data = data.values
		y, x = data.T[0], data.T[1:].T
		model = get_model()
		reporter_callback = TFReportDisc(webhook_url, embed_reports=embed)
		reporter_callback.reporter.report("Starting TensorFlow...")
		model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
		model.fit(x, y, batch_size=64, epochs=5, validation_split=0.2, callbacks=[reporter_callback])
	else:
		rd = ReportDisc(webhook_url, embed_reports=embed)
		array = ["Can deal with:", "Loops", "PyTorch", "Keras"]
		rd.report("Iterating through: "+str(array))
		for index, item in enumerate(array):
			rd.report_stats({"index": index, "item": item})
		
if __name__ == '__main__':
	with open(".secrets", "r") as f:
		webhook_url = f.read()
	train(webhook_url, mode="tf")
	train(webhook_url, mode="torch")
	train(webhook_url, mode="list")

	train(webhook_url, mode="tf", embed=True)
	train(webhook_url, mode="torch", embed=True)
	train(webhook_url, mode="list", embed=True)