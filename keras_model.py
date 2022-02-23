import tensorflow as tf

def get_model(in_features=784, out_features=10, num_layers=3):
	layers = [tf.keras.layers.Input(in_features)]
	layers_n = [in_features]+[out_features+i*(in_features - out_features)//num_layers for i in range(num_layers)][::-1]
	for i in range(1, num_layers):
		layers.append(tf.keras.layers.Dense(layers_n[i], activation="relu"))
	layers.append(tf.keras.layers.Dense(out_features, activation="softmax"))
	return tf.keras.Sequential(layers)