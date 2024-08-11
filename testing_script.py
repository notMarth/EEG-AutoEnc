#imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import helper
import sys
#Load EEG data - preprocessed
import mne
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import importlib
if __name__ == "__main__":

	tf.config.run_functions_eagerly(True)

	#Subject numbers and experiment
	sub = "sub-28"
	exp = "fixthemix"

	fname = f"derivatives/eegprep/{sub}/{sub}_task-{exp}_eegprep.vhdr"

	raw = mne.io.read_raw_brainvision(fname, preload=True)
	events, event_dict = mne.events_from_annotations(raw)
	#start of songs in sample numbers
	song_starts = np.array(events)[events[:,2] == 10001][2:,0]
	press_starts = []
	press_starts = events[2:,0]

	#set sample rate
	sample_rate = 250
	#raw.plot()

	#Load FLAC Audio in
	import pyflac
	import scipy.io.wavfile as wav
	import scipy.signal as sig
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()

	aname = f"derivatives/audio/{sub}/{sub}_task-{exp}_aud.flac"
	decoder = pyflac.FileDecoder(aname, "temp.wav")
	aud_rate, audio = wav.read("temp.wav")
	dtype = audio.dtype


	#eeg_length = raw.get_data().shape[1]*sample_rate

	#Resample audio to EEG sample rate and get audio envelope
	audio = sig.resample(audio, raw.get_data().shape[1])
	audio = np.abs(sig.hilbert(audio.T))
	audio = np.average(audio, axis=0)


	#normalize audio
	audio_scaler = StandardScaler()
	audio = audio_scaler.fit_transform(audio.reshape(-1,1)).reshape(1, -1)


	audio = np.atleast_2d(audio)

	#SPLIT BY SEGMENT

	#Split audio and eeg up into their corresponding songs
	#note that the 1st elements are the whitespace before the first song starts
	scaler = StandardScaler()

	num_segments = 1000
	seconds = 10

	split_eeg = raw.get_data()

	#split data into 300 segments
	times = np.linspace(song_starts[1], raw.get_data().shape[1], num_segments, dtype = int)
	#each segment has a bound of "seconds" seconds before and after the segment
	split_audio, split_eeg = helper.split_events(audio, split_eeg, times, sample_rate, seconds)

	fs = sample_rate

	labels = [f'song{i}' for i in range(1, len(split_audio))]
	labels_train, labels_test = train_test_split(labels, train_size=0.7, test_size=0.3, random_state=5)
	labels_test, labels_val = train_test_split(labels_test, train_size=0.5, test_size=0.5, random_state=5)

	#X and Y are dictionaries so that the ordering of the corresponding segments can
	#be maintained
	X = {}
	Y = {}
	for i in range(1,len(split_audio)):
		X[f'song{i}'] = split_audio[i][0]

		Y[f'song{i}'] = split_eeg[i]

	size = Y['song3'].T.shape

	X_test, X_train, X_val, Y_test, Y_train, Y_val = [],[],[],[],[],[]

	for i in labels_train:
		if(Y[i].T.shape == size):
			Y_train.append(scaler.fit_transform(X[i].reshape(-1,1)).reshape(1,-1))
			X_train.append(scaler.fit_transform(Y[i].T))

	for i in labels_val:
		if(Y[i].T.shape == size):
			Y_val.append(scaler.fit_transform(X[i].reshape(-1,1)).reshape(1,-1))
			X_val.append(scaler.fit_transform(Y[i].T))

	for i in labels_test:
		if(Y[i].T.shape == size):
			Y_test.append(scaler.fit_transform(X[i].reshape(-1,1)).reshape(1,-1))
			X_test.append(scaler.fit_transform(Y[i].T))

	X_train = np.array(X_train)
	Y_train = np.array(Y_train)
	X_test = np.array(X_test)
	Y_test = np.array(Y_test)
	X_val = np.array(X_val)
	Y_val = np.array(Y_val)


latents = [10, 20, 30, 40, 50]
epochs = 100

test_ind = 0

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
input_shape = X_train.shape[1:]
output_shape = Y_train.shape[1:]
best_model_loss = []

for mod in sys.argv[1:]:
	#import autoencoder model here
	print(f"MODEL {mod}\n")
	model_lib = importlib.import_module(mod, '.')
	test_losses = []
	for latent in latents:
		print(f"LATENT SPACE {latent} FOR MODEL {mod}\n")
		model = model_lib.Autoencoder(latent, input_shape = input_shape, output_shape = output_shape)
		model.compile(optimizer="Adam", loss=losses.MeanSquaredError())


		history = model.fit(X_train, Y_train,
				epochs=epochs,
				validation_data=(X_val, Y_val),
				)

		plt.figure()
		plt.title(f"Model Loss for Epochs = {epochs} and Latent Space = {latent}")
		plt.plot(history.history["loss"], label="Training Loss")
		plt.plot(history.history["val_loss"], label="Validation Loss")
		plt.legend()
		plt.savefig(f"figs/Model_{mod}_{latent}_Loss.png")

		X_test = np.array(X_test)
		Y_test = np.array(Y_test)
		model_Y = model(X_test[test_ind].reshape(1, X_test[0].shape[0], X_test[0].shape[1])).numpy().reshape(1,-1)
		true_Y = Y_test[test_ind]

		display_scale = StandardScaler().fit(true_Y)
		display_Y = display_scale.transform(model_Y)

		pred, loss = model.predict(X_test, Y_test)
		test_losses.append(np.average(loss, axis=0)[0])

		fig, axs = plt.subplots(3,1)
		axs[0].plot(display_Y[0])
		axs[0].set_title(f"Reconstructed Audio Envelope (Latent Space = {latent})")
		axs[1].plot(true_Y[0])
		axs[1].set_title("True Audio Envelope")
		axs[2].plot(-1*display_Y[0], color="Red", label="Reconstructed (Flipped)")
		axs[2].plot(true_Y[0], label="True", alpha=0.5, color="Green")
		axs[2].set_title("True Audio Envelope vs Reconstructed")
		fig.legend()
		plt.savefig(f"figs/Model_{mod}_{latent}_Recon.png")

	plt.figure()
	plt.title(f"Average Model Loss Over Test Data")
	plt.scatter(latents, test_losses)
	plt.xlabel("Dimension of Latent Space")
	plt.ylabel("Average MSE")
	plt.savefig(f"figs/Model_{mod}_Test_Loss.png")

	test_losses.sort()
	best_model_loss.append(test_losses[-1])

plt.figure()
plt.title(f"Average Model Loss Over Test Data For Each Model (Best)")
plt.scatter(sys.argv[1:], best_model_loss)
plt.xlabel("Model Number")
plt.ylabel("Average MSE")
plt.savefig("figs/Model_Comparison.png")