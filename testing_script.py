#imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import scipy.signal as sig
import helper
import sys
import numpy as np
import importlib
from copy import deepcopy
from pathlib import Path

""" Run user_provided models with selected EEG and audio data. Generate plots
based on results.


"""

#options
ENVELOPE_OPTIONS = ['env', 'envelope']
SPECTRO_OPTIONS = ['spec', 'spectrogram']
MODE = "segments"
NUM_SEGMENTS = 1000
SECONDS = 10
RANDOM_STATE = 5
TRAIN_SIZE = 0.7
TEST_SIZE = 0.3
SPEC=100
LATENT_SPACES = [20]
EPOCHS = 50

if __name__ == "__main__":

	#run if user doesn't provide arguments
	if len(sys.argv) == 1:
		print("USAGE: testing_script.py model_name [other_model_name ...] epochs")
		sys.exit()

	#tf.config.run_functions_eagerly(True)

	#Subject numbers and experiment
	sub = "sub-28"
	exp = "fixthemix"
	eeg_file = f"eegprep\{sub}\{sub}_task-{exp}_eegprep.vhdr"
	audio_file = f'audio\{sub}\{sub}_task-{exp}_aud.flac'
	#set sample rate
	sample_rate = 250

	#Load data in
	raw = helper.load_eeg(eeg_file, sample_rate)
	eeg = raw.get_data()
	audio = helper.load_audio(audio_file, sample_rate, eeg.shape[1])
	

	#latent space dimensions to try
	latents = LATENT_SPACES
	EPOCHS = int(sys.argv[-1])

	#test data index to display reconstruction for
	test_ind = 0
	best_model_loss = []

	for mod in sys.argv[1:-1]:
		#import autoencoder model here
		print(f"MODEL {mod}\n")
		model_lib = importlib.import_module(mod, '.')
		test_losses = []

		for latent in latents:
			print(f"LATENT SPACE {latent} FOR MODEL {mod}\n")
			model = model_lib.Autoencoder(latent, TRAIN_SIZE, TEST_SIZE, EPOCHS, RANDOM_STATE)
			Path(f"figs/{model.name}").mkdir(parents=True, exist_ok=True)

			model.process_data(deepcopy(eeg), deepcopy(audio), sample_rate, MODE, NUM_SEGMENTS, SECONDS)
			model.train()
			model.plot_losses()

			test_losses.append(model.test_loss)

			model.visualize_activations()
			

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
	plt.scatter(sys.argv[1:-1], best_model_loss)
	plt.xlabel("Model Number")
	plt.ylabel("Average MSE")
	plt.savefig("figs/Model_Comparison.png")