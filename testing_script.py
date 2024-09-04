#imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.signal as sig
import etc.helper as helper
import sys
import numpy as np
import importlib
from copy import deepcopy
from pathlib import Path

from mtrf.model import TRF
from mtrf.stats import neg_mse

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
LATENT_SPACES = [5, 10, 15, 20, 25, 30]
EPOCHS = 100

if __name__ == "__main__":

    #run if user doesn't provide arguments
    if len(sys.argv) == 1:
        print("USAGE: testing_script.py model_name [other_model_name ...] epochs")
        sys.exit()

    #tf.config.run_functions_eagerly(True)

    #Subject numbers and experiment
    sub = "sub-28"
    exp = "fixthemix"
    eeg_file = f"data/eegprep{sub}/{sub}_task-{exp}_eegprep.vhdr"
    audio_file = f'data/audio/{sub}/{sub}_task-{exp}_aud.flac'
    #set sample rate
    sample_rate = 250

    #Load data in
    raw = helper.load_eeg(eeg_file, sample_rate)
    eeg = raw.get_data()
    audio = helper.load_audio(audio_file, sample_rate, eeg.shape[1])
    
    plt.gcf().set_size_inches(10, 5)

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

            #model.visualize_activations()
            
        try:
            plt.figure()
            plt.title(f"Average Model Loss Over Test Data")
            print(latents, test_losses)
            plt.scatter(latents, test_losses)
            plt.xlabel("Dimension of Latent Space")
            plt.ylabel("Average MSE")
            plt.savefig(f"figs/Model_{mod}_Test_Loss.png")
        except:
            print(latents, test_losses)

        test_losses.sort()
        best_model_loss.append(test_losses[-1])
 
 
    #Load data in
    raw = helper.load_eeg(eeg_file, sample_rate)
    eeg = raw.get_data()
    audio = helper.load_audio(audio_file, sample_rate, eeg.shape[1])
    
    audio = np.abs(sig.hilbert(audio.T))
    audio = np.average(audio, axis=0)
    #normalize audio
    audio_scaler = StandardScaler()
    audio = audio_scaler.fit_transform(audio.reshape(-1,1)).reshape(1, -1)

    #split data into train, test, validation
    X_train, Y_train, X_test, Y_test, X_val, Y_val = \
    helper.train_test_val_split(eeg, audio, 0.7, \
                        0.3, sample_rate, 5, \
                        mode="segments", num_segments=1000, \
                        seconds=10)
    
    # X_train.extend(X_test)
    # X_test.extend(X_val)
    
    # Y_train.extend(Y_test)
    # Y_test.extend(Y_val)

    X_train = list(X_train)
    X_test = list(X_test)
    X_val = list(X_val)

    Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[-1])
    Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[-1])
    Y_val = Y_val.reshape(Y_val.shape[0], Y_val.shape[-1])
    
    Y_train = list(Y_train)
    Y_test = list(Y_test)
    Y_val = list(Y_val)
    
    
    bwd_trf = TRF(direction=-1, metric=neg_mse)
    bwd_trf.train(stimulus=Y_train, response=X_train, fs=250, tmin=-200/1000, tmax=800/1000, regularization=10**20, seed=5)
    
    
    print(X_test[0].shape, Y_test[0].shape)
    
    res, loss = bwd_trf.predict(stimulus=Y_test[0], response=X_test[0])
    _, test_loss = bwd_trf.predict(stimulus=Y_test, response=X_test)
    
    true_Y = Y_test[0]
    display_scale = StandardScaler.fit(true_Y)
    pred = display_scale.tranform(res[0])
    
    fig, axs = plt.subplots(3,1)
    axs[0].plot(pred)
    axs[0].set_title(f"Reconstructed Audio Envelope (Regularization=10^20)")
    axs[1].plot(true_Y)
    axs[1].set_title("True Audio Envelope")
    axs[2].plot(pred, color="Red", label="Reconstructed")
    axs[2].plot(true_Y, label="True", alpha=0.5, color="Green")
    axs[2].set_title(f"True Audio Envelope vs Reconstructed (loss={-1*loss})")
    fig.legend()
    
    Path(f"figs/back_TRF_1000").mkdir(parents=True, exist_ok=True)

    plt.savefig(f"figs/back_TRF_1000/back_TRF_1000_Recon.png")
    
    best_model_loss.append(-1*test_loss)
    
    plt.show()
    
    labels = sys.argv[1:-1]
    labels.append("BackwardsTRF")
    plt.figure()
    plt.title(f"Average Model Loss Over Test Data For Each Model (Best)")
    plt.scatter(labels, best_model_loss)
    plt.xlabel("Model Number")
    plt.ylabel("Average MSE")
    plt.savefig("figs/Model_Comparison.png")
    
    plt.show()