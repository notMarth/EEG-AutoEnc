#imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.signal as sig
import etc.helper as helper
import numpy as np
from copy import deepcopy
from pathlib import Path
from mtrf.model import TRF
from mtrf.stats import neg_mse
from mtrf.stats import crossval

#model definition for Backwards Modeling (TRF approach)
class BackwardsModel:
    
    def __init__(self, regularization=5, train_size=0.7, test_size=0.3, epochs=100, random_state=5):
        
        #These are used during function calls
        
        #name is used in plotting as the name for the model
        self.name = "Backwards Modeling"
        self.train_size = train_size
        self.test_size = test_size
        self.epochs = epochs
        self.rand_state = random_state
        self.regularization = regularization
        
    def process_data(self, eeg, audio, sample_rate, mode, segments, seconds):
        #calculate audio envelope
        audio = np.abs(sig.hilbert(audio.T))
        audio = np.average(audio, axis=0)
        #normalize audio
        audio_scaler = StandardScaler()
        audio = audio_scaler.fit_transform(audio.reshape(-1,1)).reshape(1, -1)

        #split data into train, test, validation
        self.Y_train, self.X_train, self.Y_test, self.X_test, self.Y_val, self.X_val = \
        helper.train_test_val_split(eeg, audio, self.train_size, \
                            self.test_size, sample_rate, self.random_state, \
                            mode=mode, num_segments=segments, \
                            seconds=seconds)
        
    def train(self):
        X_train = list(self.X_train)
        X_test = list(self.X_test)
        X_val = list(self.X_val)

        Y_train = self.Y_train.reshape(self.Y_train.shape[0], self.Y_train.shape[-1])
        Y_test = self.Y_test.reshape(self.Y_test.shape[0], self.Y_test.shape[-1])
        Y_val = self.Y_val.reshape(self.Y_val.shape[0], self.Y_val.shape[-1])
        
        Y_train = list(Y_train)
        Y_test = list(Y_test)
        Y_val = list(Y_val)
        
        self.bwd_trf = TRF(direction=-1, metric=neg_mse)
        self.bwd_trf.train(stimulus=Y_train, response=X_train, fs=250, tmin=-200/1000, tmax=800/1000, regularization=self.regularization, seed=5)
    
    def plot_losses(self):
        res, loss = self.bwd_trf.predict(stimulus=Y_test[0], response=X_test[0])
        _, test_loss = self.bwd_trf.predict(stimulus=Y_test, response=X_test)
        
        true_Y = self.Y_test[0]
        display_scale = StandardScaler().fit(true_Y)
        pred = display_scale.transform(res[0])
        
        fig, axs = plt.subplots(3,1)
        axs[0].plot(pred)
        axs[0].set_title(f"Reconstructed Audio Envelope (Regularization=10^5)")
        axs[1].plot(true_Y)
        axs[1].set_title("True Audio Envelope")
        axs[2].plot(pred, color="Red", label="Reconstructed")
        axs[2].plot(true_Y, label="True", alpha=0.5, color="Green")
        axs[2].set_title(f"True Audio Envelope vs Reconstructed (loss={-1*loss})")
        fig.legend()
        
        Path(f"figs/back_TRF_10^5").mkdir(parents=True, exist_ok=True)

        plt.savefig(f"figs/back_TRF_10^20/back_TRF_10^5_Recon.png")