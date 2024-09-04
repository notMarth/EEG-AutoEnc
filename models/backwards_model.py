#imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.signal as sig
import helper
import numpy as np
from copy import deepcopy
from pathlib import Path


from mtrf.model import TRF
from mtrf.stats import neg_mse
from mtrf.stats import crossval

REGULARIZATIONS = [10**5]

if __name__ == "__main__":

    #run if user doesn't provide arguments
    # if len(sys.argv) == 1:
    # 	print("USAGE: testing_script.py model_name [other_model_name ...] epochs")
    # 	sys.exit()

    #tf.config.run_functions_eagerly(True)

    #Subject numbers and experiment
    sub = "sub-28"
    exp = "fixthemix"
    eeg_file = f"../eegprep/{sub}/{sub}_task-{exp}_eegprep.vhdr"
    audio_file = f'../audio/{sub}/{sub}_task-{exp}_aud.flac'
    #set sample rate
    sample_rate = 250

    #Load data in
    raw = helper.load_eeg(eeg_file, sample_rate)
    eeg = raw.get_data()
    audio = helper.load_audio(audio_file, sample_rate, eeg.shape[1])
    
    #latent space dimensions to try
    #EPOCHS = int(sys.argv[-1])

    #test data index to display reconstruction for
    test_ind = 0
    losses = []
        
    for lam in REGULARIZATIONS:
        print(lam)
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
        bwd_trf.train(stimulus=Y_train, response=X_train, fs=250, tmin=-200/1000, tmax=800/1000, regularization=lam, seed=5)
        
        
        print(X_test[0].shape, Y_test[0].shape)
        
        res, loss = bwd_trf.predict(stimulus=Y_test[0], response=X_test[0])
        _, test_loss = bwd_trf.predict(stimulus=Y_test, response=X_test)
        
        true_Y = Y_test[0]
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
        
        losses.append(-1*test_loss)
        
    plt.figure()
    plt.title(f"Average Loss Over Test Data Per Regularization Parameter")
    plt.scatter(["10^5"], losses)
    plt.xlabel("Dimension of Latent Space")
    plt.ylabel("Average MSE")
    plt.savefig(f"figs/back_TRF_test_Loss.png")
    