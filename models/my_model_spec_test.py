import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
import numpy as np
import etc.helper as helper
import scipy.signal as sig
from matplotlib import pyplot as plt
from IPython import display
import soundfile as sf

SPEC = 100

#Autoencoder model definition
class Autoencoder(Model):
    def __init__(self, latent_dim, train_size=0.7, test_size=0.3, epochs=100, random_state=5):
        super(Autoencoder, self).__init__()
        self.model_name = "custom_spectral_model_test"

        self.latent_dim = latent_dim
        self.in_shape = ((0,0,0))
        self.out_shape = ((0,0,0))

        self.train_size = train_size
        self.test_size = test_size
        self.epochs = epochs
        self.random_state = random_state
        

    def call(self, x):
        encoded_real = self.encoder_real(x)
        encoded_imag = self.encoder_imag(x)
        decoded_real = self.decoder_real(encoded_real)
        decoded_imag = self.decoder_imag(encoded_imag)
        return [decoded_real, decoded_imag]
    
    def predict(self, data, labels):
        reconstructions = self(data)
        loss = losses.mse(reconstructions, labels)
        return reconstructions, loss
    
    def plot_losses(self):
        test_ind = 0
        plt.figure()
        plt.title(f"Model Loss for Epochs = {self.epochs} and Latent Space = {self.latent_dim}")
        plt.plot(self.history.history["loss"], label="Training Loss")
        plt.plot(self.history.history["val_loss"], label="Validation Loss")
        plt.legend()
        plt.savefig(f"figs/{self.model_name}/Model_{self.model_name}_{self.latent_dim}_Loss.png", dpi=300)

        X_test = np.array(self.X_test)
        Y_test = np.array(self.Y_test)
        model_Y1 = np.array(self(self.X_test[test_ind].reshape(1, self.X_test[0].shape[0], self.X_test[0].shape[1]))).reshape(2, self.Y_test.shape[1], self.Y_test.shape[2])
        model_Y = np.abs(model_Y1[0] + (1.j * model_Y1[1]))
        print("model_y", (model_Y1[0] + (1.j * model_Y1[1])).shape)
        modelYwave = self.spectro.istft(model_Y1[0] + (1.j * model_Y1[1]))
        true_Y = np.abs(Y_test[test_ind])
        true_wave = self.spectro.istft(Y_test[test_ind])

        display_scale = StandardScaler().fit(true_Y)
        display_Y = display_scale.transform(model_Y)
        #true_Y = display_scale.transform(true_Y)
        display_Y = np.abs(model_Y)
        _, loss = self.predict(self.X_test, [np.real(self.Y_test), np.imag(self.Y_test)])

        plt.plot(true_wave)
        plt.savefig(f"figs/{self.model_name}/Model_{self.model_name}_{self.latent_dim}_realwave.png")
        
        sf.write('ogoutput.flac', true_wave, 1600)

        plt.plot(modelYwave)
        plt.savefig(f"figs/{self.model_name}/Model_{self.model_name}_{self.latent_dim}_wave.png")
        fig, axs = plt.subplots(2,1)

        sf.write('output.flac', modelYwave, 1600)

        
        # axs[0].imshow(modelYwave, origin='lower', aspect='auto',
        #     extent=self.spectro.extent(self.out_shape[0]), cmap='inferno')
        # axs[0].set_title(f"Reconstructed Audio Spectrogram (Latent Space = {self.latent_dim})")
        
        # axs[1].imshow(true_Y, origin='lower', aspect='auto',
        #     extent=self.spectro.extent(self.out_shape[0]), cmap='inferno')		
        # axs[1].set_title("True Audio Spectrogram")
       
        fig.legend()
        
        plt.savefig(f"figs/{self.model_name}/Model_{self.model_name}_{self.latent_dim}_Recon.png", dpi=300)

        # plt.figure()
        # plt.plot(display_Y)
        # plt.savefig(f"figs/{self.model_name}/Model_{self.model_name}_{self.latent_dim}_ReconWave.png")
        print("display_Y: \n", display_Y)


        # Convert the waveform to a spectrogram via a STFT.
        spectrogram = tf.signal.stft(
            true_Y, frame_length=255, frame_step=128)
        # Obtain the magnitude of the STFT.
        spectrogram = tf.abs(spectrogram)
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram = spectrogram[..., tf.newaxis]
        print('Spectrogram shape:', spectrogram.shape)
        
        self.test_loss =  np.average(loss)
        print(loss)
        print(self.test_loss)


    def process_data(self, eeg, audio, sample_rate, mode, segments, seconds):

        #calculate spectrogram of average of the two audio channels
        audio = np.atleast_2d(np.average(audio.T, axis=0))
        audio_scaler = StandardScaler()
        #plt.figure()
        # plt.subplot(1,1,1)
        #plt.plot(audio)
        
       
        #plt.plot(true_Y)
        #plt.savefig(f"figs/{self.model_name}/Model_{self.model_name}_{self.latent_dim}_wave.png", dpi=300)


        #split data into train, test, validation
        self.X_train, self.Y_train, self.X_test, self.Y_test, self.X_val, self.Y_val = \
        helper.train_test_val_split(eeg, audio, self.train_size, \
                            self.test_size, sample_rate, self.random_state, \
                            mode=mode, num_segments=segments, \
                            seconds=seconds)

        self.Y_train = self.Y_train[:,0,:]
        self.Y_test = self.Y_test[:,0,:]
        self.Y_val = self.Y_val[:,0,:]

        window = sig.windows.gaussian(30, std=5, sym=True)
        spectro = sig.ShortTimeFFT(win=window, hop=19, fs=sample_rate, scale_to='magnitude')
        #waveform = spectro.istft()
        self.Y_train = spectro.stft(self.Y_train)
        
        self.Y_test = spectro.stft(self.Y_test)
        
        self.Y_val = spectro.stft(self.Y_val)

        self.spectro = spectro


    def train(self):
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2], 1)
        self.in_shape = self.X_train.shape[1:]
        self.out_shape = self.Y_train.shape[1:]

        self.Y_train_real = np.real(self.Y_train)
        self.Y_train_imag = np.imag(self.Y_train)

        self.Y_val_real = np.real(self.Y_val)
        self.Y_val_imag = np.imag(self.Y_val)

        self.encoder_real = tf.keras.Sequential([
            layers.Input(shape=self.in_shape),
            layers.Conv2D(16, (3,3), strides=2, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(8, (3,3), strides=2, padding='same', activation='relu'),\
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(4, (3,3), strides=2, padding='same', activation='relu'),
            #layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(2, (3,3), strides=2, padding='same', activation='relu'),
            #layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(self.latent_dim),
        ])

        self.encoder_imag = tf.keras.Sequential([
            layers.Input(shape=self.in_shape),
            layers.Conv2D(16, (3,3), strides=2, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(8, (3,3), strides=2, padding='same', activation='relu'),\
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(4, (3,3), strides=2, padding='same', activation='relu'),
            #layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(2, (3,3), strides=2, padding='same', activation='relu'),
            #layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(self.latent_dim),
        ])

        self.decoder_real = tf.keras.Sequential([
            layers.Dense(2*3),
            layers.Dropout(0.25),
            layers.Reshape((3, 2, 1)),
            layers.Conv2DTranspose(5, 3, strides=2, padding='same', activation='sigmoid'),
            layers.Dropout(0.25),
            #layers.Reshape((14, 3, 1)),
            layers.Conv2DTranspose(7, 3, strides=2, padding='same', activation='sigmoid'),
            layers.Dropout(0.25),
            layers.Conv2DTranspose(11, 3, strides=2, padding='same', activation='sigmoid'),
            #layers.Conv2D(7, (3,3), strides=2, padding='same', activation='sigmoid'),
            layers.Dropout(0.25),
            #layers.Reshape((self.out_shape[0], self.out_shape[1]*2)),
            #layers.MaxPooling1D(),
            #layers.Dense(self.out_shape[1]),
            layers.Reshape(self.out_shape)
        ])

        self.decoder_imag = tf.keras.Sequential([
            layers.Dense(2*3),
            layers.Dropout(0.25),
            layers.Reshape((3, 2, 1)),
            layers.Conv2DTranspose(5, 3, strides=2, padding='same', activation='sigmoid'),
            layers.Dropout(0.25),
            #layers.Reshape((14, 3, 1)),
            layers.Conv2DTranspose(7, 3, strides=2, padding='same', activation='sigmoid'),
            layers.Dropout(0.25),
            layers.Conv2DTranspose(11, 3, strides=2, padding='same', activation='sigmoid'),
            #layers.Conv2D(7, (3,3), strides=2, padding='same', activation='sigmoid'),
            layers.Dropout(0.25),
            #layers.Reshape((self.out_shape[0], self.out_shape[1]*2)),
            #layers.MaxPooling1D(),
            #layers.Dense(self.out_shape[1]),
            layers.Reshape(self.out_shape)
        ])

        self.compile(optimizer="Adam", loss=losses.MeanSquaredError())


        self.history = self.fit(self.X_train, [self.Y_train_real, self.Y_train_imag],
                epochs=self.epochs,
                validation_data=(self.X_val, [self.Y_val_real, self.Y_val_imag]),
                )

#python testing_script.py my_model_spec_test 100