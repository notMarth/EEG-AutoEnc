import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
import numpy as np
import etc.helper as helper
import scipy.signal as sig
from matplotlib import pyplot as plt

#Autoencoder model definition
class Autoencoder(Model):
    def __init__(self, latent_dim, train_size=0.7, test_size=0.3, epochs=100, random_state=5):
        super(Autoencoder, self).__init__()
        self.model_name = "custom_model_backwards"

        self.latent_dim = latent_dim
        self.in_shape = None
        self.out_shape = None

        self.train_size = train_size
        self.test_size = test_size
        self.epochs = epochs
        self.random_state = random_state


    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def predict(self, data, labels):
        reconstructions = self(data).numpy()
        loss = losses.mse(reconstructions, labels)
        return reconstructions, loss
    
    def plot_losses(self):
        plt.figure()
        plt.title(f"Model Loss for Epochs = {self.epochs} and Latent Space = {self.latent_dim}")
        plt.plot(self.history.history["loss"], label="Training Loss")
        plt.plot(self.history.history["val_loss"], label="Validation Loss")
        plt.legend()
        plt.savefig(f"figs/{self.model_name}/Model_{self.model_name}_{self.latent_dim}_Loss.png", dpi=300)

        self.X_test = np.array(self.X_test)
        self.Y_test = np.array(self.Y_test)
        model_Y = self(np.expand_dims(self.X_test[0],0)).numpy().reshape((5000, 31))
        true_Y = self.Y_test[0].reshape((5000, 31))

        display_scale = StandardScaler().fit(true_Y)
        display_Y = display_scale.transform(model_Y)
        #true_Y = display_scale.transform(true_Y)
        _, loss = self.predict(self.X_test, self.Y_test)

        for i in range(4):
            for j in range(8):
                plt.plot(display_Y[:,i+j], label="Reconstructed", color='red')
                plt.plot(true_Y[:,i+j], label="True", color='blue', alpha=0.5)
        
                plt.legend()
                plt.savefig(f"figs/{self.model_name}/Model_{self.model_name}_{self.latent_dim}_Recon.png", dpi=300)
                plt.figure()

        self.test_loss = np.average(loss, axis=0)[0]


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
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[-1], 1)
        self.X_val = self.X_val.reshape(self.X_val.shape[0], self.X_val.shape[-1], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[-1], 1)
        self.in_shape = self.X_train.shape[1:]
        self.out_shape = self.Y_train.shape[1:]

        self.decoder = tf.keras.Sequential([
            layers.Reshape((1, self.latent_dim)),
            layers.Conv1DTranspose(4, 3, strides=2, padding='same', activation='relu'),
            layers.Dropout(0.25),
            layers.Reshape((1, 2, 4)),
            layers.Conv2DTranspose(8, (3,3), strides=2, padding='same', activation='relu'),\
            layers.Dropout(0.25),
            layers.Conv2DTranspose(16, (3,3), strides=5, padding='same', activation='relu'),
            #layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2DTranspose(31, (3,3), strides=5, padding='same', activation='relu'),
            #layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Reshape(self.out_shape)
        ])

        self.encoder = tf.keras.Sequential([
            layers.Conv1D(self.in_shape[-2]*3//4, 3, strides=2, padding='same', activation='sigmoid'),
            layers.MaxPooling1D(),
            layers.Dropout(0.25),
            layers.Conv1D(self.in_shape[-2]//2, 3, strides=2, padding='same', activation='sigmoid'),
            layers.MaxPooling1D(),
            layers.Dropout(0.25),
            layers.Conv1D(self.in_shape[-2]//4, 3, strides=2, padding='same', activation='sigmoid'),
            layers.MaxPooling1D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(self.latent_dim),
        ])

        self.compile(optimizer="Adam", loss=losses.MeanSquaredError())


        self.history = self.fit(self.X_train, self.Y_train,
                epochs=self.epochs,
                validation_data=(self.X_val, self.Y_val),
                )
