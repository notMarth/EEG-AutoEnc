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
        self.name = "custom_model"

        self.latent_dim = latent_dim
        self.input_shape = None
        self.output_shape = None

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
        plt.savefig(f"figs/{self.name}/Model_{self.name}_{self.latent_dim}_Loss.png", dpi=300)

        X_test = np.array(self.X_test)
        Y_test = np.array(self.Y_test)
        model_Y = self(self.X_test[0].reshape(1, self.X_test[0].shape[0], self.X_test[0].shape[1])).numpy().reshape(1,-1)
        true_Y = Y_test[0]

        display_scale = StandardScaler().fit(true_Y)
        display_Y = display_scale.transform(model_Y)
        #true_Y = display_scale.transform(true_Y)
        pred, loss = self.predict(self.X_test, self.Y_test)

        fig, axs = plt.subplots(3,1)
        axs[0].plot(display_Y[0])
        axs[0].set_title(f"Reconstructed Audio Envelope (Latent Space = {self.latent_dim})")
        axs[1].plot(true_Y[0])
        axs[1].set_title("True Audio Envelope")
        axs[2].plot(-1*display_Y[0], color="Red", label="Reconstructed (Flipped)")
        axs[2].plot(true_Y[0], label="True", alpha=0.5, color="Green")
        axs[2].set_title("True Audio Envelope vs Reconstructed")
        fig.legend()
        
        plt.savefig(f"figs/{self.name}/Model_{self.name}_{self.latent_dim}_Recon.png", dpi=300)

        self.test_loss = np.average(loss, axis=0)[0]


    def process_data(self, eeg, audio, sample_rate, mode, segments, seconds):
        #calculate audio envelope
        audio = np.abs(sig.hilbert(audio.T))
        audio = np.average(audio, axis=0)
        #normalize audio
        audio_scaler = StandardScaler()
        audio = audio_scaler.fit_transform(audio.reshape(-1,1)).reshape(1, -1)

        #split data into train, test, validation
        self.X_train, self.Y_train, self.X_test, self.Y_test, self.X_val, self.Y_val = \
        helper.train_test_val_split(eeg, audio, self.train_size, \
                            self.test_size, sample_rate, self.random_state, \
                            mode=mode, num_segments=segments, \
                            seconds=seconds)
        

    def train(self):
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2], 1)
        self.input_shape = self.X_train.shape[1:]
        self.output_shape = self.Y_train.shape[1:]

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=self.input_shape),
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

        self.decoder = tf.keras.Sequential([
            layers.Dense(50),
            layers.Dropout(0.25),
            layers.Reshape((25, 2)),
            layers.Conv1DTranspose(5, 3, strides=2, padding='same', activation='sigmoid'),
            layers.Dropout(0.25),
            layers.Conv1DTranspose(10, 3, strides=2, padding='same', activation='sigmoid'),
            layers.Dropout(0.25),
            layers.Conv1DTranspose(25, 3, strides=2, padding='same', activation='sigmoid'),
            layers.Dropout(0.25),
            layers.Flatten(),
            #layers.Dense(self.output_shape[1]),
            layers.Reshape(self.output_shape)
        ])

        self.compile(optimizer="Adam", loss=losses.MeanSquaredError())


        self.history = self.fit(self.X_train, self.Y_train,
                epochs=self.epochs,
                validation_data=(self.X_val, self.Y_val),
                )

    def visualize_activations(self):
        layer = self.encoder.get_layer(index=0)
        visual = self.X_train[0].reshape(1, 5000, 31, 1)
        first = layer(visual).numpy().reshape(2500, 16, 16)
        
        plt.figure()
        plt.plot(visual[0,:,:,0])

        for i in range(first.shape[-1]):
            plt.figure()
            plt.plot(first[:,:,i])

        plt.show()