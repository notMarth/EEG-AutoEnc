import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

#Autoencoder model definition
class Autoencoder(Model):
    def __init__(self, latent_dim, input_shape, output_shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv2D(25, (4,4), strides=2, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            # layers.Conv2D(13, (4,4), strides=2, padding='same', activation='relu'),\
            # layers.MaxPooling2D(),
            layers.Conv2D(4, (4,4), strides=2, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            # layers.Conv2D(2, (4,4), strides=2, padding='same', activation='relu'),
            # layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(self.latent_dim),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(25),
            #layers.Dropout(0.25),
            layers.Reshape((25, 1)),
            # layers.Conv1DTranspose(2, 4, strides=2, padding='same', activation='relu'),
            # layers.UpSampling1D(2),
            layers.Conv1DTranspose(4, 4, strides=2, padding='same', activation='relu'),
            layers.UpSampling1D(2),
            # layers.Conv1DTranspose(13, 4, strides=2, padding='same', activation='relu'),
            #layers.UpSampling1D(2),
            layers.Conv1DTranspose(25, 4, strides=2, padding='same', activation='relu'),
            layers.Flatten(),
            #layers.Dense(self.output_shape[1]),
            layers.Reshape(self.output_shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def predict(self, data, labels):
        reconstructions = self(data).numpy()
        loss = losses.mse(reconstructions, labels)
        return reconstructions, loss
