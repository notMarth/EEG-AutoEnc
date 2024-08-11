import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import numpy as np

#Autoencoder model definition
class Autoencoder(Model):
    def __init__(self, latent_dim, input_shape, output_shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = (input_shape[0], input_shape[-1])
        self.output_shape = output_shape
        self.encoder_unit = tf.keras.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Flatten(),
            layers.Dense(self.latent_dim),
        ])

        self.decoder_unit = tf.keras.Sequential([
            layers.Dense(self.output_shape[-1]),
            layers.Dropout(0.3),
            layers.Reshape(self.output_shape),
        ])

    def encoder(self, x) -> list:
        out = []
        
        for i in range(x.shape[-1]):
            out.append(self.encoder_unit(x[:,:,i]).numpy())
        return out

    def decoder(self, x) -> np.ndarray:
        out = np.array(x[0])
        for i in x[1:]:
            np.concatenate((out, i), axis=0)
        return self.decoder_unit(out)

    def call(self, x) -> np.ndarray:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def predict(self, data, labels):
        reconstructions = self(data).numpy()
        loss = losses.mse(reconstructions, labels)
        return reconstructions, loss
