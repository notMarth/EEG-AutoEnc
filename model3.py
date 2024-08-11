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
            layers.Conv2D(16, (3,3), strides=(2,2)),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.10),
            layers.Dropout(0.20),
            layers.Conv2D(8, (3,3), strides=(1,1)),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.10),
            layers.Dropout(0.20),


            layers.Conv2D(32, (3,3), strides=(2,2)),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.10),
            layers.Dropout(0.20),
            layers.Conv2D(16, (3,3), strides=(1,1)),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.10),
            layers.Dropout(0.20),
            layers.Reshape((1246, 4*16)),
            layers.LSTM(self.latent_dim, recurrent_dropout=0.20),
        ])

        self.decoder = tf.keras.Sequential([
            #layers.RepeatVector(1246),
            layers.Reshape((1, latent_dim)),
            layers.LSTM(50, return_sequences=True),
            #layers.Reshape(1246, 4*16),
            #layers.Reshape((1246, 4, 16)),
            layers.Reshape((25, 2)),
            #layers.UpSampling1D(2),
            #layers.ZeroPadding1D(1),
            layers.Conv1DTranspose(5, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.10),
            layers.Dropout(0.20),
            layers.Conv1DTranspose(10, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.10),
            layers.Dropout(0.20),
            layers.Conv1DTranspose(25, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.10),
            layers.Dropout(0.20),

            # #layers.UpSampling1D(2),
            # #layers.ZeroPadding1D(1),
            # layers.Conv1DTranspose(5, 3, strides=2, padding='same'),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(0.10),
            # layers.Dropout(0.20),
            # layers.Conv1DTranspose(10, 3, strides=2, padding='same'),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(0.10),
            # layers.Dropout(0.20),
            # layers.Conv1DTranspose(10, 3, strides=2, padding='same'),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(0.10),
            # layers.Dropout(0.20),

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
