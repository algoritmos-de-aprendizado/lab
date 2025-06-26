import numpy as np
import tensorflow as tf

class SOM:
    """
    This class builds and trains a Self-Organizing Map (SOM) using TensorFlow. This implementation uses a rectangular grid.
    """

    def __init__(self,
                 width: int,
                 height: int,
                 input_dim: int,
                 initial_learning_rate: float,
                 final_learning_rate: float,
                 initial_neighbourhood_radius: float,
                 final_neighbourhood_radius: float,
                 epochs: int):
        """Initializes the Self-Organizing Map (SOM) class using TensorFlow.

        Args:
            width (int): width of the SOM
            height (int): height of the SOM
            input_dim (int): dimension of the input data
            initial_learning_rate (float): initial learning rate
            final_learning_rate (float): final learning rate
            initial_neighbourhood_radius (float): initial neighbourhood radius
            final_neighbourhood_radius (float): final neighbourhood radius
            epochs (int): epochs to train the network
        """

        # sets output and input dimensions
        self.m = width  # width of output map
        self.n = height  # height of output map
        self.dimension = input_dim  # dimension of input data (vector)

        # Configuração de alguns hiperparâmetros
        self.alpha_0 = initial_learning_rate  # Learning rate inicial
        self.alpha_T = final_learning_rate  # Learning rate final
        self.epochs = epochs  # number of epochs

        self.sigma_0 = initial_neighbourhood_radius  # Raio inicial da vizinhança
        self.sigma_T = final_neighbourhood_radius  # Raio final da vizinhança

        # Inicialização aleatória dos pesos da rede SOM.
        self.map_weights = tf.Variable(
            tf.random.uniform(
                shape=[self.m * self.n, self.dimension],
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32
            )
        )

        # Posições dos neurônios
        self.map_loc = tf.constant(
            np.array(
                list(self._neuron_locations(self.m, self.n))
            )
        )

        self.n_shots = 10  # Número de vezes que é salvo o peso do SOM por época.
        self.saved_weights = tf.Variable(tf.zeros([self.epochs * self.n_shots + 1,
                                                   self.m * self.n,
                                                   self.dimension],
                                                  dtype=tf.float32))

    def _neuron_locations(self, m, n):
        # nested iterations over both dimensions to yield one by one the 2-d locations of the individual neurons in the SOM
        for i in range(m):
            for j in range(n):
                yield np.array([i, j],
                               dtype=np.float32)  # We should use yield when we want to iterate over a sequence, but don't want to store the entire sequence in memory

    def compute_winner(self, sample):
        self.sample = sample

        # Distância euclidiana entre cara neurônio w_i e vetor de entrada x_j.
        self.squared_distance = tf.reduce_sum(
            tf.square(
                tf.subtract(
                    self.map_weights,  # [m*n, dimension]
                    tf.expand_dims(
                        self.sample,  # [dimension] -> [1, dimension]
                        axis=0
                    )
                )
            ),
            axis=1
        )

        # BMU
        self.bmu_idx = tf.argmin(
            input=self.squared_distance,
            axis=0
        )

        # extract the bmu's 2-d location
        self.bmu_loc = tf.gather(
            self.map_loc,
            self.bmu_idx
        )

    def update_network(self, epsilon, eta):
        # compute the squared manhattan distance between the bmu and the neurons
        self.bmu_distance_squares = tf.reduce_sum(
            tf.square(
                tf.subtract(
                    self.map_loc,  # [m*n, 2]
                    tf.expand_dims(
                        self.bmu_loc,  # [2] -> [1, 2]
                        axis=0
                    )
                )
            ),
            axis=1
        )

        # compute the neighborhood function
        self.neighbourhood_func = tf.exp(
            tf.negative(
                tf.math.divide(
                    self.bmu_distance_squares,
                    tf.multiply(
                        tf.square(
                            eta,
                        ),
                        2.0
                    )
                )
            )
        )

        # compute the overall learning of each neuron
        self.learning = tf.multiply(
            self.neighbourhood_func,
            epsilon
        )

        # compute the difference between the neurons weights and the input
        self.delta_wgt = tf.subtract(
            tf.expand_dims(
                self.sample,  # [dimension] -> [1, dimension]
                axis=0
            ),
            self.map_weights,  # [m*n, dimension]
        )

        # compute the weights update according to the learning and delta_wgt and update the weights
        tf.compat.v1.assign_add(
            self.map_weights,
            tf.multiply(
                tf.expand_dims(
                    self.learning,  # [m*n] -> [m*n, 1]
                    axis=-1
                ),
                self.delta_wgt  # [m*n, dimension]
            )
        )

    def get_weights(self):
        return self.map_weights

    @tf.function
    def train(self, x_train):
        pos = 0
        self.saved_weights[pos].assign(self.map_weights)
        with tf.device('/device:gpu:0'):

            for epoch in tf.range(self.epochs):
                tf.print("---------- Epoch", epoch + 1, "----------")

                # updates the learning rate alpha, it decreases monotically with each epoch
                alpha = self._learning_rate(
                    alpha_0=self.alpha_0,
                    alpha_T=self.alpha_T,
                    epoch=epoch
                )

                # update the gaussian neighborhood witdh eta
                # sigma is the width of the neighborhood radius with each epoch
                sigma = self._neighbourhood_radius(
                    sigma_0=self.sigma_0,
                    sigma_T=self.sigma_T,
                    epoch=epoch
                )

                # shuffle the training dataset so that for each epoch we get a different sample order
                tf.random.shuffle(x_train)

                # bmu computing and network update for each sample
                # this n variable is just to save the weights of the network at each epoch n_shots times
                batch = int(len(x_train) / self.n_shots)
                for i in range(self.n_shots):
                    pos = pos + 1
                    for x_trn in x_train[i * batch:(i + 1) * batch]:
                        sample = tf.cast(x_trn, dtype=tf.float32)
                        self.compute_winner(sample)
                        self.update_network(alpha, sigma)
                    self.saved_weights[pos].assign(self.map_weights)

    def _learning_rate(self, alpha_0: float, alpha_T: float, epoch: int):
        # updates the learning rate alpha, it decreases monotically with each epoch
        alpha = tf.multiply(
            alpha_0,
            tf.pow(
                tf.math.divide(
                    alpha_T,
                    alpha_0
                ),
                tf.cast(  # Casts a tensor to a new type.
                    tf.math.divide(
                        epoch,
                        self.epochs - 1
                    ),
                    dtype=tf.float32
                )
            )
        )
        return alpha

    def _neighbourhood_radius(self, sigma_0: float, sigma_T: float, epoch: int):
        # updates the gaussian neighborhood witdh eta
        # sigma is the width of the neighborhood radius with each epoch
        sigma = tf.multiply(
            sigma_0,
            tf.pow(
                tf.math.divide(
                    sigma_T,
                    sigma_0
                ),
                tf.cast(  # Casts a tensor to a new type.
                    tf.math.divide(
                        epoch,
                        self.epochs - 1
                    ),
                    dtype=tf.float32
                )
            )
        )
        return sigma