# import libraries
import numpy as np


def label(class_nbr, weights, x_label, y_label):
    """Function responsible for labeling the SOM output.

    Args:
        class_nbr (int): amount of classes in our database training
        weights (_type_): weights for the resulting SOM
        x_label (_type_): training input data
        y_label (_type_): label for the input data

    Returns:
        neuron_label: returns a matrix of the same size of the SOM weights input with labels.
    """
    dist_sum = np.zeros((len(weights), class_nbr))
    nbr_digits = np.zeros((class_nbr,))

    # accumulate the normalized gaussian distance for the labeling dataset
    for (x, y) in zip(x_label, y_label):
        nbr_digits[y] += 1
        dist_neuron = np.exp(-np.linalg.norm(x - weights, axis=1))
        dist_bmu = np.max(dist_neuron)
        for i, distn in enumerate(dist_neuron):
            dist_sum[i][y] += distn / dist_bmu

    # normalize the activities on the number of samples per class
    for i, dists in enumerate(dist_sum):
        dist_sum[i] = dists / nbr_digits

    # assign the neurons labels
    neuron_label = np.argmax(dist_sum, axis=1)
    print("Neurons labels = ")
    print(neuron_label)

    return neuron_label


def test(weights, x_test, neuron_label):
    """Function responsible for testing the SOM output and measuring its prediction.

    Args:
        weights (_type_): weights for the resulting SOM
        x_test (_type_): testing input data
        neuron_label (_type_): labels of the resulting SOM

    Returns:
        y_pred: array with the predicted labels, index of BMU for each input
    """
    neuron_index = np.zeros((len(x_test),), dtype=int)
    y_pred = []

    # calculate the BMUs for the test dataset
    for i, x in enumerate(x_test):
        dist_neuron = np.linalg.norm(x - weights, axis=1)
        neuron_index[i] = np.argmin(dist_neuron)  # saves the index of the neuron with the lowest distance (BMU)

    # save the BMUs labels as prediction
    for p in neuron_index:
        y_pred.append(neuron_label[p])

    return y_pred