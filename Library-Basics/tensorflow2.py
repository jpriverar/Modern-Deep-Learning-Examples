import sys
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def print_progress(current, total, label="", info="", char_space=50):
    # Saving space for the start and end bars
    char_space -= 2

    # Calculating the completed characters in the progress bar
    completed_char = int((current/total) * char_space) if current != 0 else 0
    missing_char = char_space - completed_char
    progress_string = f"\r[{'='*completed_char}{' '*missing_char}] {current}/{total} {label} {info}"

    # Filling the completed chars
    sys.stdout.write(progress_string)
    sys.stdout.flush()

def get_normalized_data():
    print("Reading and formatting data, this might take a second...")
    df = pd.read_csv('../train.csv')
    data = df.values.astype(np.float32)

    # Shuffle the data to be in random order
    np.random.shuffle(data)

    # Define a test size percentage and split our data for train and testing
    X_data = data[:, 1:]
    y_data = data[:, 0]

    test_size = 0.3 # 30% for test
    test_index = int((1-test_size) * data.shape[0])
    X_train, X_test = X_data[:test_index], X_data[test_index:]
    y_train, y_test = y_data[:test_index], y_data[test_index:]

    # Normalize the data, image pixels, so they go from 0-255
    X_train /= 255
    X_test /= 255

    return X_train, y_train, X_test, y_test

def one_hot_encode(y):
    N = len(y)
    y = y.astype(np.int32)
    K = y.max() + 1
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def error_rate(p, t):
    return np.mean(p != t)

class NeuralNetwork(tf.Module):
    def __init__(self, input_units, hidden_units_1, hidden_units_2, output_units):

        # Initialization variables
        W1_init = np.random.randn(input_units, hidden_units_1) / np.sqrt(input_units)
        b1_init = np.zeros((1, hidden_units_1))
        W2_init = np.random.randn(hidden_units_1, hidden_units_2) / np.sqrt(hidden_units_1)
        b2_init = np.zeros((1, hidden_units_2))
        W3_init = np.random.randn(hidden_units_2, output_units) / np.sqrt(hidden_units_2)
        b3_init = np.zeros((1, output_units))

        # Creating the tensorflow variables
        self.W1 = tf.Variable(W1_init.astype(np.float32))
        self.b1 = tf.Variable(b1_init.astype(np.float32))
        self.W2 = tf.Variable(W2_init.astype(np.float32))
        self.b2 = tf.Variable(b2_init.astype(np.float32))
        self.W3 = tf.Variable(W3_init.astype(np.float32))
        self.b3 = tf.Variable(b3_init.astype(np.float32))

        # Saving the parameters
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, X):
        # Making sure x is a tensor
        X = tf.cast(X, dtype=np.float32)
    
        # Moving forward
        Z1 = tf.matmul(X, self.W1) + self.b1
        Z1 = tf.nn.relu(Z1)
        Z2 = tf.matmul(Z1, self.W2) + self.b2
        Z2 = tf.nn.relu(Z2)
        Z3 = tf.matmul(Z2, self.W3) + self.b3
        return Z3

    def predict(self, X):
        A = self.forward(X)
        return tf.argmax(tf.nn.softmax(A), axis=1) # Argmax along the columns for every row

    def loss(self, A, Y):
        # Making sure input as tensors
        A = tf.cast(A, dtype=np.float32)
        Y = tf.cast(Y, dtype=np.float32)

        # Cross entropy with logits
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=A)
        return tf.reduce_mean(loss)

    def train(self, X, Y, epochs, batch_size):
        # To save the losses
        losses = np.zeros(epochs)

        # Defining the optimizer
        optimizer = tf.keras.optimizers.Adam()

        # Computing the number of batches
        n_batches = int(np.ceil(X.shape[0]/batch_size))

        # Iterating over all epochs and batches for every epoch
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(n_batches):
                # Grabbing the corresponding batch of data
                Xb = X[i*batch_size:(i+1)*batch_size]
                Yb = Y[i*batch_size:(i+1)*batch_size]

                with tf.GradientTape() as tape:
                    # First getting our current predictions
                    predicted = self.forward(Xb)
                    loss = self.loss(predicted, Yb)

                # Saving the error
                epoch_loss += loss.numpy()

                # Getting the gradients of the loss with respect to all the model parameters and updating
                grads = tape.gradient(loss, self.params)
                optimizer.apply_gradients(zip(grads, self.params))

                # Print progress
                print_progress(i+1, n_batches, label="Batches", info=f"Epoch: {epoch+1}")

            losses[epoch] = epoch_loss/n_batches

        print("Done training...")
        return losses


# copy this first part from theano2.py
def main():
    # step 1: get the data and define all the usual variables
    Xtrain, Ytrain, Xtest, Ytest = get_normalized_data()

    Ytrain_ind = one_hot_encode(Ytrain)
    Ytest_ind = one_hot_encode(Ytest)

    # Creating the model and training
    model = NeuralNetwork(Xtrain.shape[1], 500, 100, 10) # Units per layer, 4 layers in this case
    loss = model.train(Xtrain, Ytrain_ind, epochs=100, batch_size=8192) # Training for 10 epochs
    
    plt.plot(loss)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()