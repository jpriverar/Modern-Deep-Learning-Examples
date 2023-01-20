import sys
import numpy as np
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

def relu(x):
    z = x.copy()
    z[z < 0] = 0
    return z

def softmax(x):
    expx = np.exp(x)
    return expx / expx.sum(axis=1, keepdims=True)

def one_hot_encode(y):
    N = len(y)
    y = y.astype(np.int32)
    K = y.max() + 1
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def error_rate(ym, y):
    # ym and y are in probability format, we have to extract first the most probable class
    ym_class = np.argmax(ym, axis=1)
    y_class = np.argmax(y, axis=1)

    # Then get the mean of the missed predictions
    return np.mean(np.not_equal(ym_class, y_class))

def get_normalized_data():
    df = pd.read_csv('train.csv')
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

class MLP:
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        # Initializing weights and biases for the network
        self.W1 = np.random.randn(input_neurons, hidden_neurons) / np.sqrt(input_neurons)
        self.b1 = np.zeros(hidden_neurons)

        self.W2 = np.random.randn(hidden_neurons, output_neurons) / np.sqrt(hidden_neurons)
        self.b2 = np.zeros(output_neurons)

    def predict(self, x):
        # Predicting a class given the current weights and biases
        a1 = x.dot(self.W1) + self.b1
        z = relu(a1)
        a2 = z.dot(self.W2) + self.b2
        y = softmax(a2)
        return z, y

    def train(self, X, y, learning_rate, epochs, validation=None):
        # To store the error per epoch
        train_loss = np.zeros(epochs)
        val_loss = np.zeros(epochs)

        train_error = np.zeros(epochs)
        val_error = np.zeros(epochs)

        for epoch in range(epochs):
            # If we want to validate, start by that
            if validation is not None:
                # Predicting validation data
                X_val, y_val = validation
                _, ym_val = self.predict(X_val)

                # Computing the validation loss
                L_val = -(y_val * np.log(ym_val)).mean()
                val_loss[epoch] = L_val

                # Compute the error
                err_val = error_rate(ym_val, y_val)
                val_error[epoch] = err_val

            # Start by predicting the class, @ -> matrix multiplication, * -> element-wise multiplication
            Z, ym = self.predict(X)

            # Compute the training loss
            L = -(y * np.log(ym)).mean()
            train_loss[epoch] = L

            # Computing the error
            err = error_rate(ym, y)
            train_error[epoch] = err

            # Compute the gradients of the error with respect to the model parameters
            grad_L_W2 = Z.T.dot(ym-y)
            grad_L_b2 = (ym-y).sum(axis=0)
            # Gradient of relu(x) equals 1 if x != 0, otherwise equals 0
            grad_L_W1 = X.T.dot((ym-y).dot(self.W2.T) * (Z > 0))
            grad_L_b1 = ((ym-y).dot(self.W2.T) * (Z > 0)).sum(axis=0)

            # Performing gradient descent to update the model parameters
            self.W2 -= learning_rate*grad_L_W2
            self.b2 -= learning_rate*grad_L_b2
            self.W1 -= learning_rate*grad_L_W1
            self.b1 -= learning_rate*grad_L_b1

            # Updating the progress
            training_data = f"loss: {L:.4f}, err: {err:.4f}, val_loss: {L_val:.4f}, val_err: {err_val:.4f}"
            print_progress(epoch+1, epochs, label="Epochs", info=training_data)

        return train_loss, val_loss, train_error, val_error


if __name__ =="__main__":
    # Reading in the MNIST dataset
    X_train, y_train, X_test, y_test = get_normalized_data()

    # One-hot encoding the labels, 10 classes for the MNIST dataset
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded = one_hot_encode(y_test)
    
    # Creating the model
    model = MLP(X_train.shape[1], 100, y_train_encoded.shape[1])

    # Training the model
    epochs = 500
    train_loss, val_loss, train_error, val_error = model.train(X_train, y_train_encoded, learning_rate=0.00001, epochs=epochs, validation=(X_test, y_test_encoded))

    # Plotting the loss
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.grid()
    plt.legend()
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Plotting the error
    plt.plot(train_error, label="train")
    plt.plot(val_error, label="val")
    plt.grid()
    plt.legend()
    plt.title("Error per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()