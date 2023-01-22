import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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
    print("Reading and formatting data, this might take a second...")
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

    # Train using Adam optimizer
    def train(self, X, y, epochs, validation=None, batch_size=None, learning_rate=0.001, beta1=0.9, beta2=0.999):
        # Initialize first and second gradient moments
        epsilon = 1e-8
        # First moment
        m2w = np.zeros_like(self.W2)
        m2b = np.zeros_like(self.b2)
        m1w = np.zeros_like(self.W1)
        m1b = np.zeros_like(self.b1)
        # Second moment
        v2w = np.zeros_like(self.W2)
        v2b = np.zeros_like(self.b2)
        v1w = np.zeros_like(self.W1)
        v1b = np.zeros_like(self.b1)

        # To store the error per epoch
        train_loss = np.zeros(epochs)
        val_loss = np.zeros(epochs)

        train_error = np.zeros(epochs)
        val_error = np.zeros(epochs)

        # Training step
        t = 1
        for epoch in range(epochs):
            # Shuffle the samples
            # # # 

            # If we want to validate, start by that
            if validation is not None:
                X_val, y_val = validation
                # Predicting validation data
                _, ym_val = self.predict(X_val)

                # Computing the validation loss
                L_val = -(y_val * np.log(ym_val)).mean()
                val_loss[epoch] = L_val

                # Compute the error
                err_val = error_rate(ym_val, y_val)
                val_error[epoch] = err_val

            # Computing the number of batches and start training for all batches
            if batch_size is None:
                n_batches = 1
                batch_size = X.shape[0]
            else:
                n_batches = int(np.ceil(X_train.shape[0] / batch_size))

            # To store the loss and the error during the training
            epoch_loss = 0
            epoch_error = 0
            
            # Updating the progress
            print_progress(0, n_batches, label="Batches")
            for i in range(n_batches):
                # Grab the corresponding batch from the samples
                Xb = X[i*batch_size:(i+1)*batch_size, :]
                yb = y[i*batch_size:(i+1)*batch_size, :]

                # Start by predicting the class, @ -> matrix multiplication, * -> element-wise multiplication
                Z, ym = self.predict(Xb)

                # Compute the training loss
                L = -(yb * np.log(ym)).mean()
                epoch_loss += L

                # Computing the error
                err = error_rate(ym, yb)
                epoch_error += err

                # Compute the gradients of the error with respect to the model parameters
                grad_L_W2 = Z.T.dot(ym-yb)
                grad_L_b2 = (ym-yb).sum(axis=0)
                # Gradient of relu(x) equals 1 if x != 0, otherwise equals 0
                grad_L_W1 = Xb.T.dot((ym-yb).dot(self.W2.T) * (Z > 0))
                grad_L_b1 = ((ym-yb).dot(self.W2.T) * (Z > 0)).sum(axis=0)

                # Updating first and second moment terms for the gradient
                # First moment
                m2w = beta1*m2w + (1-beta1)*grad_L_W2
                m2b = beta1*m2b + (1-beta1)*grad_L_b2
                m1w = beta1*m1w + (1-beta1)*grad_L_W1
                m1b = beta1*m1b + (1-beta1)*grad_L_b1
                # Second moment
                v2w = beta2*v2w + (1-beta2)*np.power(grad_L_W2, 2)
                v2b = beta2*v2b + (1-beta2)*np.power(grad_L_b2, 2)
                v1w = beta2*v1w + (1-beta2)*np.power(grad_L_W1, 2)
                v1b = beta2*v1b + (1-beta2)*np.power(grad_L_b1, 2)

                # Then, performing bias correction on the first and second moment terms
                # First moment correction
                m2w_hat = m2w / (1-np.power(beta1, t))
                m2b_hat = m2b / (1-np.power(beta1, t))
                m1w_hat = m1w / (1-np.power(beta1, t))
                m1b_hat = m1b / (1-np.power(beta1, t))
                # Second moment correction
                v2w_hat = v2w / (1-np.power(beta2, t))
                v2b_hat = v2b / (1-np.power(beta2, t))
                v1w_hat = v1w / (1-np.power(beta2, t))
                v1b_hat = v1b / (1-np.power(beta2, t))

                # Finally performing the step of gradient descent with the corrected moments
                self.W2 -= learning_rate*m2w_hat/(np.sqrt(v2w_hat) + epsilon)
                self.b2 -= learning_rate*m2b_hat/(np.sqrt(v2b_hat) + epsilon)
                self.W1 -= learning_rate*m1w_hat/(np.sqrt(v1w_hat) + epsilon)
                self.b1 -= learning_rate*m1b_hat/(np.sqrt(v1b_hat) + epsilon)

                # Increasing time step
                t += 1

                # Updating the progress
                print_progress(i+1, n_batches, label="Batches", info=f"Epoch: {epoch+1}")

            # After going through all batches, save the mean loss and error for the epoch
            train_loss[epoch] = epoch_loss/n_batches
            train_error[epoch] = epoch_error/n_batches

            # Print some info every 10 epochs
            if ((epoch+1) % 10 == 0):
                training_data = f"\nEpoch {epoch+1} -> loss: {train_loss[epoch]:.4f}, err: {train_error[epoch]:.4f}, val_loss: {val_loss[epoch]:.4f}, val_err: {val_error[epoch]:.4f}"
                print(training_data)

        return train_loss, val_loss, train_error, val_error


if __name__ =="__main__":
    # Reading in the MNIST dataset
    X_train, y_train, X_test, y_test = get_normalized_data()

    # One-hot encoding the labels, 10 classes for the MNIST dataset
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded = one_hot_encode(y_test)

    # Training the model with diferent momentum
    epochs = 100

    # No momentum, no adaptive learning rate
    print("RMSProp")
    model = MLP(X_train.shape[1], 100, y_train_encoded.shape[1])
    train_loss, _, train_error, _ = model.train(X_train, y_train_encoded,  
                                                               epochs=epochs, 
                                                               validation=(X_test, y_test_encoded),
                                                               batch_size=4096,
                                                               beta1=0)

    # Full Adam Optimizer  
    print("Adam")               
    model = MLP(X_train.shape[1], 100, y_train_encoded.shape[1])
    adam_train_loss, _, adam_train_error, _ = model.train(X_train, y_train_encoded, 
                                                               epochs=epochs, 
                                                               validation=(X_test, y_test_encoded),
                                                               batch_size=4096)

    # Plotting the loss
    plt.plot(train_loss, label="plain SGD")
    plt.plot(adam_train_loss, label="Adam")
    plt.grid()
    plt.legend()
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Plotting the error
    plt.plot(train_error, label="plain SGD")
    plt.plot(adam_train_error, label="Adam")
    plt.grid()
    plt.legend()
    plt.title("Error per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()