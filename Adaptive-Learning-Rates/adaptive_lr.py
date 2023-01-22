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

        # Initializing velocity terms to compute momentum
        self.v2 = np.zeros_like(self.W2)
        self.v1 = np.zeros_like(self.W1)

    def predict(self, x):
        # Predicting a class given the current weights and biases
        a1 = x.dot(self.W1) + self.b1
        z = relu(a1)
        a2 = z.dot(self.W2) + self.b2
        y = softmax(a2)
        return z, y

    def train(self, X, y, learning_rate, epochs, validation=None, batch_size=None, momentum=0.9, optimizer=None):
        # If we have an optimizer, initialize the cache
        if optimizer is not None:
            epsilon = 1e-10 # To avoid division by zero
            if optimizer == "adagrad":
                cacheW2 = np.zeros_like(self.W2)
                cacheb2 = np.zeros_like(self.b2)
                cacheW1 = np.zeros_like(self.W1)
                cacheb1 = np.zeros_like(self.b1)

            elif optimizer == "rmsprop":
                cacheW2 = np.ones_like(self.W2)
                cacheb2 = np.ones_like(self.b2)
                cacheW1 = np.ones_like(self.W1)
                cacheb1 = np.ones_like(self.b1)
                decay = 0.99

        # To store the error per epoch
        train_loss = np.zeros(epochs)
        val_loss = np.zeros(epochs)

        train_error = np.zeros(epochs)
        val_error = np.zeros(epochs)

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

                # Updating the cache for each param if we want to optimize
                if optimizer is not None:
                    # Update the cache with the current gradients
                    if optimizer == "adagrad":
                        cacheW2 += np.power(grad_L_W2, 2)
                        cacheb2 += np.power(grad_L_b2, 2)
                        cacheW1 += np.power(grad_L_W1, 2)
                        cacheb1 += np.power(grad_L_b1, 2)

                    elif optimizer == "rmsprop":
                        cacheW2 = decay*cacheW2 + (1-decay)*np.power(grad_L_W2, 2)
                        cacheb2 = decay*cacheb2 + (1-decay)*np.power(grad_L_b2, 2)
                        cacheW1 = decay*cacheW1 + (1-decay)*np.power(grad_L_W1, 2)
                        cacheb1 = decay*cacheb1 + (1-decay)*np.power(grad_L_b1, 2)

                    # Set the effective learning rates taking into account the cache
                    effective_lr_W2 = learning_rate/np.sqrt(cacheW2 + epsilon)
                    effective_lr_b2 = learning_rate/np.sqrt(cacheb2 + epsilon)
                    effective_lr_W1 = learning_rate/np.sqrt(cacheW1 + epsilon)
                    effective_lr_b1 = learning_rate/np.sqrt(cacheb1 + epsilon)
                    
                else:
                    # Else just keep the same fixed learning rate for all parameters
                    effective_lr_W2 = learning_rate
                    effective_lr_b2 = learning_rate
                    effective_lr_W1 = learning_rate
                    effective_lr_b1 = learning_rate

                # Performing gradient descent with momentum to update the model parameters
                self.v2 = momentum*self.v2 - effective_lr_W2*grad_L_W2
                self.v1 = momentum*self.v1 - effective_lr_W1*grad_L_W1

                self.W2 += self.v2
                self.b2 -= effective_lr_b2*grad_L_b2
                self.W1 += self.v1
                self.b1 -= effective_lr_b1*grad_L_b1
                
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

    # Vanilla SGD
    print("Vanilla Gradient Descent")                
    model = MLP(X_train.shape[1], 100, y_train_encoded.shape[1])
    train_loss, _, train_error, _ = model.train(X_train, y_train_encoded, 
                                                               learning_rate=0.00001, 
                                                               epochs=epochs, 
                                                               validation=(X_test, y_test_encoded),
                                                               batch_size=4096,
                                                               momentum=0)


    # Regular momentum
    print("Regular momentum")                
    model = MLP(X_train.shape[1], 100, y_train_encoded.shape[1])
    momentum_loss, _, momentum_error, _ = model.train(X_train, y_train_encoded, 
                                                               learning_rate=0.00001, 
                                                               epochs=epochs, 
                                                               validation=(X_test, y_test_encoded),
                                                               batch_size=4096)

    # AdaGrad
    print("AdaGrad Optimizer")                
    model = MLP(X_train.shape[1], 100, y_train_encoded.shape[1])
    adagrad_loss, _, adagrad_error, _ = model.train(X_train, y_train_encoded, 
                                                               learning_rate=0.001, 
                                                               epochs=epochs, 
                                                               validation=(X_test, y_test_encoded),
                                                               batch_size=4096,
                                                               optimizer="adagrad")

    # RMSProp
    print("RMSProp Optimizer")                
    model = MLP(X_train.shape[1], 100, y_train_encoded.shape[1])
    rmsprop_loss, _, rmsprop_error, _ = model.train(X_train, y_train_encoded, 
                                                               learning_rate=0.001, 
                                                               epochs=epochs, 
                                                               validation=(X_test, y_test_encoded),
                                                               batch_size=4096,
                                                               optimizer="rmsprop")

    # Plotting the loss
    plt.plot(train_loss, label="Vanilla SGD")
    plt.plot(momentum_loss, label="Momentum")
    plt.plot(adagrad_loss, label="AdaGrad")
    plt.plot(rmsprop_loss, label="RMSProp")
    plt.grid()
    plt.legend()
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Plotting the error
    plt.plot(train_error, label="Vanilla SGD")
    plt.plot(momentum_error, label="Momentum")
    plt.plot(adagrad_error, label="AdaGrad")
    plt.plot(rmsprop_error, label="RMSProp")
    plt.grid()
    plt.legend()
    plt.title("Error per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()