import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    return x + y

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  
            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.
    """
    with gzip.open(image_filename, "rb") as img_file:
        magic_num, num_images, rows, cols = struct.unpack(">4i", img_file.read(16))
        assert(magic_num == 2051)
        total_pixels = rows * cols
        X = np.vstack([np.array(struct.unpack(f"{total_pixels}B", img_file.read(total_pixels)), dtype=np.float32) for _ in range(num_images)])
        X -= np.min(X)
        X /= np.max(X)

    with gzip.open(label_filename, "rb") as label_file:
        magic_num, num_labels = struct.unpack(">2i", label_file.read(8))
        assert(magic_num == 2049)
        y = np.array(struct.unpack(f"{num_labels}B", label_file.read()), dtype=np.uint8)

    return X, y


def softmax_loss(logits, true_labels):
    """ Return softmax loss.

    Args:
        logits (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        true_labels (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    return (np.sum(np.log(np.sum(np.exp(logits), axis=1))) - np.sum(logits[np.arange(true_labels.size), true_labels]))/true_labels.size


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    num_iterations = (y.size + batch - 1) // batch
    for i in range(num_iterations):
        batch_X = X[i * batch : (i+1) * batch, :]
        batch_y = y[i * batch : (i+1) * batch]
        softmax_input = batch_X @ theta
        softmax_output = np.exp(softmax_input)
        softmax_output = softmax_output / np.sum(softmax_output, axis=1, keepdims=True)
        one_hot_y = np.zeros((batch, y.max() + 1))
        one_hot_y[np.arange(batch), batch_y] = 1
        grad = batch_X.T @ (softmax_output - one_hot_y) / batch
        assert(grad.shape == theta.shape)
        theta -= lr * grad



def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarrray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarrray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_iterations = (y.size + batch - 1) // batch
    for i in range(num_iterations):
        batch_X = X[i * batch : (i+1) * batch, :]
        batch_y = y[i * batch : (i+1) * batch]
        
        # Forward propagation
        hidden_layer = batch_X @ W1
        hidden_layer[hidden_layer < 0] = 0  # ReLU activation
        output_logits = hidden_layer @ W2
        output_probs = np.exp(output_logits)
        output_probs = output_probs / np.sum(output_probs, axis=1, keepdims=True)
        
        # Create one-hot encoded labels
        one_hot_y = np.zeros((batch, y.max() + 1))
        one_hot_y[np.arange(batch), batch_y] = 1
        
        # Calculate gradients
        output_error = output_probs - one_hot_y
        relu_derivative  = np.zeros_like(hidden_layer)
        relu_derivative [hidden_layer > 0] = 1
        hidden_error = relu_derivative  * (output_error @ W2.T)
        
        grad_W1 = batch_X.T @ hidden_error / batch
        grad_W2 = hidden_layer.T @ output_error / batch
        
        # Update weights
        W1 -= lr * grad_W1
        W2 -= lr * grad_W2
    ### END YOUR CODE





### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
