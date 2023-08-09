import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname, "rb") as img_file:
        magic_num, img_num, row, col = struct.unpack(">4i", img_file.read(16))
        assert(magic_num == 2051)
        tot_pixels = row * col
        X = np.vstack([np.array(struct.unpack(f"{tot_pixels}B", img_file.read(tot_pixels)), dtype=np.float32) for _ in range(img_num)])
        X -= np.min(X)
        X /= np.max(X)

    with gzip.open(label_filename, "rb") as label_file:
        magic_num, label_num = struct.unpack(">2i", label_file.read(8))
        assert(magic_num == 2049)
        y = np.array(struct.unpack(f"{label_num}B", label_file.read()), dtype=np.uint8)

    return X, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    return (ndl.log(ndl.exp(Z).sum((1,))).sum() - (y_one_hot * Z).sum()) / Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    iterations = (y.size + batch - 1) // batch
    for i in range(iterations):
        x = ndl.Tensor(X[i * batch : (i+1) * batch, :])
        Z = ndl.relu(x.matmul(W1)).matmul(W2)
        yy = y[i * batch : (i+1) * batch]

        y_one_hot = np.zeros((batch, y.max() + 1))
        y_one_hot[np.arange(batch), yy] = 1
        y_one_hot = ndl.Tensor(y_one_hot)
        loss = softmax_loss(Z, y_one_hot)
        loss.backward()
        
        W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
    return W1, W2
    ### END YOUR SOLUTION
# 我们的计算图：
# x -> ReLU(x*w1+b1) = h -> ReLU(h*w2+b2) = y -> (y-target)^2 = L

# 我们有以下参数和数据：
# 输入 x = 2.0
# 目标值 target = 1.0
# 权重和偏置初始值：w1 = 0.5, b1 = 0.1, w2 = 0.6, b2 = 0.2

# 前向传播过程：
# h = ReLU(x * w1 + b1) = ReLU(2.0 * 0.5 + 0.1) = ReLU(1.1) = 1.1
# y = ReLU(h * w2 + b2) = ReLU(1.1 * 0.6 + 0.2) = ReLU(0.86) = 0.86
# L = (y - target)^2 = (0.86 - 1.0)^2 = 0.0196

# 接下来我们计算各个参数的梯度。
# 对于输出层：
# dL/dy = 2*(y-target) = 2*(0.86-1.0) = -0.28
# dL/dw2 = dL/dy * dh/dw2 = dL/dy * h = -0.28 * 1.1 = -0.308
# dL/db2 = dL/dy = -0.28
# dL/dh = dL/dy * dy/dh = dL/dy * w2 = -0.28 * 0.6 = -0.168

# 对于隐藏层：
# dL/dw1 = dL/dh * dx/dw1 = dL/dh * x = -0.168 * 2.0 = -0.336
# dL/db1 = dL/dh = -0.168

# 接下来，我们会用这些梯度来更新我们的权重和偏置。

# 接下来我们需要选择一个学习率（learning rate）来进行参数更新。假设我们选择学习率为0.1。
# 参数更新的公式是：新参数 = 旧参数 - 学习率 * 梯度。所以我们有：
# w1_new = w1 - learning_rate * dL/dw1 = 0.5 - 0.1 * -0.336 = 0.5336
# b1_new = b1 - learning_rate * dL/db1 = 0.1 - 0.1 * -0.168 = 0.1168
# w2_new = w2 - learning_rate * dL/dw2 = 0.6 - 0.1 * -0.308 = 0.6308
# b2_new = b2 - learning_rate * dL/db2 = 0.2 - 0.1 * -0.28 = 0.228

# 那么，更新后的权重和偏置值为：w1 = 0.5336，b1 = 0.1168，w2 = 0.6308，b2 = 0.228。
# 这就是我们进行一次梯度下降后的结果。在神经网络训练中，我们会重复这个过程多次（即多个"epoch"），
# 每次都基于当前的梯度来更新我们的权重和偏置，最终希望得到一个能够最小化损失的模型。





### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
