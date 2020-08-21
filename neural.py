from mnist import MNIST
import numpy as np
import math

mndata = MNIST('C:\\Users\\10755\\PycharmProjects\\AIlab_q2\\venv\\Include')
images, labels = mndata.load_training()
test_image, test_label = mndata.load_testing()

alpha = 0.05
m = 10
epoch = 30
n0 = 784        # number of neurons in each layer
n1 = 30
n2 = 10

image = np.array(images) / 255.0    # transfer list to matrix
label = np.array(labels)
t_image = np.array(test_image) / 255.0
t_label = np.array(test_label)

nl = len(image)
nt = len(t_image)

def shuffle_image(mini_batch_size):
    permutation = list(np.random.permutation(nl)
    image_batch = []
    label_batch = []
    num = math.floor(nl / m)
    for i in range(num):
        new_array = []
        for j in range(mini_batch_size):
            new_array.append(permutation[i*mini_batch_size + j])
        mb = np.array(new_array)
        mini_batch_X = image[mb]
        mini_batch_Y = label[mb]
        image_batch.append(mini_batch_X)
        label_batch.append(mini_batch_Y)
    return image_batch, label_batch

def ga(x):
    x2 = [[1 / (1 + math.exp(-x[i][j])) for j in range(len(x[i]))] for i in range(len(x))]
    x1 = np.array(x2)
    return x1

def ga1(x):
    x2 = [1 / (1 + math.exp(-x[i])) for i in range(len(x))]
    x1 = np.array(x2)
    return x1

def trainingdata():
    w1 = np.random.normal(0, 1 / np.sqrt(n0), size=(n1, n0))    #initialize the weight
    w2 = np.random.normal(0, 1 / np.sqrt(n1), size=(n2, n1))
    b1 = np.zeros((n1, 1))      #initialize the bias
    b2 = np.zeros((n2, 1))
    one1 = np.ones((1, m))
    oneg = np.ones((n1, m))
    count = 0
    while count < epoch:
        shuffled_images, shuffled_labels = shuffle_image(m)
        num = math.floor(nl / m)
        for i in range(num):
            input = shuffled_images[i].T
            expect_y = label_m(shuffled_labels[i])
            z1 = np.dot(w1, input) + np.dot(b1, one1)   # forward computation
            a1 = ga(z1)
            z2 = np.dot(w2, a1) + np.dot(b2, one1)
            a2 = ga(z2)
            delta2 = a2 - expect_y                      # back propogation
            delta1 = np.dot(w2.T, delta2) * a1 * (oneg - a1)
            w2 = w2 - alpha / m * np.dot(delta2, a1.T)
            w1 = w1 - alpha / m * np.dot(delta1, input.T)
            b2 = b2 - alpha / m * np.dot(delta2, one1.T)
            b1 = b1 - alpha / m * np.dot(delta1, one1.T)
        count = count + 1
        test_m(w1, w2, b1, b2)

def test_m(w1, w2, b1, b2):     #test the model by other images
    correct = 0
    for i in range(nt):
        input = t_image[i]
        z1 = np.dot(w1, input.T) + np.dot(b1, [1])
        a1 = ga1(z1)
        z2 = np.dot(w2, a1) + np.dot(b2, [1])
        a2 = ga1(z2)
        index = a2.argmax(axis=0)
        if index == t_label[i]:
            correct = correct + 1
    print(correct/nt)

def label_m(x): # turn label into a vector
    A = [[0 for j in range(m)] for i in range(n2)]
    for i in range(n2):
        for j in range(m):
            if i == x[j]:
                A[i][j] = 1
    A1 = np.array(A)
    return A1

trainingdata()