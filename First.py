import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
import platform
from keras.datasets import mnist
#get in the data

#X, Y = loadlocal_mnist(images_path ="train-images.idx3-ubyte",
                        #bels_path="train-labels.idx1-ubyte")
#test_x,test_y = loadlocal_mnist(images_path="t10k-images.idx3-ubyte",
                         #       labels_path="t10k-labels.idx1-ubyte")

(train_X, train_y),(test_X, test_y) = mnist.load_data()

print("train_x.shape: ",train_X.shape)
print("train_y.shape: ",train_y.shape)
print('Digits:  0 1 2 3 4 5 6 7 8 9')
print('labels: %s' % np.unique(train_y))
print('Class distribution: %s' % np.bincount(train_y))

len_inp = 28*28
len_hidden = 16
len_outp = 10
shape_w1 = (len_inp,len_hidden)
shape_w2 = (len_hidden, len_outp)


#random init for weights with the uniform distribution basicaly just nicer values
w1 = np.random.uniform(-1,1,shape_w1)/np.sqrt(len_inp)
w2 = np.random.uniform(-1,1,shape_w2)/np.sqrt(len_hidden)

#init biasees with .full which makes an array based on a shape
b1 = np.full(len_hidden,0)
b2 = np.full(len_outp,0)
print("the shape of w1: ",w1.shape)
print("the shape of w2: ",w2.shape)
print("the shape of b1: ",b1.shape)
print("the shape of b2: ",b1.shape)

iteration = 2000
learning_rate = 0.05
batch_size = 32
n_tests = 2000

def sigmoid(dim):
    return 1/(1+np.exp(-dim))

def relu(x):
    x[x<0]=0
    return x
def forward_prop(img):
    x = img.flatten()/255
    h = np.dot(x,w1)+b1
    ha = 1/(1+np.exp(-h))

    y = np.dot(ha,w2) + b2
    exp_y = np.exp(y)
    ya = exp_y / exp_y.sum()
    
    return x,h,ha,y,ya

def loss(ya,t ):
    return -np.log(ya[t])

def backpropagation(x,h,ha,ya,t):
    d_b2 = ya
    d_b2[t] -= 1
    d_w2 = np.outer(ha,d_b2)
    d_b1 = np.dot(w2,d_b2) *ha *(1-ha)
    d_w1 = np.outer(x,d_b1)
    return d_w1, d_w2,d_b1,d_b2

def train():
    for k in range(iteration):
        sum_d_w1 = np.zeros(shape_w1)
        sum_d_w2 = np.zeros(shape_w2)
        sum_d_b1 = np.zeros(len_inp)
        sum_d_b2 = np.zeros(len_hidden)
        print("sum_d_w1.shape: ",sum_d_w1.shape)
        print("sum_d_w2.shape: ",sum_d_w2.shape)
        print("sum_d_b1.shape: ",sum_d_b1.shape)
        print("sum_d_b2.shape: ",sum_d_b2.shape)
        for i in range(batch_size):
            index = k * batch_size + i
            image = train_X[index]
            t = train_y[index]

            x,h,ha,y,ya = forward_prop(image)
            d_w1,d_w2,d_b1,d_b2 = backpropagation(x,h,ha,ya,t)
            print("d_w1.shape: ",d_w1.shape)
            print("d_w2.shape: ",d_w2.shape) 
            print("d_b1.shape: ",d_b1.shape) 
            print("d_b2.shape: ",d_b2.shape)
 
            sum_d_w1 += d_w1
            sum_d_w2 += d_w2
            sum_d_b1 += d_b1
            sum_d_b2 += d_b2
            
    w1[:] -= learning_rate * sum_d_w1
    w2[:] -= learning_rate * sum_d_w2
    b1[:] -= learning_rate * sum_d_b1
    b2[:] -= learning_rate * sum_d_b2




def test():
    if len(testy) != 1:
        random_number = np.random.randint(0,(len(test_y)-1))
    else:
        random_number = 0
    image = test_X[random_number]
    label = test_y[random_number]
    x,h,y,ya = forward_prop(image)
    result = ya.argmax()
    return result
def accuracy(result):
    acc = 0
    for i in range(n_tests):
        if result[i]:
            acc += 1
    return acc / n_tests

train()
result = test()
#print(result)
acc = accuracy(result)

print("acc: ",acc)
