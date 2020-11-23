import numpy as np
from keras.datasets import mnist
from tqdm import trange

(train_X, train_y),(test_X, test_y) = mnist.load_data()
print("train_x.shape: ",train_X.shape)

train_X.reshape((-1,28,28))
test_X.reshape((-1,28,28))
print("train_x.shape: ",train_X.shape)

train_X = train_X.reshape(-1,28*28)
print("train_x.shape: ",train_X.shape)

len_inp = 28*28
len_hidden = 128
len_out = 10
batch_size = 128
def layer_init(m, h):
  # gaussian is strong
  #ret = np.random.randn(m,h)/np.sqrt(m*h)
  # uniform is stronger
  ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
  return ret.astype(np.float32)
def logsumexp(x):
  #return np.log(np.exp(x).sum(axis=1))
  # http://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
  c = x.max(axis=1)
  return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))

w1 = layer_init(len_inp,len_hidden)
w2 = layer_init(len_hidden, len_ou)



def forward_backward(x, y):
  # training
  out = np.zeros((len(y),10), np.float32)
  out[range(out.shape[0]),y] = 1

  # forward pass
  x_l1 = x.dot(w1)
  x_relu = np.maximum(x_l1, 0)
  x_l2 = x_relu.dot(w2)
  x_lsm = x_l2 - logsumexp(x_l2).reshape((-1, 1))
  x_loss = (-out * x_lsm).mean(axis=1)

  # training in numpy (super hard!)
  # backward pass

  # will involve x_lsm, x_l2, out, d_out and produce dx_sm
  d_out = -out / len(y)

  # derivative of logsoftmax
  # https://github.com/torch/nn/blob/master/lib/THNN/generic/LogSoftMax.c
  dx_lsm = d_out - np.exp(x_lsm)*d_out.sum(axis=1).reshape((-1, 1))

  # derivative of l2
  d_l2 = x_relu.T.dot(dx_lsm)
  dx_relu = dx_lsm.dot(l2.T)

  # derivative of relu
  dx_l1 = (x_relu > 0).astype(np.float32) * dx_relu

  # derivative of l1
  d_l1 = x.T.dot(dx_l1)
  
  return x_loss, x_l2, d_l1, d_l2

samp = [0,1,2,3]
x_loss, x_l2, d_l1, d_l2 = forward_backward(test_X[samp].reshape((-1, 28*28)), test_y[samp])

w1_shape = (len_inp,len_hidden)
w2_shape = (len_hidden,len_out)
losses, accuracies = [],[]
for i in (t:= trange(1000)):
    samp = np.random.randint(0, train_X.shape[0], size=(batch_size))
    X = train_X[samp].reshape((-1,28*28))
    Y = train_y[samp]
    x_loss,x_l2 , d_l1,d_l2 = forward_backward(X,Y)
