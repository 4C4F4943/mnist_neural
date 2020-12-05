# Neural network for the mnist dataset

first get the dataset from this <a href="https://www.kaggle.com/oddrationale/mnist-in-csv?select=mnist_train.csv"> kaggle</a> link.

Here i was trying to make a neural network from scratch for the mnist dataset but i predicted values based on the learning rate...So wanted to try a framework.
##### Pytorch the best framework;), this is personal.
## the net

the network itselft is quite simple it has 
748 input linear neurons 
128 hidden linear neurons and then at last 
10 output neurons which uses a softmax as an activation function.
The loss function is negative log likelyhood and the optimize is Adam.
If you have looked at the code then u see it only uses weights. Whith Pytorch it's very easy to test the difference and the accuracy was better without them.
Here is a graph of the cost and the acc.
<img src="https://github.com/4C4F4943/mnist_neural/blob/main/cost:acc_mnist.png"></img>
#### the accuracy
For some reason the Adam optimize works wonders and almost always gives me a redicoulesly high training accuracy of 95-100% accuracy.
But with the eval() function you can see the test accuracy is quite a lot lower but still good around 80%.
I left all of the activation functions there for the forward pass so you can play around with them like this.
```python
def forward(x):
    x = x.dot(l1)
    x = x.dot(l2)
    #here you can change the activation function
    x = the_function_you_want(x)
    return x```
