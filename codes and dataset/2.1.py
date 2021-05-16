import math
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

def tanh(z):
    return np.tanh(z)
def sigmoid(z):
    return 1/(1 + np.exp(-z))
def tanh(z):
    return np.tanh(z)
def sigmoid_der(z):
    return sigmoid(z)*(1.0-sigmoid(z))
def tanh_der(z):
    return 1.0 - np.power(z,2)
def relu(z):
    return z * (z>0)
def relu_der(z):
    x[x<0] = 0
    x[x>0] = 1
    return x
def sine(x) :
    return np.sin(x)
def generate_training_data(nsamples,f):
    pi = math.pi
    ll = -2*pi
    ul = 2*pi
    incr = (ul-ll)/nsamples
    x = np.arange(ll,ul,incr)
    y = f(x)
    x = np.array([x])
    y = np.array([y])
    return x,y
def generate_validation_data(nsamples):
    pi = math.pi
    ll = -2*pi
    ul = 2*pi
    incr = (ul-ll)/nsamples
    x = np.arange(ll,ul,incr)
    x = np.array([x])
    return x
def initialise_parameters(num_layers,layer_neurons):
    weights = []
    biases = []
    for l in range(0,num_layers-1) :
        w = np.random.randn(layer_neurons[l+1],layer_neurons[l])
        b = np.zeros((layer_neurons[l+1],1))
        weights.append(w)
        biases.append(b)
        l = l+1
    return weights,biases  
def forward_pass(xdata,weights,biases,act_func):
    I = len(weights)
    i = 0
    zs = []
    acts = []
    z = np.dot(weights[i],xdata)+biases[i]
    a = act_func(z)
    zs.append(z)
    acts.append(a)
    i = i+1
    while i<I-1 :
        z = np.dot(weights[i],acts[i-1])+biases[i]
        a = act_func(z)
        zs.append(z)
        acts.append(a)
        i = i+1
    z = np.dot(weights[i],acts[i-1])+biases[i]
    a = z   #last layer activation function control
    acts.append(a)
    return zs,acts
def calculate_cost(acts,ydata):
    y = ydata
    yp = acts[-1]
    m = y.shape[1]
    cost = np.sum(np.square(yp-y))/m
    return cost
def backward_pass(xdata,ydata,zs,acts,weights,biases,act_flag):
    dzs = []
    dws = []
    dbs = []
    i = len(weights)-1
    for k in range(i+1):
        dzs.append(0)
        dws.append(0)
        dbs.append(0)
    m = ydata.shape[1]
    dz = acts[i] - ydata
    dw = np.dot(dz,acts[i-1].T)/m
    db = np.sum(dz,axis=1,keepdims=True)/m
    dzs[i] = dz
    dws[i] = dw
    dbs[i] = db
    i = i-1
    while i>0 :
        if act_flag == "tanh" :
            dz = np.multiply(np.dot(weights[i+1].T,dz),tanh_der(acts[i]))
        if act_flag == "sigmoid":
            pass
        
        dw = np.dot(dz,acts[i-1].T)/m
        db = np.sum(dz,axis=1,keepdims=True)/m
        i = i-1
    if act_flag == "tanh" :
            dz = np.multiply(np.dot(weights[i+1].T,dz),tanh_der(acts[i]))
    if act_flag == "sigmoid":
            pass
    dw = np.dot(dz,xdata.T)/m
    db = np.sum(dz,axis=1,keepdims=True)/m
    dzs[i] = dz
    dws[i] = dw
    dbs[i] = db
    return dzs,dws,dbs

def update_param(weights,biases,dw,db,learning_rate):
    for i in range(len(weights)) :
        weights[i] = weights[i] - learning_rate*dw[i]
        biases[i] = biases[i] - learning_rate*db[i]
        i = i+1
    return weights,biases
def predict(xdata,weights,biases,act_func):
    zs,acts = forward_pass(xdata,weights,biases,act_func)
    yp = acts[-1]
    yp = np.squeeze(yp)
    return yp



# learning_rate = float(input("enter the value of learning rate - "))
learning_rates = [0.01,0.001,0.0001]
xdata,ydata = generate_training_data(1000,sine)
l1 = int(input("enter total number of layer (including input layer and output layer) - "))
l = []
print("layer 0 is the input layer and layer %s is the output layer"%(l1-1))
for i in range(l1) :
    l.append(int(input("enter the number of neurons in layer %s "%i)))
# weights,biases = initialise_parameters(5,[1,32,128,32,1])
weights,biases = initialise_parameters(l1,l)
e = int(input("enter number of epochs - "))
costs_maps = []

color_list = ["black","red","blue"]
color_list1 = ["black","red","blue"]
for learning_rate in learning_rates :
    costs_map = []
    iter_map = []
    weights,biases = initialise_parameters(l1,l)
    for i in range(e) :
        zs,acts = forward_pass(xdata,weights,biases,tanh)
        cost = calculate_cost(acts,ydata)
        dzs,dws,dbs = backward_pass(xdata,ydata,zs,acts,weights,biases,"tanh")
        weights,biases = update_param(weights,biases,dws,dbs,learning_rate)
        costs_map.append(cost)
        iter_map.append(i)
        if i%50 == 0 :
            print("cost at iteration",i,cost)

    costs_maps.append(costs_map)
   
p2 = plt.plot(iter_map,costs_maps[1-1],color="green")
p3 = plt.plot(iter_map,costs_maps[2-1],color="blue")
p4 = plt.plot(iter_map,costs_maps[3-1],color="red")
# plt.legend((p1,p2,p3,p4), ( 'a = 0.01', 'a = 0.001','a = 0.0001'))
plt.gca().legend(( 'a = 0.01', 'a = 0.001','a = 0.0001'))
plt.show()

 

    

