import math
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import pickle


def tanh(z):
    return np.tanh(z)
def sigmoid(z):
    return 1.0/(1 + np.exp(-z))
def tanh(z):
    return np.tanh(z)
def sigmoid_der(z):
    return sigmoid(z)*(1.0-sigmoid(z))
def tanh_der(z):
    return 1.0 - np.power(z,2)
def relu(z):
    return z * (z>0)
def relu_der(x):
    x[x<0] = 0
    x[x>0] = 1
    return x
def sine(x) :
    return np.sin(x)

import copy
def normalize(data):
    data2=copy.deepcopy(data)
    #print "data",data
    for i in range(len(data)):
        max_elem=np.max(data[i])
        min_elem=np.min(data[i])
        su=max_elem+min_elem
        diff=max_elem-min_elem
        #print su,diff
        if data.shape[1]>1:
            for j in range(len(data[i])):    
                num=2*data[i][j]-su
                #print "dd", data[i][j],"num", num
                data2[i][j]=num/(1.0*diff)     
        else:
            num=2*data[i]-su
            data2[i]=num/(1.0*diff)   
    return data2

def denormalize(data,data2): #data is actual, data2 is normalized
    data3=copy.deepcopy(data2)
    for i in range(len(data2)):
        max_elem=np.max(data[i])
        min_elem=np.min(data[i])
        su=float(max_elem+min_elem)
        diff=float(max_elem-min_elem)
        if data.shape[1]>1:
            for j in range(len(data2[i])):
                data3[i][j]=(data2[i][j]*(diff)+su)/2
        else:
            data3[i]=(data2[i]*(diff)+su)/2
    return data3

def gen_data():
    df = pd.read_excel("Folds5x2_pp.xlsx")  
    Df = df
    data = Df.values
    test_data = data[8611:]
    train_data = data[:8611]
    train_data1 = train_data[2:]
    train_act = []
    valid_act = []
    train_act.append(train_data[0])
    valid_act.append(train_data[1])
    for i in train_data1 :
        if np.where(train_data1==i)[0][0] %5 == 0 :
            valid_act.append(i)
        else :
            train_act.append(i)
    valid_act = np.array(valid_act)
    valid_act.shape
    train_act = np.array(train_act)
    train_act.shape   
    x_data = train_act[...,:4].T
    #print x_data
    #exit()
    y_data = train_act[...,4:].T
    x_valid = valid_act[...,:4].T
    y_valid = valid_act[...,4:].T  
    x_test = test_data[...,:4].T 
    y_test = test_data[...,4:].T   

    x_d=normalize(x_data)
    print ("x_d",x_d)
    y_d=normalize(y_data)
    x_v=normalize(x_valid)
    y_v=normalize(y_valid)
    x_t=normalize(x_test)
    y_t=normalize(y_test)

    return x_d,y_d,x_v,y_v,x_t,y_t  , x_data,y_data,x_valid,y_valid,x_test,y_test

def initialise_parameters(num_layers,layer_neurons):
    weights = []
    biases = []
    for l in range(0,num_layers-1) :
        w = (1.0/np.sqrt(layer_neurons[l]))*np.random.randn(layer_neurons[l+1],layer_neurons[l])
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

def denn(val,max_elem,min_elem):
    op=(val*(max_elem-min_elem)+(max_elem+min_elem))/2
    return op

def calculate_cost_test(acts,ydata,data_norm_y,data_norm_x):#data for denormalizing
    y = denormalize(data_norm_y,ydata)
    yp = denormalize(data_norm_y,acts[-1])
    # y=ydata
    # yp=acts[-1]
    m = y.shape[1]
    cost = np.sqrt(np.sum(np.square(yp-y))/m)
    return cost

def calculate_cost(acts,ydata,data_norm_y,data_norm_x,weights):#data for denormalizing
    # cost_weights_term = []
    # m = ydata.shape
    # # print("shape - ",m)
    # for w in weights :
    #     # print("w",w.shape)
    #     cost_weights_term_1_layer = []
    #     for ww in w :
    #         for www in ww :
    #             cost_weights_term_1_layer = cost_weights_term_1_layer + np.power(www,2)
    #     cost_weights_term = cost_weights_term_1_layer
    # print()
    y = denormalize(data_norm_y,ydata)
    yp = denormalize(data_norm_y,acts[-1])
    m = y.shape[1]
    cost = np.sqrt(np.sum(np.square(yp-y))/m)
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
            dz = np.multiply(np.dot(weights[i+1].T,dz),sigmoid_der(acts[i]))
        if act_flag == "relu" :
            dz = np.multiply(np.dot(weights[i+1].T,dz),relu_der(acts[i]))
        
        dw = np.dot(dz,acts[i-1].T)/m
        db = np.sum(dz,axis=1,keepdims=True)/m
        i = i-1
    if act_flag == "tanh" :
            dz = np.multiply(np.dot(weights[i+1].T,dz),tanh_der(acts[i]))
    if act_flag == "sigmoid":
            dz = np.multiply(np.dot(weights[i+1].T,dz),sigmoid_der(acts[i])) 
    if act_flag == "relu" :
            dz = np.multiply(np.dot(weights[i+1].T,dz),relu_der(acts[i]))
    dw = np.dot(dz,xdata.T)/m
    db = np.sum(dz,axis=1,keepdims=True)/m
    dzs[i] = dz
    dws[i] = dw
    dbs[i] = db
    return dzs,dws,dbs

def update_param(weights,biases,dw,db,learning_rate,lamda):
    for i in range(len(weights)) :
        # print("weights[i] shape - ",weights[i].shape)
        # weights[i] = weights[i] - learning_rate*dw[i] - lamda*(np.absolute(weights[i]))
        weights[i] = weights[i] - learning_rate*dw[i] - lamda*weights[i]
        biases[i] = biases[i] - learning_rate*db[i] 
        # #added weights cap
        #  for j in range(len(weights[i])):
        #      for k in range(len(weights[i][j])):
        #          if weights[i][j][k]>1:
        #              weights[i][j][k]=1
        #          if weights[i][j][k]<-1:
        #              weights[i][j][k]=-1
                
    return weights,biases
def predict(xdata,weights,biases,act_func):
    zs,acts = forward_pass(xdata,weights,biases,act_func)
    yp = acts[-1]
    yp = np.squeeze(yp)
    return acts



#learning_rate = float(input("enter the value of learning rate - "))
learning_rate=0.001
lamda = 0.1
#n means not normlalized
xdata_ok,ydata_ok,x_valid,y_valid,xtest,ytest,nx_data,ny_data,nx_valid,ny_valid,nx_test,ny_test = gen_data()
print(xtest.shape)
print(x_valid.shape)
#l1 = int(input("enter total number of layer (including input layer and output layer) - "))
l = []

#print("layer 0 is the input layer and layer %s is the output layer"%(l1-1))
#for i in range(l1) :
 #   l.append(int(input("enter the number of neurons in layer %s "%i)))
# weights,biases = initialise_parameters(5,[1,32,128,32,1])
l1=4
l=[4,64,32,1]
weights,biases = initialise_parameters(l1,l)
#e = int(input("enter number of epochs - "))
batch_size=100
num_of_batches=xdata_ok.shape[1]/batch_size
print( "len_x",xdata_ok.shape[1],"num_batches",num_of_batches)
e=50
k=zip(xdata_ok.T,ydata_ok.T)

max_elem=np.max(ny_data)
min_elem=np.min(ny_data)
costs_map = []
for i in range(e) :
    
    random.shuffle(k)
    xdata_full,ydata_full=zip(*k)
    xdata_full=np.array(xdata_full).T
    ydata_full=np.array(ydata_full).T
    cost=0
    for batch_num in range(num_of_batches):

        #xdata=np.array(xdata_full[batch_size*batch_num:batch_size*(batch_num+1)])
        xdata=xdata_full[:,batch_size*batch_num:batch_size*(batch_num+1)]
        #print "Xxx",xdata.shape
        #exit()
        #print X[batch_size*batch_num:batch_size*(batch_num+1):1]
        #ydata=np.array([ydata_full[batch_size*batch_num:batch_size*(batch_num+1)]])
        ydata=ydata_full[:,batch_size*batch_num:batch_size*(batch_num+1)]

        zs,acts = forward_pass(xdata,weights,biases,relu)
        # cost = calculate_cost(acts,ydata_full)
        # acts_valid = predict(x_valid,weights,biases,tanh)
        # cost_valid = calculate_cost(acts_valid,y_valid)
        dzsl,dws,dbs = backward_pass(xdata,ydata,zs,acts,weights,biases,"relu")
        weights,biases = update_param(weights,biases,dws,dbs,learning_rate,lamda)
    
        
        # cost = calculate_cost()
        
        if i%10 == 0 and batch_num==num_of_batches-1:
            zs,acts = forward_pass(xdata_full,weights,biases,relu)
            cost = calculate_cost(acts,ydata_full,ny_data,nx_data,weights)
            print("cost at iteration",i,cost)
            acts_valid = predict(x_valid,weights,biases,relu)
            cost_valid = calculate_cost(acts_valid,y_valid,ny_valid,nx_data,weights)
            #print("cost at iteration",i,denn(cost,max_elem,min_elem))
            
            #print("validation cost at iteration",i,denn(cost_valid,max_elem,min_elem))
            print("validation cost at iteration",i,cost_valid)
    zs,acts = forward_pass(xdata_full,weights,biases,relu)      
    cost = calculate_cost(acts,ydata_full,ny_data,nx_data,weights)
    costs_map.append(cost)
    
acts_test = predict(xtest,weights,biases,relu)
cost_test = calculate_cost_test(acts_test,ytest,ny_test,nx_data)
print("test error - ",cost_test)

iter_map = []
for ee in range(e) :
    iter_map.append(ee)
plt.plot(iter_map,costs_map)
plt.show()
with open('6.2.cost_convergence_cap.pkl','rb') as f:
    x = pickle.load(f)
    print(x.shape)

plt.plot(iter_map,x,"blue")
plt.plot(iter_map,costs_map,"red")
plt.gca().legend(( 'weight cap', 'l2'))
plt.show()
cost_convergence = np.array(costs_map)
with open('6.2.cost_convergence_l2.pkl','wb') as f:
    pickle.dump(cost_convergence, f)


 

    

