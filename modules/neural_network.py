import numpy as np
from math import *
from random import *
import pickle
import pandas as pd

def average(matrix):
    count = 1
    S = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            for h in range(len(matrix[i][j])):
                S += matrix[i][j][h]
                count += 1
    return S/count




#a modifier
def sigmoid(x):
    x = np.clip(x, -500, 500)  # Limiter les valeurs de x entre -500 et 500 pour éviter les débordements
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def sigmoid_derivative_vect(x):
    a = []
    for i in x:
        a.append(sigmoid_derivative(i))
    return np.array(a)


class Activation_function:
    """
    Class defining the structure of the activation functions.

    self.f: function
        function corresponding to the activation function
    self.df: function
        derived function of the activation function
    """
    def __init__(self):
        self.name = None
        self.f = None
        self.df = None


class Sigmoid(Activation_function):
    def __init__(self):
        Activation_function.__init__(self)
        self.name = "sigmoid"
        self.f = sigmoid
        self.df = sigmoid_derivative
        self.df_vect = lambda x: np.array([np.exp(-x[i])/(1+np.exp(-x[i]))**2 for i in range(len(x))])
        self.df_vect = sigmoid_derivative_vect

class Loss_function:
    """
    Class defining the structure of the loss functions.

    self.f: function
        function corresponding to the loss function
    self.df: function
        derived function of the loss function
    """
    def __init__(self):
        self.name = None
        self.f = None
        self.df = None

class Distance(Loss_function):
    """
    self.f: function
        function corresponding to the loss function
    self.df: function
        derived function of the activation function
    """
    def __init__(self):
        Loss_function.__init__(self)
        def f(y, t):
            y = y[0]
            t = t[0]
            real_t = np.log(t/(1-t))
            d = y-real_t
            return d
        
        def df(y,t):
            y = y[0]
            t = t[0]
            d = 1/t/(t-1)
            return [d]
        self.name = "Ltest"
        self.f = f
        self.df = df


class Distance_L1(Loss_function):
    """
    self.f: function
        function corresponding to the loss function
    self.df: function
        derived function of the activation function
    """
    def __init__(self):
        Loss_function.__init__(self)
        def f(y, t):
            a = 0
            l = len(t)
            for i in range(l):
                a += abs(y[i]-t[i])
            a/= l
            return a
        
        def df(y,t):
            a = []
            l = len(y)
            for i in range(l):
                if y[i]-t[i] > 0:
                    a.append(-1/l)
                else:
                    a.append(+1/l)
            return a
        self.name = "L1"
        self.f = f
        self.df = df

class Distance_L2(Loss_function):
    """
    self.f: function
        function corresponding to the loss function
    self.df: function
        derived function of the activation function
    """
    def __init__(self):
        Loss_function.__init__(self)
        def f(y, t):
            a = 0
            l = len(t)
            for i in range(l):
                a += (y[i]-t[i])**2
            a /= l
            return a
        
        def df(y,t):
            a = []
            l = len(y)
            for i in range(l):
                a.append(2*(t[i]-y[i])/l)
            return a
        self.name = "L2"
        self.f = f
        self.df = df


class Neural_network():
    """
    Class representing a neural network consisting of many neuron layers.
    The network is represented as a list of neuron_layer class objects.

    N:
        number of layer in the neural network
    entry_size_list: int list
        List containing the number of entry for each neuron layer, with the number of output at the end.
        It thus contains N+1 elements.
    activation_function_list: Activation_function list
        List of length N that gives the activation function associated to each neuron layer.
    loss_function: Loss_function
        function representing the distance between the expected result and the output of the neural network
    """
    def __init__(self, entry_size_list:list, activation_function_list:list, loss_function:Loss_function, column_titles_list, output_column):
        self.N = len(activation_function_list)
        self.entry_size_list = np.array(entry_size_list)
        self.activation_function_list = activation_function_list
        self.loss_function = loss_function

        self.activation_function_list = activation_function_list
        self.weigth_matrix_list = [0]*self.N
        self.bias_list = np.zeros(self.N)
        for k in range(self.N):
            self.bias_list[k] = uniform(-1,1)
            self.weigth_matrix_list[k] = np.array([[uniform(-5, 5) for i in range(entry_size_list[k])] for j in range(entry_size_list[k+1])])


        self.column_titles_list = column_titles_list
        self.output_column = output_column

        #the variables below are initialised when we start the teaching methods of the neural network
        self.data = None
        self.data_size = -1
        self.learning_factor = 1

        #the variables below depend of the data analysed, and are thus changed when a new entry is analysed
        self.entry = None
        self.expected_output = None
        self.output = None
        self.H_list = [np.zeros(self.entry_size_list[i+1]) for i in range(self.N)]
        self.Z_list = [np.zeros(self.entry_size_list[i]) for i in range(self.N+1)]
        self.dLdZ_list = [np.zeros(entry_size_list[i+1]) for i in range(self.N)]
        self.dLdW_list = [np.zeros((entry_size_list[i+1], entry_size_list[i])) for i in range(self.N)]
        self.dBias_list = np.zeros(self.N)

    def initialise_training(self, data, learning_factor:float):
        self.data = data
        self.data_size = len(data)
        self.learning_factor = learning_factor


    def initialise_training(self, data, learning_factor:float):
        self.data = data
        self.data_size = len(data)
        self.learning_factor = learning_factor
    

    def treatment(self, entry:list):
        """
        Given an entry to the neural network, return the output given by the last layer of neurons.
        """
        for i in range(self.N):
            output = np.zeros(self.entry_size_list[i+1])
            for j in range(self.entry_size_list[i+1]):
                X = self.bias_list[i]
                X += + np.sum(self.weigth_matrix_list[i][j]*entry)
                output[j] = self.activation_function_list[i].f(X)
            entry = output
        return output

    def real_treatment(self, entry:list):
        for i in range(self.N):
            output = np.zeros(self.entry_size_list[i+1])
            for j in range(self.entry_size_list[i+1]):
                X = self.bias_list[i]
                X += + np.sum(self.weigth_matrix_list[i][j]*entry)
                output[j] = self.activation_function_list[i].f(X)
            entry = output
        output = output[0]
        output = np.log(output/(1-output))
        return output

    def forward_processing(self):
        """
        Given an entry to the neural network, return two matrix H_list and Z_list.
        Z_list consist of the list of each input given to the successive layers, with the final output at the end.
        H_list consist of the output of each neuron layer after the dot product, before we apply the activation function
        """

        self.Z_list[0] = self.entry
        for i in range(self.N):
            for j in range(self.entry_size_list[i+1]):
                X = self.bias_list[i]
                X += np.sum(self.weigth_matrix_list[i][j]*self.Z_list[i])
                self.H_list[i][j] = X
                self.Z_list[i+1][j] = self.activation_function_list[i].f(X)
        self.output = self.Z_list[-1]


    def get_data(self, i:int):
        """
        actualise the attributes that depends of the piece of data we are working on
        """
        self.entry = self.data[self.column_titles_list].iloc[i].tolist()
        self.expected_output = self.data[self.output_column].iloc[i].tolist()


    def backpropagation(self):
        """
        Backpropagation in order to modify the weigths of the neural network in order to improve it.
        """
        entry_size_list = self.entry_size_list
        loss_function = self.loss_function
        learning_factor = self.learning_factor
        H_list = self.H_list
        Z_list = self.Z_list
        output = self.output
        expected_output = self.expected_output
        n_layer = self.N
        dLdZ_list = self.dLdZ_list
        dLdW_list = self.dLdW_list
        dBias_list = self.dBias_list

        layer_size_list = [len(Z_list[i+1]) for i in range(n_layer)]

        real_output =[np.log(output[0]/(1-output[0]))]

        loss = loss_function.f(expected_output, output)

        aze = 12
        if aze == 1:
            print("expected output réel", expected_output[0], "\n")
            print("output", real_output[0], "\n")
            print("loss", loss, "\n")
            input()

        dLdZ_list[-1] = loss_function.df(expected_output, output)

        for k in range(n_layer-1, 0, -1):
            for i in range(layer_size_list[k-1]):
                dLdZ_list[k-1][i] = np.sum(dLdZ_list[k]*self.activation_function_list[k].df_vect(H_list[k])*self.weigth_matrix_list[k][:,i])

        for k in range(n_layer):
            dLdBias_k = 0
            for i in range(entry_size_list[k]):
                for j in range(entry_size_list[k+1]):    
                    x1 = dLdZ_list[k][j]
                    x2 = self.activation_function_list[k].df(H_list[k][j])
                    x3 = Z_list[k][i]
                    dLdW_ijk = x1*x2*x3
                    dLdW_list[k][j][i] = dLdW_ijk

            for j in range(entry_size_list[k+1]):
                #print("H_list[k][j]:", H_list[k][j])
                dLdBias_k += dLdZ_list[k][j] * self.activation_function_list[k].df(H_list[k][j])
            dBias_list[k] = dLdBias_k


        for k in range(n_layer):
            self.bias_list[k] -= learning_factor * dBias_list[k] * loss
            for i in range(entry_size_list[k]):
                for j in range(entry_size_list[k+1]):
                    self.weigth_matrix_list[k][j][i] -= learning_factor * dLdW_list[k][j][i] *loss

        self.sum_loss += abs(loss)


    def epoch(self):
        """
        Apply the backpropagation method to every data piece of the data set, in a random order.
        """      
        self.sum_loss = 0
        l = [i for i in range(0, self.data_size)]
        shuffle(l)

        for i in l:
            self.get_data(i)
            self.forward_processing()
            self.backpropagation()
        

    def epoch_while(self, data, learning_factor, score, max_epoch):
        self.initialise_training(data, learning_factor)
        self.sum_loss = score*self.data_size + 1
        n = 0
        while abs(self.sum_loss/self.data_size) > score and n < max_epoch:
            n += 1
            self.m = 0
            self.epoch()
            print("\nmean_loss", self.sum_loss/self.data_size, "\n")
            print(n)

    def train_n_epoch(self, n:int, data, learning_factor:float):
        self.initialise_training(data, learning_factor)
        for i in range(n):
            self.epoch()
            print(i+1)
            print("\nmean_loss", self.sum_loss/self.data_size, "\n")
            self.m = 0




    def save(self, path:str):
        """
        Method saving the essencial attributes needed to know the neuron network throught pickle.
        The saved attributes are:
            entry_size_list
            the weight_matrix of each neuron layer
            the bias of each neuron layer
            the activation function of each layer
            learning_factor
            loss_function

        """
        with open(path, 'wb') as f1:
            to_save = []
            to_save.append(self.entry_size_list)
            to_save.append(self.weigth_matrix_list)
            to_save.append(self.bias_list)
            to_save.append([self.activation_function_list[i].name for i in range(self.N)])
            to_save.append(self.loss_function.name)
            to_save.append(self.column_titles_list)
            to_save.append(self.output_column)
            pickle.dump(to_save, f1)


    @classmethod
    def load(cls, path:str):
        """
        Class method returning a saved neural network, which location is given by path.
        The data is saved and loaded through pickle
        """
        with open(path, 'rb') as f1:
            saved_network = pickle.load(f1)
        entry_size_list = saved_network[0]
        weigth_matrix_list = saved_network[1]
        bias_list = saved_network[2]

        activation_function_list = []
        for activation_function_name in saved_network[3]:
            if activation_function_name == "sigmoid":
                activation_function_list.append(Sigmoid())

        if saved_network[4] == "Ltest":
            loss_function = Distance()
        if saved_network[4] == "L1":
            loss_function = Distance_L1()
        if saved_network[4] == "L2":
            loss_function = Distance_L2()

        column_titles_list = saved_network[5]
        output_column = saved_network[6]

        network = Neural_network(entry_size_list, activation_function_list, loss_function, column_titles_list, output_column)
        network.weigth_matrix_list = weigth_matrix_list
        network.bias_list = bias_list
        network.activation_function_list = activation_function_list
        network.loss_function = loss_function
        return network

    def __repr__(self):
        a = str(self.N) + " layers of neuron\n" + self.loss_function.name + " loss function\n"
        return a

def create_new_network(path, loss_function:Loss_function, size_list:list , function_list: list, column_titles_list, output_column):
    """
    Given some parameters, create a network and save it at a given path
    """
    network = Neural_network(size_list, function_list, loss_function, column_titles_list, output_column)
    network.save(path)

def copy_network(path_1:str, path_2:str):
    """
    given path_1 and path_2 two paths, create a copy of the network at path_1 on the location given by path_2
    """
    network = Neural_network.load(path_1)
    network.save(path_2)

def train_network(network_path:str, data, n_epoch:int = 1, learning_factor:float = 1):
    """
    Given the path where the network is saved, the path where the data is saved, a number of epoch and a learning factor,
    train the neural network for n_epoch epochs, by doing backpropagation with a gradient descent step equal to learning_factor.
    At the end of the n_epochs epochs, save the network at the same path, replacing the original one.
    """
    working_network = Neural_network.load(network_path)
    working_network.train_n_epoch(n_epoch, data, learning_factor)
    working_network.save(network_path)

def train_network_score(network_path:str, data, score:int, n_max:int = 20, learning_factor:float = 1):
    """
    Given the path where the network is saved, the path where the data is saved, a number of epoch and a learning factor,
    train the neural network for n_epoch epochs, by doing backpropagation with a gradient descent step equal to learning_factor.
    At the end of the n_epochs epochs, save the network at the same path, replacing the original one.
    """
    working_network = Neural_network.load(network_path)
    working_network.epoch_while(data, learning_factor, score, n_max)
    working_network.save(network_path)

def apply_network(network_path:str, data):
    working_network = Neural_network.load(network_path)
    data["Neural prediction"] = data.apply(lambda row: working_network.real_treatment(row[working_network.column_titles_list].tolist()), axis = 1)

"""
original_data = pd.read_csv("original_data.csv")
original_data.to_csv("working_data.csv", index = False)
data = pd.read_csv("working_data.csv")


data = data[(data["Rareté"] != "Online Code Card") & (data["Rareté"] != "Oversized") & (data["Rareté"] != "Fixed")]
data = data.dropna(subset=["Min price"])
data["Min price"] = (data['Min price'].str[:-5] + data['Min price'].str[-4:-2]).astype(int)/100
data = data[(data["Min price"] >= 0.5)]


data[["Name", "Code"]] = data["Name"].str.split(" \(", expand = True)




sub_data = data.dropna().copy()

sub_data["Price trend"] = (sub_data["Price trend"].str[:-5] + sub_data["Price trend"].str[-4:-2]).astype(int)/100
sub_data["Price 7 days"] = (sub_data["Price 7 days"].str[:-5] + sub_data["Price 7 days"].str[-4:-2]).astype(int)/100
sub_data["Price 30 days"] = (sub_data["Price 30 days"].str[:-5] + sub_data["Price 30 days"].str[-4:-2]).astype(int)/100
order = ["Index", "Name", "Expansion", "Rareté", "Min price", "Price trend", "Price 7 days", "Price 30 days", "Tournament_last_month"]
sub_data = sub_data[order]




d = {}
l = sub_data["Expansion"].unique()
for i in range(len(l)):
    expansion = l[i]
    d[expansion] = i
sub_data["Expansion number"] = sub_data["Expansion"].map(d)

d = {}
l = sub_data["Rareté"].unique()
for i in range(len(l)):
    rarity = l[i]
    d[rarity] = i
sub_data["Rareté number"] = sub_data["Rareté"].map(d)

column_titles_list = ["Tournament_last_month", "Price 30 days", "Price 7 days", "Min price"]
output_column = ["Price trend"]

size_list = [len(column_titles_list), 20, len(output_column)]
function_list = []
for i in range(len(size_list)-1):
    function_list.append(Sigmoid())


path = "neural_network"
create_network_path = path + "/original_network"
working_network_path = path + "/working_network"

#create_new_network(path = create_network_path, loss_function = Distance(), size_list = size_list , function_list = function_list, column_titles_list= column_titles_list, output_column= output_column)
#copy_network(create_network_path, working_network_path)


#train_network_score(working_network_path, sub_data, score = 0.15, n_max= 20, learning_factor= 0.000001)

"""