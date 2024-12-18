import numpy as np
from math import *
from random import *
import pickle
import time


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
        self.f = lambda x: 1/(1+np.exp(-x))
        self.df = lambda x: np.exp(-x)/(1+np.exp(-x))**2
        self.df_vect = lambda x: np.array([np.exp(-x[i])/(1+np.exp(-x[i]))**2 for i in range(len(x))])


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

class Distance_L4(Loss_function):
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
                a += (y[i]-t[i])**4
            a/= l
            return a
        
        def df(y,t):
            a = []
            l = len(y)
            for i in range(l):
                t1 = t[i]
                t2 = t[i]**2
                t3 = t[i]**3
                y1 = y[i]
                y2 = y[i]**2
                y3 = y[i]**3
                a.append((4*t3 + 12*t2*y1 + 12*t1*y2 + 4*y3)/l)
            return a
        self.name = "L1"
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
    neuron_layer_list: neuron_layer list
        List of neuron_layer that represents the neural network
    loss_function: Loss_function
        function representing the distance between the expected result and the output of the neural network
    """
    def __init__(self, entry_size_list:list, activation_function_list:list, loss_function:Loss_function):
        self.N = len(activation_function_list)
        self.entry_size_list = np.array(entry_size_list)
        self.activation_function_list = activation_function_list
        self.neuron_layer_list = []
        self.loss_function = loss_function

        self.activation_function_list = activation_function_list
        self.weigth_matrix_list = [0]*self.N
        self.bias_list = np.zeros(self.N)
        for k in range(self.N):
            self.bias_list[k] = uniform(-1,1)
            self.weigth_matrix_list[k] = np.array([[uniform(-3, 3) for i in range(entry_size_list[k])] for j in range(entry_size_list[k+1])])


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


        #the variables below are only used to test things and don't impact the functions
        self.sum_loss = 0
        self.m = 0
        self.T1 = 0
        self.T2 = 0
        self.T3 = 0
        self.T4 = 0
        self.T5 = 0
        self.T6 = 0
        self.T7 = 0

    def initialise_training(self, data:list, learning_factor:float):
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
        self.entry = self.data[i-1][0]
        self.expected_output = self.data[i-1][1]

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
        loss = loss_function.f(expected_output,output)

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
                dLdBias_k += dLdZ_list[k][j] * self.activation_function_list[k].df(H_list[k][j])
            dBias_list[k] = dLdBias_k

            a = 21
            if a == 1:
                dLdBias_k = dLdZ_list[k][j] * self.activation_function_list[k].df(H_list[k][j]) * 1
                dBias_list[k] = dLdBias_k



        for k in range(n_layer):
            self.bias_list[k] -= learning_factor * dBias_list[k] * loss
            for i in range(entry_size_list[k]):
                for j in range(entry_size_list[k+1]):
                    self.weigth_matrix_list[k][j][i] -= learning_factor * dLdW_list[k][j][i] *loss

        self.sum_loss += loss


    def epoch(self):
        """
        Apply the backpropagation method to every data piece of the data set, in a random order.
        """
        self.sum_loss = 0
        l = [i for i in range(1, self.data_size+1)]
        shuffle(l)

        for i in l:
            self.get_data(i)
            self.forward_processing()
            self.backpropagation()
        print("\nsum_loss", self.sum_loss, "\n")

    def train_n_epoch(self, n:int, data:list, learning_factor:float):
        self.initialise_training(data, learning_factor)
        for i in range(n):
            self.epoch()
            print(i+1)
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

        if saved_network[4] == "L1":
            loss_function = Distance_L1()
        if saved_network[4] == "L2":
            loss_function = Distance_L2()
        if saved_network[4] == "L4":
            loss_function = Distance_L4()

        network = Neural_network(entry_size_list, activation_function_list, loss_function)
        network.weigth_matrix_list = weigth_matrix_list
        network.bias_list = bias_list
        network.activation_function_list = activation_function_list
        network.loss_function = loss_function
        return network



    def __repr__(self):
        a = str(self.N) + " layers of neuron\n" + self.loss_function.name + " loss function\n"
        return a


class Networker_tester():
    def __init__(self, data, neural_network, test_function):
        self.path = path
        self.neural_network = neural_network
        self.data = data
        self.data_size = len(data)
        print(self.data_size)

        #the variables below depend of the data analysed, and are thus changed when a new entry is analysed
        self.working_entry = None
        self.expected_output = None
        self.entry = None
        self.output = None
        #the variables below are only used to test things and don't impact the functions
        self.correct_count = 0
        self.false_count = 0

        self.test_function = test_function

    def get_data(self, i:int):
        self.entry = self.data[i-1][0]
        self.expected_output = self.data[i-1][1]

    def test(self):
        self.correct_count = 0
        self.false_count = 0
        for i in range(1, self.data_size + 1):
            self.get_data(i)
            t = self.neural_network.treatment(self.entry)
            y = self.expected_output
            t1 = self.test_function(t)
            y1 = self.test_function(y)

            if t1 == y1:
                self.correct_count += 1
            else:
                self.false_count += 1

        return (self.correct_count, self.false_count)



#We create the neural network

def get_number(x:list):
    """
    get the index of the biggest value of the list x
    """
    l = len(x)
    k = 0
    max = x[0]
    for i in range(l):
        if x[i] > max:
            max = x[i]
            k = i
    return k

def create_new_network(path, loss_function:Loss_function, size_list:list , function_list: list):
    """
    given some parameters, create a network and save it at a given path
    """
    network = Neural_network(size_list, function_list, loss_function)
    network.save(path)

def copy_network(path_1:str, path_2:str):
    """
    given path_1 and path_2 two paths, create a copy of the network at path_1 on the location given by path_2
    """
    network = Neural_network.load(path_1)
    network.save(path_2)

def train_network(network_path:str, training_data_path:str, n_epoch:int, data_size = -1, learning_factor:float = 1):
    """
    Given the path where the network is saved, the path where the data is saved, a number of epoch and a learning factor,
    train the neural network for n_epoch epochs, by doing backpropagation with a gradient descent step equal to learning_factor.
    At the end of the n_epochs epochs, save the network at the same path, replacing the original one.
    """
    working_network = Neural_network.load(network_path)
    data = get_data(training_data_path, n = data_size)
    working_network.train_n_epoch(n_epoch, data, learning_factor)
    working_network.save(network_path)

def test_network(path_data:str, path_network:str):
    """
    Given the path where a network is saved and the path where the testing data is saved,
    show the percentage of good answer as well as the total number of answer given by the network.
    """
    working_network = Neural_network.load(path_network)
    D = get_data(path_data)
    U = Networker_tester(D, neural_network = working_network, test_function = get_number)
    t,f = U.test()
    n = t + f
    print(t, " valeurs correctement prédites\n", f, " valeurs incorrectement prédites\n", n, " valeurs au total\n", "Soit une proportion de bonne réponse de ", t/n, "!")


def get_entry_from_png_path(png_path:str):
    img = Image()
    img.load(png_path)
    entry = img.to_neuron_entry()
    return entry

def get_output(network, entry):
    output = network.treatment(entry)
    number = get_number(output)
    return number

def get_output_2(path_network, path_png):
    working_network = Neural_network.load(path_network)
    from convert_png import Image
    img = Image()
    img.load(path_png)
    entry = img.to_neuron_entry()
    img.plot()
    output = working_network.treatment(entry)
    number = get_number(output)
    return number
 
def reconaissance(img:Image, network:Neural_network, goal_mean_colour:float, step:float, bias:float = 0):
    goal_mean_colour = goal_mean_colour + bias
    if goal_mean_colour - step < 0:
        print("ca peut pas marcher...")
    mean_colour = img.get_mean_white()
    l = []
    while mean_colour > goal_mean_colour - step:
        if mean_colour < goal_mean_colour + step:
            entry = img.to_neuron_entry()
            l.append(get_output(network, entry))
        img.thicken_black(1)
        mean_colour = img.get_mean_white()
    l_count = [0]*10
    
    if len(l) == 0:
        return "step is too small !"
    
    for i in l:
        l_count[i] += 1
    i_max = 0
    iter_max = 0
    for j in range(10):
        if l_count[j] > iter_max:
            i_max = j
            iter_max = l_count[j]
    return i_max







size_list = [256, 10, 10]
function_list = []
for i in range(len(size_list)-1):
    function_list.append(Sigmoid())


size = 5832
mean_white = 0.7451475671517215

path = "neural_network_1"
create_network_path = path + "/original_network"
working_network_path = path + "/working_network"
working_network_path2 = path + "/working_network2"
testing_data_path = "data/testing_data.xlsx"
training_data_path = "data/training_data.xlsx"

#create_new_network(path = create_network_path, loss_function = Distance_L2(), size_list = size_list , function_list = function_list)
#copy_network(create_network_path, working_network_path)


#train_network(working_network_path, training_data_path, n_epoch = 40, data_size = 5832, learning_factor = 3)


test_network(testing_data_path, working_network_path)

#train_network(working_network_path2, training_data_path,  n_epoch = 10, data_size = 1000, learning_factor = 1)
#test_network(testing_data_path, working_network_path2)

#copy_network(path + "/network_1", path + "/save_network")


#a = Neural_network.load(working_network_path)
#a.save(path + "/original_network")



#test_network(training_data_path, working_network_path)





a = 0
if a == 1:
    for i in range(10):
        img = Image()
        img.load("image/collection_4/" + str(i) + ".png")
        network = Neural_network.load(working_network_path)
        r = reconaissance(img, network, mean_white, 0.15, 0)
        print(i, r)



a = 0
if a == 1:
    for i in range(7,9):
        img = Image()
        img.load("Number_detection/image/collection_3/" + str(i) + ".png")
        img.squared()
        img.color_to_gray()
        for i in range(20):
            img.save_copy(i)
            img.thicken_black(1)



a = 0
if a == 1:
    n = 20
    network = Neural_network.load(working_network_path)
    for j in range(4, 5):
        print("on test les ", j)
        for i in range(n):
            png_path = "Number_detection/image/collection_3/"+ str(j) + "_" + str(i) + ".png"
            img = Image()
            img.load(png_path)
            mean_white = img.get_mean_white()
            entry = get_entry_from_png_path(png_path)
            output = get_output(network, entry)
            print(i, j, output, mean_white)




#sert à récupérer les données de la base de données sous la forme de png
a = 0
if a == 1:
    D = get_data(training_data_path)
    for i in range(len(D)):
        print(i)
        vector = np.array(D[i][0])
        img = Image.vector_to_img(vector, 16,16)
        path_png = "Number_detection/image/data_as_png/" + str(i) + ".png"
        img.save(path_png)

#permet de récupérer mean_black moyen sur l'échantillon d'apprentissage
a = 0
if a == 1:
    count = 0
    for i in range(size):
        img = Image()
        img.load("Number_detection/image/data_as_png/" + str(i) + ".png")
        count += img.get_mean_white()
    count /= size
    print(count)


a = 0
if a == 1:
    n = 30
    D = get_data(testing_data_path)
    network = Neural_network.load(working_network_path)
    for i in range(n):
        vector = np.array(D[i][0])
        expected_output = get_number(np.array(D[i][1]))
        output = get_output(network, vector)

        img = Image.vector_to_img(vector, 16, 16)
        path_png = "Number_detection/image/data_as_png/" + str(i) + ".png"
        img.save(path_png)

        img = Image()
        img.load(path_png)
        vector_2 = img.to_neuron_entry()
        output_2 = get_output(network, vector_2)
        print(expected_output, output, output_2)




###
a = 0
if a == 1:
    n = 6
    y = [i for i in range(n)]
    t = []
    network = Neural_network.load(working_network_path)
    for i in range(n):
        png_path = "Number_detection/image/collection_2/" + str(i) + ".png"
        entry = get_entry_from_png_path(png_path)
        output = get_output(network, entry)
        print(i, output)







a = 0
if a == 1:
    n = 6
    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    t = []
    for i in range(n):
        path_png = "/image/collection_2" + "/" + str(i) + ".png"
        a = get_output(working_network_path, path_png)
        print(a, i)
        t.append(a)
    error = 0
    for x in range(n):
        if y[x] != t[x]:
            error += 1
    print("erreurs: ", error)



