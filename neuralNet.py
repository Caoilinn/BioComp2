import numpy as np

INERTIA_WEIGHT = 0.5
COGNITIVE_WEIGHT = 0.5
SOCIAL_WEIGHT = 0.5
NUM_PARTICLES = 10

class NN:   
        
    def __init__(self, input_layer_size, hidden_layer_nodes, inputs):

        #Used to setup initial weights - architecture of the NN        
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_nodes
        self.init_vel()
        self.weights = self.init_weights()
        
        #Initially set the best position to the initial random position
        self.best_position = self.weights
        
        

        self.inputs = inputs
        self.outputs = []

    def init_weights(self):
        all_sorted_weights = []
        for i in range(len(self.hidden_layer_size)):
        
            # Weights are random values set in a matrix the size of the input at a given layer
            #TODO: Find a better way to randomly set weights
            if(i == 0):
                #print("Input Layer Size: ", self.input_layer_size)
                #print("Hidden Layer Nodes: ", self.hidden_layer_size[i])
                weights_hidden = np.random.rand(self.input_layer_size, self.hidden_layer_size[i]) 
            else:
                #print("Input Layer Size: ", self.input_layer_size)
                #print("Hidden Layer Nodes: ", self.hidden_layer_size[i])
                weights_hidden = np.random.rand(self.hidden_layer_size[i-1], self.hidden_layer_size[i])

            all_sorted_weights.append(weights_hidden)
        
        return all_sorted_weights
    
    def init_vel(self):
        self.current_vel = np.random.rand(1, 2) 

    def update_vel(curr_vel, curr_pos, loc_best_pos, glob_best_pos):
        return (INERTIA_WEIGHT * curr_vel) + (COGNITIVE_WEIGHT * (loc_best_pos - curr_pos)) + (SOCIAL_WEIGHT * (glob_best_pos - curr_pos))

    def update_pos(curr_pos, updated_vel):
        return curr_pos + updated_vel

    #Activation
    def sigmoid(self, X, weights):
        z = np.dot(X, weights)
        return (1/ (1 + np.exp(-z)))

    #Loss Function
    def cross_entropy(self, y, y_hat):
        return -(np.multiply(y, np.log(y_hat)) + np.multiply((1-y), np.log(1-y_hat)))

    def forward_pass(self):
        inputs = self.inputs
        self.activations = []

        for i in range (len(self.hidden_layer_size)):
            A = self.sigmoid(inputs, self.weights[i])
            self.activations.append(A)
            inputs  = A

        A_output = self.sigmoid(inputs, self.weights[-1])
        self.activations.append(A_output)
        self.outputs = A_output