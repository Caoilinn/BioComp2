import numpy as np
import sys

INERTIA_WEIGHT = 0.5
COGNITIVE_WEIGHT = 0.5
SOCIAL_WEIGHT = 0.5
NUM_PARTICLES = 10

class NN:   
        
    def __init__(self, input_layer_size, hidden_layer_nodes, inputs):

        #Used to setup initial weights - architecture of the NN        
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_nodes
        
        #Initialise Velocity and Weights(position)
        self.init_vel()
        self.weights = self.init_weights()
        self.output_weights()
        
        #Initially set the best position to the initial random position
        self.best_position = self.weights
        self.best_error = sys.float_info.max
        self.inputs = inputs

    def init_weights(self):
        all_sorted_weights = []
        for i in range(len(self.hidden_layer_size)):
        
            # Weights are random values set in a matrix the size of the input at a given layer
            #TODO: Find a better way to randomly set weights
            if(i == 0):
                weights_hidden = np.random.rand(self.input_layer_size, self.hidden_layer_size[i]) 
            else:
                weights_hidden = np.random.rand(self.hidden_layer_size[i-1], self.hidden_layer_size[i]) 

            all_sorted_weights.append(weights_hidden)
        
        return all_sorted_weights
    
    def init_vel(self):
        
        velocities = []
        for i in range(len(self.hidden_layer_size)):
        
            # Weights are random values set in a matrix the size of the input at a given layer
            #TODO: Find a better way to randomly set weights
            if(i == 0):
                velocities_hidden = np.random.rand(self.input_layer_size, self.hidden_layer_size[i]) 
            else:
                velocities_hidden = np.random.rand(self.hidden_layer_size[i-1], self.hidden_layer_size[i]) 

            velocities.append(velocities_hidden)

        velocities_output = np.random.rand(self.hidden_layer_size[-1], 1)
        velocities.append(velocities_output)

        self.current_vel = velocities


    def update_vel(curr_vel, curr_pos, loc_best_pos, glob_best_pos):
        return (INERTIA_WEIGHT * curr_vel) + (COGNITIVE_WEIGHT * (loc_best_pos - curr_pos)) + (SOCIAL_WEIGHT * (glob_best_pos - curr_pos))

    def update_pos(curr_pos, updated_vel):
        return curr_pos + updated_vel

    #Activation
    def sigmoid(self, X, weights):
        z = np.dot(X, weights)
        return (1/ (1 + np.exp(-z)))
  
    def output_weights(self):   
        weights_output = np.random.rand(self.hidden_layer_size[-1], 1)
        self.weights.append(weights_output)

    def check_best_error(self):
        if(self.error < self.best_error):
            self.best_position = self.weights

    #Loss Function
    def cross_entropy(self, y):
        self.error =  np.average(-(np.multiply(y, np.log(self.output)) + np.multiply((1-y), np.log(1-self.output))))
        self.check_best_error()

    def forward_pass(self):
        inputs = self.inputs
        self.activations = []

        for i in range (len(self.hidden_layer_size)):
            A = self.sigmoid(inputs, self.weights[i])
            self.activations.append(A)
            inputs  = A

        #Setup output_weights
        
        A_output = self.sigmoid(inputs, self.weights[-1])
        
        self.activations.append(A_output)
        self.output = A_output