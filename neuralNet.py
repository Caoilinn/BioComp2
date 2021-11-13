import numpy as np

INERTIA_WEIGHT = 0.5
COGNITIVE_WEIGHT = 0.5
SOCIAL_WEIGHT = 0.5
NUM_PARTICLES = 10

class NN:
    
    input_layer_size = 1
    hidden_layer_size = [1]


    
    def __init__(self, input_layer_size, hidden_layer_nodes):

        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_nodes

        self.weights = self.init_weights()
        self.outputs = []

    def init_weights(self):
        all_sorted_weights = []
        for i in range(len(self.hidden_layer_size)):
        
            # Weights are random values set in a matrix the size of the input at a given layer
            #TODO: Find a better way to randomly set weights
            if(i == 0):
                weights_hidden = np.random.random(self.input_layer_size, self.hidden_layer_size[i]) 
            else:
                weights_hidden = np.random.random(self.hidden_layer_size[i-1], self.hidden_layer_size[i])

            all_sorted_weights.append(weights_hidden)
        
        return all_sorted_weights



    def update_vel(curr_vel, curr_pos, loc_best_pos, glob_best_pos):
        return (INERTIA_WEIGHT * curr_vel) + (COGNITIVE_WEIGHT * (loc_best_pos - curr_pos)) + (SOCIAL_WEIGHT * (glob_best_pos - curr_pos))

    def update_pos(curr_pos, updated_vel):
        return curr_pos + updated_vel