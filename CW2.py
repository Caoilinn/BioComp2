import numpy as np
import pandas as pd
import neuralNet
from neuralNet import NN

dataset_inputs = pd.read_csv(
    './CW2/data_banknote_authentication.txt', delimiter=',', header=None, usecols=[0, 1, 2, 3])
dataset_inputs.columns = ["Input 1", "Input 2", "Input 3", "Input 4"]

dataset_outputs = pd.read_csv(
    './CW2/data_banknote_authentication.txt', delimiter=',', header=None, usecols=[4])
dataset_outputs.columns = ["Outputs"]

#Output matrix or "y"
output_matrix = np.matrix(dataset_outputs, dtype=None, copy=True)
inputMatrix = np.matrix(dataset_inputs, dtype=None, copy=True)

# Neural Network architecture
arch = {
    'input_layer_size' : inputMatrix.shape[1],
    'hidden_layer_nodes' : (4,2,2),
    'output_layer_nodes' : 1,
    'iterations': 5000,
    'learning_rate': 0.001
}

all_sorted_weights = []
all_layers_activations = []
def initial_weights(iteration,layer):

    if(iteration == 0):
        # Weights are random values set in a matrix the size of the input at a given layer
        #TODO: Find a better way to randomly set weights
        if(layer == 0):
            weights_hidden = np.random.random((arch['input_layer_size'], arch['hidden_layer_nodes'][layer])) 
        else:
            weights_hidden = np.random.random((arch['hidden_layer_nodes'][layer-1], arch['hidden_layer_nodes'][layer]))

        all_sorted_weights.append(weights_hidden) 
      
def outputs(iteration): 
    if (iteration == 0):
        weights_output = np.random.random((arch['hidden_layer_nodes'][-1], arch['output_layer_nodes']))
        all_sorted_weights.append(weights_output)
    
def sigmoid(X, weights):
    z = np.dot(X, weights)
    z = np.array(z, dtype=np.float128)
    return (1/ (1 + np.exp(-z)))

def sigmoid_derr(val):
    return np.multiply(val, 1-val)

def tanH(X, weights):
    z = np.dot(X, weights) 
    return ((np.exp(z) - np.exp(-z) / (np.exp(z) + np.exp(-z))))

def tanH_derr(val):
    return 1 - np.square(val) 

def relu(X):
    return np.maximum(0, X)

def relu_derr(val):
    if(val <= 0):
        return 0
    
def cross_entropy(y, y_hat):
    return -(np.multiply(y, np.log(y_hat)) + np.multiply((1-y), np.log(1-y_hat)))

def forward_pass(iteration):
    input = inputMatrix
    for i in range(len(arch['hidden_layer_nodes'])):
        
        initial_weights(iteration,i)
        A = sigmoid(input, all_sorted_weights[i])
        all_layers_activations.append(A)
        input = A
    outputs(iteration)
    A_output = sigmoid(input,all_sorted_weights[-1])
    print("Weights: ", all_sorted_weights)
    all_layers_activations.append(A_output)
    return A_output

def backwards_pass(iteration):
    forward = forward_pass(iteration)
    error = cross_entropy(output_matrix, forward)

    dz = forward - error
    dw = np.dot(np.transpose(dz), all_layers_activations[-2])
    
    all_sorted_weights[-1] = arch['learning_rate'] * np.subtract(all_sorted_weights[-1], np.transpose(dw))

    for i in range(len(arch['hidden_layer_nodes']) -1):
        dz = np.multiply(np.dot(dz, dw), sigmoid_derr(all_layers_activations[-2-i])) 
        dw = np.dot(np.transpose(all_layers_activations[-3-i]), dz)
        all_sorted_weights[-2-i] = arch['learning_rate'] * all_sorted_weights[-2-i] - dw 
 
             
#####MAIN#######
#for i in range(arch['iterations']): ##big loop
 #   backwards_pass(i)
  #  all_layers_activations.clear()
    
#final_result = forward_pass(0)
#print("Final Result: ", final_result)

def update_vel(curr_vel, curr_pos, loc_best_pos, glob_best_pos):
    return (constant.INERTIA_WEIGHT * curr_vel) + (constant.COGNITIVE_WEIGHT * (loc_best_pos - curr_pos)) + (constant.SOCIAL_WEIGHT * (glob_best_pos - curr_pos))

def update_pos(curr_pos, updated_vel):
    return curr_pos + updated_vel


#def PSO():
  #  swarm = []




swarm = []
for i in range(neuralNet.NUM_PARTICLES):
    particle = NN(arch['input_layer_size'], arch['hidden_layer_nodes'], inputMatrix)
    swarm.append(particle)

for part in swarm:
    #print(part.weights)
    print("Current Vel: ", part.current_vel)
    #print("Output: ", part.outputs)

for part in swarm:
    part.forward_pass()

print("Outputs")
print(swarm[0].outputs)