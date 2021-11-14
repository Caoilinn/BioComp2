import numpy as np
import pandas as pd
import neuralNet
import sys
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
     

swarm = []
best_global_position = []
best_global_error = sys.float_info.max

#Setup the swarm
for i in range(neuralNet.NUM_PARTICLES):
    particle = NN(arch['input_layer_size'], arch['hidden_layer_nodes'], inputMatrix)
    particle.forward_pass()
    particle.cross_entropy(output_matrix)
    swarm.append(particle)

    if(particle.error < best_global_error):
            best_global_error = particle.error
            best_global_position = particle.weights


max_num_iterations = 1000
i = 0


def update_velocity(particle, global_best):
    #print("Particle Current Vel: ", particle.current_vel)
    #print("particle Weights (Positions): ", particle.weights)
    #print("particle Best Position: ", particle.best_position)
    #print("Inertian Val: ", neuralNet.INERTIA_WEIGHT)
    #print("Intertia Mult: ", neuralNet.INERTIA_WEIGHT * particle.current_vel)
    #print("Best Position - Current: ", particle.best_position - particle.weights)

    #print(type(np.asmatrix(particle.weights)))
    
    #print("Global Best: ", global_best)
    #print("Position: ", particle.weights)
    
    print(neuralNet.INERTIA_WEIGHT * particle.current_vel)
    print()
    print()
    print()
    print()
    print()
    print()
    

    return (neuralNet.INERTIA_WEIGHT * particle.current_vel) + (neuralNet.COGNITIVE_WEIGHT * np.subtract(particle.best_position ,particle.weights) + (neuralNet.SOCIAL_WEIGHT * np.subtract(global_best, particle.weights)))
    

def update_position(particle):
    print()
    #return particle.weights + particle.current_vel


#PSO
while( i < max_num_iterations ):
    
    for part in swarm:
        new_vel = update_velocity(part, best_global_position)
        part.current_vel = new_vel

        new_pos = update_position(part)
        part.weights = new_pos

        part.forward_pass()
        part.cross_entropy(output_matrix)
        
        #Check if the particle has the new best error
        if(part.error < best_global_error):
            best_global_error = part.error
            best_global_position = part.weights

    i += 1


print("The best weights found are: ", best_global_position)
