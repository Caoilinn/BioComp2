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
    'hidden_layer_nodes' : (4,2),
    'output_layer_nodes' : 1,
    'iterations': 1000,
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


def update_velocity(particle, global_best):

    #print(particle.current_vel)
    inertia_vel = np.multiply(particle.current_vel, neuralNet.INERTIA_WEIGHT ) 
    cognitive_factor = neuralNet.COGNITIVE_WEIGHT * np.subtract(particle.best_position ,particle.weights)
    social_factor = neuralNet.SOCIAL_WEIGHT * np.subtract(global_best, particle.weights)
   
    return inertia_vel + cognitive_factor + social_factor

def update_position(particle):
    return particle.weights + particle.current_vel


def cross_entropy(y, y_hat):
    return -(np.multiply(y, np.log(y_hat)) + np.multiply((1-y), np.log(1-y_hat)))

max_num_iterations = 1000
i = 0

#PSO
while( i < arch['iterations'] ):
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

final = NN(arch['input_layer_size'], arch['hidden_layer_nodes'], inputMatrix)
final.weights = best_global_position
final.forward_pass()
final_result = final.output

f = pd.DataFrame(np.around(final_result, decimals=6)).join(dataset_outputs)
f['pred'] = f[0].apply(lambda x : 0 if x < 0.5 else 1)
print("Accuracy (Loss minimization):")
print(f.loc[f['pred']==f['Outputs']].shape[0] / f.shape[0] * 100)

#For Confusion Matrix
YActual = f['Outputs'].tolist()
YPredicted =  f['pred'].tolist()

#print(YActual)
#print(YPredicted)

TP = 0
TN = 0
FP = 0
FN = 0

for l1,l2 in zip(YActual, YPredicted):
    if (l1 == 1 and  l2 == 1):
        TP = TP + 1
    elif (l1 == 0 and l2 == 0):
        TN = TN + 1
    elif (l1 == 1 and l2 == 0):
        FN = FN + 1
    elif (l1 == 0 and l2 == 1):
        FP = FP + 1

print("Confusion Matrix: ")

print("TP=", TP)
print("TN=", TN)
print("FP=", FP)
print("FN=", FN)

# Precision = TruePositives / (TruePositives + FalsePositives)
# Recall = TruePositives / (TruePositives + FalseNegatives)


P = TP/(TP + FP)
R = TP/(TP + FN)

print("Precision = ", P)
print("Recall = ", R)

#F-Measure = (2 * Precision * Recall) / (Precision + Recall), sometimes called F1

F1 = (2* P * R)/(P + R)

print("F score = ")
print(F1)
print("--------------------FINAL OUTPUTS--------------------")
print(final_result)

print("--------------------ERRORS--------------------")
error = cross_entropy(output_matrix, final_result)
print(error)


