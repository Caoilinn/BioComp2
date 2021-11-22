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

# Output matrix or "y"
output_matrix = np.matrix(dataset_outputs, dtype=None, copy=True)
inputMatrix = np.matrix(dataset_inputs, dtype=None, copy=True)


# Neural Network & PSO architecture
arch = {
    'input_layer_size': inputMatrix.shape[1],
    'hidden_layer_nodes': (4, 2),
    'output_layer_nodes': 1,
    'iterations': 1000,
    'particles': 10,
    'num_informants': 6
}


swarm = []
best_global_position = []
best_global_error = sys.float_info.max


def setup_swarm(best_global_position, best_global_error):
    # Setup the swarm
    for i in range(arch['particles']):
        particle = NN(arch['input_layer_size'],
                      arch['hidden_layer_nodes'], inputMatrix)
        particle.forward_pass()
        particle.cross_entropy(output_matrix)
        swarm.append(particle)

        if(particle.error < best_global_error):
            best_global_error = particle.error
            best_global_position = particle.weights


def setup_informants():
    for particles in swarm:
        # Loop for as many informants as you want each particle to have
        for i in range(arch['num_informants']):
            # use a random num to chose the index of the particle to add as an informant
            index = np.random.randint(0, arch['particles'])
            particles.add_informant(swarm[index])


def update_velocity(particle, global_best):

    # Calculate random social and cognitive weights within the determined upper limit
    rand_cognitive = np.random.uniform(0, neuralNet.COGNITIVE_WEIGHT)
    rand_social = np.random.uniform(0, neuralNet.SOCIAL_WEIGHT)
    rand_global = np.random.uniform(0, neuralNet.GLOBAL_WEIGHT)

    # Create an updated valocity array(array of matricies) to return
    updated_vel = []

    # Loop through the particle's weights - using velocity here but the length is the same
    for i in range(len(particle.current_vel)):

        # Here we use the informant list
        lowest_social_error = sys.float_info.max
        best_social_weights = []

        # Loop through the particle's informants to find the one with the best error
        for val in particle.informants:
            if(val.best_error < lowest_social_error):
                best_social_weights = val.best_position

        # Use the equations from the PSO algorithm to calculate the social, cognitive and global components and add them together - separated for readability
        inertia_val = np.multiply(
            particle.current_vel[i], neuralNet.INERTIA_WEIGHT)
        cog_val = rand_cognitive * \
            np.subtract(particle.best_position[i], particle.weights[i])
        soc_val = rand_social * \
            np.subtract(best_social_weights[i], particle.weights[i])
        global_val = rand_global * \
            np.subtract(best_global_position[i], particle.weights[i])

        updated_vel.append(inertia_val + cog_val + soc_val + global_val)

    return updated_vel


def update_position(particle):
    # Use the updated velocity to update the weights (position)
    return np.add(particle.weights, particle.current_vel)


def cross_entropy(y, y_hat):
    return -(np.multiply(y, np.log(y_hat)) + np.multiply((1-y), np.log(1-y_hat)))


# Setup for PSO
setup_swarm(best_global_position, best_global_error)
setup_informants()

i = 0
# PSO
while(i < arch['iterations']):
    for part in swarm:
        # Check if the particle has the new best error
        if(part.error < best_global_error):
            best_global_error = part.error
            best_global_position = part.weights

        new_vel = update_velocity(part, best_global_position)
        part.current_vel = new_vel

        new_pos = update_position(part)
        part.weights = new_pos

        part.forward_pass()
        part.cross_entropy(output_matrix)

    i += 1


final = NN(arch['input_layer_size'], arch['hidden_layer_nodes'], inputMatrix)
final.weights = best_global_position
final.forward_pass()
final_result = final.output

f = pd.DataFrame(np.around(final_result, decimals=6)).join(dataset_outputs)
f['pred'] = f[0].apply(lambda x: 0 if x < 0.5 else 1)
print("Accuracy (Loss minimization):")
print(f.loc[f['pred'] == f['Outputs']].shape[0] / f.shape[0] * 100)

# For Confusion Matrix
YActual = f['Outputs'].tolist()
YPredicted = f['pred'].tolist()

# print(YActual)
# print(YPredicted)

TP = 0
TN = 0
FP = 0
FN = 0

for l1, l2 in zip(YActual, YPredicted):
    if (l1 == 1 and l2 == 1):
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

# F-Measure = (2 * Precision * Recall) / (Precision + Recall), sometimes called F1

F1 = (2 * P * R)/(P + R)

print("F score = ")
print(F1)
print("--------------------FINAL OUTPUTS--------------------")
print(final_result)

print("--------------------ERRORS--------------------")
error = cross_entropy(output_matrix, final_result)
print(error)
