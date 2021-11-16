import numpy as np
import pandas as pd


dataset_inputs = pd.read_csv(
    './CW2/demo.txt', delimiter=',', header=None, usecols=[0, 1, 2, 3])
dataset_inputs.columns = ["Input 1", "Input 2", "Input 3", "Input 4"]

dataset_outputs = pd.read_csv(
    './CW2/demo.txt', delimiter=',', header=None, usecols=[4])
dataset_outputs.columns = ["Outputs"]

output_matrix = np.matrix(dataset_outputs, dtype=None, copy=True)
inputMatrix = np.matrix(dataset_inputs, dtype=None, copy=True)


arch = {
    'input_layer_size' : inputMatrix.shape[1],
    'hidden_layer_nodes' : (4,2,2),
    'output_layer_nodes' : 1,
    'iterations': 5000,
    'learning_rate': 0.001
}
  

def init_weights():
    #Wh = np.random.rand(4, 4)
    #Wo = np.random.randn(4, 1)

    Wh = np.random.rand(arch['input_layer_size'], arch['hidden_layer_nodes'][0])
    print(Wh)
    for i in range(arch['hidden_layer_nodes']):
         #Input to Hidden Layer
        if (i == 0):
            Wh = np.random.rand(arch['input_layer_size'], arch['hidden_layer_nodes'][i])
        else:
            Wh = np.random.rand(arch['hidden_layer_nodes'][i-1], arch['hidden_layer_nodes'][i])

init_weights()

#print(inputMatrix)