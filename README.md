# Neural-network-based-protocol
The mark in the code is similar to the paper Neural-network-based optimal quantum control of nonadiabatic geometric quantum computation via reverse engineering.
The specific description of the neural network is also in the paper.
You can change the number of hidden units, the connections between neurons, the initialization of the parameters and the learning rate based on your needs.
num in the codes means the number of the hidden units, T means the evolution period and epcho means the training times of the neural network.
The trainings of mu and eta are similar. You can change the form of y_train to obtain different trigonometric functions with different initial values.
You can use mu.jl to obtain the initialization of the neural network and then use T_gate.jl to perform the optimization.
