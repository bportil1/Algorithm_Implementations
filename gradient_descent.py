import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function to be minimized (a simple quadratic function)
def objective_function(x, y, data_graph, curr_idx, points):
    
    '''
    point1 = np.asarray(data_graph.loc[[curr_idx]])
    for vertex in data_graph.indices[points]:
        point2 = np.asarray(data_graph.loc[[vertex]])
        
        temp_sum = 0

        for feature in range(len(point2[0])):
            temp_sum += (point1[0][feature] - point2[0][feature]) ** 2 / 
    ''' 
       
    point1 = np.asarray(data_graph.loc[[curr_idx]])
    for vertex in data_graph.indices[points]:
        point2 = np.asarray(data_graph.loc[[vertex]])
        
        temp_sum = 0

        for feature in range(len(point2[0])):
            temp_sum += (point1[0][feature] - point2[0][feature]) ** 2 / 

    #return x**2 + y**2

# Define the partial derivatives of the function with respect to x and y
def df_dx(x, y):
    return 2 * x

def df_dy(x, y):
    return 2 * y

# Define the gradient descent algorithm
def gradient_descent(start_x, start_y, learning_rate, num_iterations, similarity):
    # Initialize the parameters
    x = start_x
    y = start_y
    history = []
    
    # Perform the gradient descent iterations
    for i in range(num_iterations):
        # Calculate the gradients
        grad_x = df_dx(x, y)
        grad_y = df_dy(x, y)
        
        # Update the parameters
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y
        
        # Save the history of the parameters
        history.append((x, y, objective_function(x, y)))
    
    return x, y, objective_function(x, y), history

'''
# Perform gradient descent and plot the results
start_x, start_y = 8, 8
learning_rate = 0.1
num_iterations = 20
x_opt, y_opt, f_opt, history = gradient_descent(start_x, start_y, learning_rate, num_iterations)
'''
