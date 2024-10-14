import numpy as np


def similarity_function(train_data, gamma, pt1_idx, pt2_idx):
    point1 = np.asarray(train_data.loc[[pt1_idx]])
    point2 = np.asarray(train_data.loc[[pt2_idx]])

    temp_res = 0

    for feature in range(len(point2[0])):
        #print(gamma[feature])
        temp_res += (point1[0][feature] - point2[0][feature]) ** 2 / gamma[feature]

    return np.exp(-temp_res)

def objective_function(train_data, train_data_graph, gamma):

    approx_error = 0
    for idx in range(train_data_graph.shape[0]):
        points = slice(train_data_graph.indptr[idx], train_data_graph.indptr[idx+1])
        dii = 0
        x_hat = 0
        temp_sum = 0
        for vertex in train_data_graph.indices[points]:
            wij = similarity_function(train_data, gamma, idx, vertex) 
            dii += wij
            temp_sum = wij * np.asarray(train_data.loc[[vertex]])
        #dii = (dii)**(-1)
        if dii > 0:
            temp_sum /= dii
        else:
            temp_sum = np.zeros(len(gamma))
        approx_error += np.sqrt(np.abs(np.asarray(train_data.loc[[idx]])**2 - temp_sum**2)) 
 
    return approx_error

def gradient_function(train_data, train_data_graph, gamma):
    #print("#############Top")
    gradient = np.zeros(train_data.loc[[0]].shape[1])
    for idx in range(train_data_graph.shape[0]):
        points = slice(train_data_graph.indptr[idx], train_data_graph.indptr[idx+1])
        dii = 0
        x_hat = 0
        sec_term_1 = 0
        sec_term_2 = np.zeros(train_data.loc[[0]].shape[1])
        #print(sec_term_2)

        point1 = np.asarray(train_data.loc[[idx]])
        for vertex in train_data_graph.indices[points]:
            point2 = np.asarray(train_data.loc[[vertex]])
            wij = similarity_function(train_data, gamma, idx, vertex) #->scalar value
            #print(wij
            dii += wij # ->scalar value
            #print(dii)
            x_hat += wij*train_data.loc[[vertex]].to_numpy()[0] #-> vector value
            dW_vals = dW_dsigma(point1, point2, idx, vertex, train_data, gamma)
            sec_term_1 += dW_vals* train_data.loc[[vertex]].to_numpy()[0] #->1x40 time 1x40 pt mult
            #print(sec_term_1)
            sec_term_2 = sec_term_2 + dW_vals
            #print(sec_term_2)
        #sec_term_2 = dD_dsigma() #-> scalar value
            #print(sec_term_2)
        #dii = (dii)**(-1) #->scalar value
        #print(dii)
        if dii > 0:
            x_hat = x_hat / dii # -> vector value
            leading_term_2 = np.divide((train_data.loc[[idx]].to_numpy()[0] - x_hat), dii, dtype=np.float128)
        else:
            x_hat = 0
            leading_term_2 = np.zeros(len(point2[0]))
        #print(x_hat)
        sec_term_2 = sec_term_2 * x_hat #-> vector value
        #print(sec_term_1)
        #print(sec_term_2)
        #print(leading_term_2)
        #print(leading_term_2.T)
        #print(gradient)
        gradient = gradient + (leading_term_2.T * (sec_term_1 - sec_term_2)) # -> vector value
        #print(gradient)
        #print("########Bottom")
    #print(gradient)
    return gradient 

# Define the partial derivatives of the function with respect to x and y
def dW_dsigma(point1, point2, pt1_idx, pt2_idx, train_data, gamma_matrix):
    
    derivative = []

    for idx in range(len(point1)):
        #inter_res = 2*similarity_matrix[pt1_idx][pt2_idx]*((point1[idx]-point2[idx])**2)*gamma_matrix[idx]**(-3)
        wij = similarity_function(train_data, gamma_matrix, pt1_idx, pt2_idx)
        inter_res = 2*wij*((point1[idx]-point2[idx])**2)*gamma_matrix[idx]**(-3)
        derivative.append(inter_res)

    return np.asarray(derivative)

def dD_dsigma(dW_vals):
    return np.sum(dW_vals)

# Define the gradient descent algorithm
def gradient_descent(learning_rate, num_iterations, tol, train_data, train_data_graph, gamma):
    print("Beggining Gradient Descent")

    #history_matrix  = np.ones((train_data_graph.shape[0], train_data_graph.shape[0]), dtype=object)

    #log = []

    # Perform the gradient descent iterations
    for i in range(num_iterations):
        print("Current Iteration: ", str(i+1))
        gradient = gradient_function(train_data, train_data_graph, gamma)
        curr_error = objective_function(train_data, train_data_graph, gamma)
        if np.all(curr_error < tol):
            break
        # Save the history of the parameters
        #log.append((gamma, ))
        print("Gamma: ", gamma)
        print("Gradient: ", gradient)
        print("Current Error: ", curr_error)
        gamma = gamma - (gradient * learning_rate)
        gamma = gamma[0]
        print("Updated Gamma: ", gamma)
    
    #gamma = objective_function(history_matrix, train_data, train_data_graph, gamma)
    print("Completed Gradient Descent")
    return gamma

'''
# Perform gradient descent and plot the results
start_x, start_y = 8, 8
learning_rate = 0.1
num_iterations = 20
x_opt, y_opt, f_opt, history = gradient_descent(start_x, start_y, learning_rate, num_iterations)
'''
