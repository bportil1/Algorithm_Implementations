import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
from math import isclose

def laplacian_normalization(similarity_matrix):
    print("Normalizing Edge Weights")
    for x in range(len(similarity_matrix)):
        for y in range(len(similarity_matrix[0])):
            x_sum = np.sum(similarity_matrix[x])
            y_sum = np.sum(similarity_matrix[y])
            if (x_sum > 0 and y_sum > 0) and (not isclose(x_sum, 0, abs_tol=1e-9) and not isclose(y_sum, 0, abs_tol=1e-9)):
                similarity_matrix[x][y] = -(similarity_matrix[x][y] / np.sqrt(x_sum * y_sum))
            else:
                similarity_matrix[x][y] = 0
    print("Finished Normalizing Edge Weights")
    return similarity_matrix

def similarity_function(train_data, gamma, pt1_idx, pt2_idx):
    point1 = np.asarray(train_data.loc[[pt1_idx]])
    point2 = np.asarray(train_data.loc[[pt2_idx]])

    temp_res = 0

    for feature in range(len(point2[0])):
        temp_res += (point1[0][feature] - point2[0][feature]) ** 2 / (gamma[feature]) ** 2


    ##### this exponent op is rarely returing an overflow, not sure the type of value thats causing it, seems stable up to 5 iterations
    #print(temp_res)
    return np.exp(-temp_res, dtype=np.longdouble)

def objective_computation(train_data, train_data_graph, gamma, section):
    approx_error = 0
    for idx in section:
        points = slice(train_data_graph.indptr[idx], train_data_graph.indptr[idx+1])
        dii = 0
        x_hat = 0
        temp_sum = 0
        for vertex in train_data_graph.indices[points]:
            wij = similarity_function(train_data, gamma, idx, vertex)
            dii += wij
            temp_sum = wij * np.asarray(train_data.loc[[vertex]])
        if dii > 0 and not isclose(dii, 0, abs_tol=1e-9):
            temp_sum /= dii
        else:
            temp_sum = np.zeros(len(gamma))
        approx_error += np.sqrt(np.abs(np.asarray(train_data.loc[[idx]])**2 - temp_sum**2))
    return approx_error

def objective_function(train_data, train_data_graph, gamma):
    errors = []

    split_data = split(range(train_data_graph.shape[0]), cpu_count())

    with Pool(processes=cpu_count()) as pool:
        errors = [pool.apply_async(objective_computation, (train_data, train_data_graph \
                                                             , gamma, section)) \
                                                             for section in split_data]

        error = [error.get() for error in errors]
    return np.sum(error)

def gradient_computation(train_data, train_data_graph, gamma, section):
    gradient = 0
    for idx in section:
        #print(idx)
        points = slice(train_data_graph.indptr[idx], train_data_graph.indptr[idx+1])
        dii = 0
        x_hat = 0
        sec_term_1 = 0
        sec_term_2 = np.zeros(train_data.loc[[0]].shape[1])

        point1 = np.asarray(train_data.loc[[idx]])
        for vertex in train_data_graph.indices[points]:
            point2 = np.asarray(train_data.loc[[vertex]])
            wij = similarity_function(train_data, gamma, idx, vertex)
            dii += wij
            x_hat += wij*train_data.loc[[vertex]].to_numpy()[0]
            dW_vals = dW_dsigma(point1, point2, idx, vertex, train_data, gamma)
            sec_term_1 += dW_vals* train_data.loc[[vertex]].to_numpy()[0]
            sec_term_2 = sec_term_2 + dW_vals
        if dii > 0 and not isclose(dii, 0, abs_tol=1e-9):
            x_hat = np.divide(x_hat, dii, casting='unsafe', dtype=np.longdouble)
            leading_term_2 = np.divide((train_data.loc[[idx]].to_numpy()[0] - x_hat), dii, casting='unsafe', dtype=np.longdouble)
        else:
            x_hat = 0
            leading_term_2 = np.zeros(len(point2[0]))
        sec_term_2 = sec_term_2 * x_hat
        gradient = gradient + (leading_term_2.T * (sec_term_1 - sec_term_2))
    return gradient

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

def gradient_function(train_data, train_data_graph, gamma):
    gradient = []
    
    split_data = split(range(train_data_graph.shape[0]), cpu_count())

    with Pool(processes=cpu_count()) as pool:
        gradients = [pool.apply_async(gradient_computation, (train_data, train_data_graph \
                                                             , gamma, section)) \
                                                             for section in split_data]

        gradients = [gradient.get() for gradient in gradients]

    gradient = np.zeros(train_data.loc[[0]].shape[1])
    for grad in gradients:
        gradient = gradient + grad
    return gradient

def dW_dsigma(point1, point2, pt1_idx, pt2_idx, train_data, gamma_matrix):
    
    derivative = []

    for idx in range(len(point1)):
        wij = similarity_function(train_data, gamma_matrix, pt1_idx, pt2_idx)
        inter_res = 2*wij*((point1[idx]-point2[idx])**2)*gamma_matrix[idx]**(-3)
        derivative.append(inter_res)

    return np.asarray(derivative)

def dD_dsigma(dW_vals):
    return np.sum(dW_vals)

# Define the gradient descent algorithm
def gradient_descent(learning_rate, num_iterations, tol, train_data, train_data_graph, gamma):
    print("Beggining Gradient Descent")

    # Perform the gradient descent iterations
    for i in range(num_iterations):
        print("Current Iteration: ", str(i+1))
        print("Computing Gradient")
        gradient = gradient_function(train_data, train_data_graph, gamma)
        print("Current Gradient: ", gradient)
        print("Computing Error")
        curr_error = objective_function(train_data, train_data_graph, gamma)
        print("Current Error: ", curr_error)
        if curr_error < tol:
            break
        print("Gamma: ", gamma)
        gamma = gamma - (gradient * learning_rate)
        gamma = gamma[0]
        print("Updated Gamma: ", gamma)
    print("Completed Gradient Descent")
    return gamma
