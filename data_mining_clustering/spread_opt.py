import numpy as np
import pandas as pd
from multiprocessing import Pool
from multiprocessing import cpu_count
from math import isclose
from math import ceil

from sklearn.decomposition import PCA

class aew():
    def __init__(self, data_graph, data, precomputed_gamma=np.empty((0,0))):

        self.similarity_matrix = np.zeros((data_graph.shape[0], data_graph.shape[0]))
        self.gamma = precomputed_gamma
        self.data_graph = data_graph
        self.data = data
        self.min_error = float('inf')
        self.eigenvectors = None

        if self.gamma.shape == (0,0):
            self.gamma = np.ones(41)

    def normalization_computation(self, section):
        res = []
        for x in section:
            for y in range(len(self.similarity_matrix[0])):
                x_sum = np.sum(self.similarity_matrix[x])
                y_sum = np.sum(self.similarity_matrix[y])
                if (x_sum > 0 and y_sum > 0) and (not isclose(x_sum, 0, abs_tol=1e-9) and not isclose(y_sum, 0, abs_tol=1e-9)):
                    res.append((x, y, self.similarity_matrix[x][y] / np.sqrt(x_sum * y_sum)))
                else:
                    res.append((x, y, 0))
        return res

    def normalization_parallel_caller(self, split_data):

        with Pool(processes=cpu_count()) as pool:
            edge_weight_res = [pool.apply_async(self.normalization_computation, (section, )) for section in split_data]

            edge_weights = [edge_weight.get() for edge_weight in edge_weight_res]

        for section in edge_weights:
            for weight in section:
                self.similarity_matrix[weight[0]][weight[1]] = weight[2]
                self.similarity_matrix[weight[1]][weight[0]] = weight[2]
        del edge_weights, edge_weight_res

    def laplacian_normalization(self):
        print("Normalizing Edge Weights")

        divisor = ceil(len(self.similarity_matrix[0]) / 32)

        curr_start = 0

        for idx in range(1, 33):

            if (divisor * idx) <= len(self.similarity_matrix[0]):
                curr_end = divisor * idx
            else:
                curr_end = len(self.similarity_matrix[0])

            split_data = self.split(range(curr_start, curr_end), cpu_count())
    
            self.normalization_parallel_caller(split_data)

            curr_start = curr_end

        print("Finished Normalizing Edge Weights")        

    def similarity_function(self, pt1_idx, pt2_idx):
        point1 = np.asarray(self.data.loc[[pt1_idx]])
        point2 = np.asarray(self.data.loc[[pt2_idx]])

        temp_res = 0

        for feature in range(len(point2[0])):
            temp_res += ((point1[0][feature] - point2[0][feature]) ** 2) / ((self.gamma[feature]) ** 2)

        return np.exp(-temp_res, dtype=np.longdouble)

    def objective_computation(self, section):
        approx_error = 0
        for idx in section:
            points = slice(self.data_graph.indptr[idx], self.data_graph.indptr[idx+1])
            dii = 0
            x_hat = 0
            temp_sum = 0
            for vertex in self.data_graph.indices[points]:
                wij = self.similarity_function(idx, vertex)
                dii += wij
                temp_sum = temp_sum + (wij * np.asarray(self.data.loc[[vertex]]))
            if dii > 0 and not isclose(dii, 0, abs_tol=1e-9):
                temp_sum /= dii
            else:
                temp_sum = np.zeros(len(self.gamma))
            approx_error += np.abs((np.asarray(self.data.loc[[idx]]) - temp_sum)**2)
        return np.sqrt(approx_error)

    def objective_function(self):
        errors = []

        split_data = self.split(range(self.data_graph.shape[0]), cpu_count())

        with Pool(processes=cpu_count()) as pool:
            errors = [pool.apply_async(self.objective_computation, (section, )) \
                                                                 for section in split_data]

            error = [error.get() for error in errors]
        return np.sum(error)

    def gradient_computation(self, section):
        gradient = np.zeros(len(self.gamma))
        for idx in section:
            points = slice(self.data_graph.indptr[idx], self.data_graph.indptr[idx+1])
            dii = 0
            x_hat = 0
            sec_term_1 = 0
            sec_term_2 = np.zeros(self.data.loc[[0]].shape[1])

            gradient_vector = np.zeros(len(self.gamma)) 

            point1 = np.asarray(self.data.loc[[idx]])
            for vertex in self.data_graph.indices[points]:
                point2 = np.asarray(self.data.loc[[vertex]])
                wij = self.similarity_function(idx, vertex)                
                dii += wij
                for feature_idx in range(len(self.gamma)):
                    diff = np.abs(point1[0][feature_idx] - point2[0][feature_idx])**2
                    gamma_term = self.gamma[feature_idx]**(-3)
                    gradient_vector[feature_idx] = diff * gamma_term * 2 * wij        
                x_hat = x_hat + (wij * point2)    
            if dii > 0 and not isclose(dii, 0, abs_tol=1e-9):
                x_hat = np.divide(x_hat, dii, casting='unsafe', dtype=np.longdouble)
                leading_term_2 = np.divide((self.data.loc[[idx]].to_numpy()[0] - x_hat), dii, casting='unsafe', dtype=np.longdouble)
            else:
                x_hat = 0
                leading_term_2 = np.zeros(len(point2[0]))
            sec_term_1 = np.multiply(gradient_vector, point2)
            sec_term_2 = np.multiply(gradient_vector, x_hat)
            gradient = gradient + (leading_term_2[0].transpose().tolist() * (sec_term_1[0] - sec_term_2[0]))
        return gradient

    def split(self, a, n):
        k, m = divmod(len(a), n)
        return [a[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

    def gradient_function(self):
        gradient = []
    
        split_data = self.split(range(self.data_graph.shape[0]), cpu_count())

        with Pool(processes=cpu_count()) as pool:
            gradients = [pool.apply_async(self.gradient_computation, (section, )) \
                                                                 for section in split_data]

            gradients = [gradient.get() for gradient in gradients]

        gradient = np.zeros(self.data.loc[[0]].shape[1])
        for grad in gradients:
            gradient = gradient + grad

        return gradient

    def dW_dsigma(self, point1, point2, pt1_idx, pt2_idx):
    
        derivative = []

        for idx in range(len(point1)):
            wij = self.similarity_function(pt1_idx, pt2_idx)
            inter_res = 2*wij*((point1[idx]-point2[idx])**2)*self.gamma[idx]**(-3)
            derivative.append(inter_res)

        return np.asarray(derivative)

    def dD_dsigma(self, dW_vals):
        return np.sum(dW_vals)

    def gradient_descent(self, learning_rate, num_iterations, tol):
        print("Beggining Gradient Descent")

        # Perform the gradient descent iterations
        for i in range(num_iterations):
            print("Current Iteration: ", str(i+1))
            print("Computing Gradient")
            gradient = self.gradient_function()
            print("Current Gradient: ", gradient)
            print("Computing Error")
            curr_error = self.objective_function()
            print("Current Error: ", curr_error)
                    
            if curr_error < tol:
                break
                
            elif self.min_error < curr_error and learning_rate > .001: 
                learning_rate -= .01 

            elif self.min_error - curr_error > 2:
                learning_rate += .01

            if curr_error < self.min_error:
                self.min_error = curr_error
            print("Gamma: ", self.gamma)
            self.gamma = self.gamma + (gradient * learning_rate)
            self.gamma = self.gamma
            print("Updated Gamma: ", self.gamma)
    
            print("Updated Learning Rate: ", learning_rate)
        print("Completed Gradient Descent")

    def generate_optimal_edge_weights(self, num_iterations):
        print("Generating Optimal Edge Weights")

        self.gradient_descent(.05, num_iterations, .01)

        self.generate_edge_weights()
    
    def edge_weight_computation(self, section):

        res = []

        for idx in section:
            point = slice(self.data_graph.indptr[idx], self.data_graph.indptr[idx+1])

            point1 = np.asarray(self.data.loc[[idx]])

            for vertex in self.data_graph.indices[point]:

                res.append((idx, vertex, self.similarity_function(idx, vertex)))

        return res

    def generate_edge_weights(self):
        print("Generating Edge Weights")

        split_data = self.split(range(self.data_graph.shape[0]), cpu_count())

        with Pool(processes=cpu_count()) as pool:
            edge_weight_res = [pool.apply_async(self.edge_weight_computation, (section, )) for section in split_data]

            edge_weights = [edge_weight.get() for edge_weight in edge_weight_res]

        for section in edge_weights:
            for weight in section:
                self.similarity_matrix[weight[0]][weight[1]] = weight[2]
                self.similarity_matrix[weight[1]][weight[0]] = weight[2]

        self.laplacian_normalization()

        self.subtract_identity()

        self.rewrite_edges()

        self.subtract_identity()

        self.eigenvectors = self.get_eigenvectors(3)

        self.data_graph = self.data_graph.toarray()

        print("Edge Weight Generation Complete")

    def rewrite_edges(self):

        rows, cols = self.data_graph.nonzero()

        for idx in range(len(rows)):
            row = rows[idx]
            col = cols[idx]
            self.data_graph[row, col] = self.similarity_matrix[row, col]

    def subtract_identity(self):
        return np.identity(len(self.similarity_matrix[0])) - self.similarity_matrix

    def get_eigenvectors(self, num_cols):
        eigenvalues, eigenvectors = np.linalg.eig(self.similarity_matrix)
        #print(eigenvalues/sum(eigenvalues))
        
        pca = PCA(n_components=len(self.similarity_matrix[0]))

        #print(pca.fit(self.similarity_matrix).explained_variance_ratio_)

        #print(pca.fit(self.similarity_matrix).singular_values_)

        pca = pca.fit_transform(self.similarity_matrix)

        #idx = eigenvalues.argsort()[-num_cols:][::-1]

        idx = eigenvalues.argsort()[::-1]

        #print(eigenvectors[:, idx])

        #print(eigenvectors[:, idx])

        #return eigenvectors[:, idx].real

        return pca.real

