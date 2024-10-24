import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
from math import isclose
from math import ceil

class aew(self, data_graph, data):
    self.similarity_matrix = similarity_matrix = np.zeros((train_data_graph.shape[0], train_data_graph.shape[0]))

    self.gamma = np.ones(train_data.loc[[0]].shape[1])
    self.data_graph = data_graph
    self.data = data


    def normalization_computation(self, section):
        res = []
        for x in section:
            for y in range(len(self.similarity_matrix[0])):
                x_sum = np.sum(self.similarity_matrix[x])
                y_sum = np.sum(self.similarity_matrix[y])
                if (x_sum > 0 and y_sum > 0) and (not isclose(x_sum, 0, abs_tol=1e-9) and not isclose(y_sum, 0, abs_tol=1e-9)):
                    res.append((x, y, -(self.similarity_matrix[x][y] / np.sqrt(x_sum * y_sum))))
                else:
                    res.append((x, y, 0))
        return res

    def normalization_parallel_caller(self, split_data):
        with Pool(processes=cpu_count()) as pool:
            edge_weight_res = [pool.apply_async(self.normalization_computation, (self.similarity_matrix, section)) for section in split_data]

            edge_weights = [edge_weight.get() for edge_weight in edge_weight_res]

        for section in edge_weights:
            for weight in section:
                self.similarity_matrix[weight[0]][weight[1]] = weight[2]

        del edge_weights, edge_weight_res

    def laplacian_normalization(self):
        print("Normalizing Edge Weights")

        divisor = ceil(len(self.similarity_matrix[0]) / 32)

        #fix repeating indices on last section split

        curr_start = 0

        for idx in range(1, 33):

            if (divisor * idx) <= len(self.similarity_matrix[0]):
                curr_end = divisor * idx
            else:
                curr_end = len(self.similarity_matrix[0])

            print(idx, " ", curr_start, " ", curr_end)

            split_data = split(range(curr_start, curr_end), cpu_count())
    
            print(split_data)

            normalization_parallel_caller(self, split_data)

            curr_start = curr_end

        print("Finished Normalizing Edge Weights")
        return similarity_matrix

    def similarity_function(pt1_idx, pt2_idx):
        point1 = np.asarray(self.data.loc[[pt1_idx]])
        point2 = np.asarray(self.data.loc[[pt2_idx]])

        temp_res = 0

        for feature in range(len(point2[0])):
            temp_res += (point1[0][feature] - point2[0][feature]) ** 2 / (self.gamma[feature]) ** 2


        ##### this exponent op is rarely returing an overflow, not sure the type of value thats causing it, seems stable up to 5 iterations
        #print(temp_res)

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
                temp_sum = wij * np.asarray(self.data.loc[[vertex]])
            if dii > 0 and not isclose(dii, 0, abs_tol=1e-9):
                temp_sum /= dii
            else:
                temp_sum = np.zeros(len(self.gamma))
            approx_error += np.sqrt(np.abs(np.asarray(self.data.loc[[idx]])**2 - temp_sum**2))
        return approx_error

    def objective_function(self):
        errors = []

        split_data = split(range(self.data_graph.shape[0]), cpu_count())

        with Pool(processes=cpu_count()) as pool:
            errors = [pool.apply_async(objective_computation, (section, )) \
                                                                 for section in split_data]

        error = [error.get() for error in errors]
        return np.sum(error)

    def gradient_computation(self, section):
        gradient = 0
        for idx in section:
            #print(idx)
            points = slice(self.data_graph.indptr[idx], self.data_graph.indptr[idx+1])
            dii = 0
            x_hat = 0
            sec_term_1 = 0
            sec_term_2 = np.zeros(self.data.loc[[0]].shape[1])

            point1 = np.asarray(self.data.loc[[idx]])
            for vertex in self.data_graph.indices[points]:
                point2 = np.asarray(self.data.loc[[vertex]])
                wij = self.similarity_function(idx, vertex)
                dii += wij
                x_hat += wij*self.data.loc[[vertex]].to_numpy()[0]
                dW_vals = dW_dsigma(point1, point2, idx, vertex)
                sec_term_1 += dW_vals* self.data.loc[[vertex]].to_numpy()[0]
                sec_term_2 = sec_term_2 + dW_vals
            if dii > 0 and not isclose(dii, 0, abs_tol=1e-9):
                x_hat = np.divide(x_hat, dii, casting='unsafe', dtype=np.longdouble)
                leading_term_2 = np.divide((self.data.loc[[idx]].to_numpy()[0] - x_hat), dii, casting='unsafe', dtype=np.longdouble)
            else:
                x_hat = 0
                leading_term_2 = np.zeros(len(point2[0]))
            sec_term_2 = sec_term_2 * x_hat
            gradient = gradient + (leading_term_2.T * (sec_term_1 - sec_term_2))
        return gradient

    def split(self, a, n):
        k, m = divmod(len(a), n)
        return [a[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

    def gradient_function(self):
        gradient = []
    
        split_data = split(range(self.data_graph.shape[0]), cpu_count())

        with Pool(processes=cpu_count()) as pool:
            gradients = [pool.apply_async(gradient_computation, (section)) \
                                                                 for section in split_data]

            gradients = [gradient.get() for gradient in gradients]

        gradient = np.zeros(train_data.loc[[0]].shape[1])
        for grad in gradients:
            gradient = gradient + grad
        return gradient

    def dW_dsigma(point1, point2, pt1_idx, pt2_idx):
    
        derivative = []

        for idx in range(len(point1)):
            wij = self.similarity_function(pt1_idx, pt2_idx)
            inter_res = 2*wij*((point1[idx]-point2[idx])**2)*self.gamma[idx]**(-3)
            derivative.append(inter_res)

        return np.asarray(derivative)

    def dD_dsigma(dW_vals):
        return np.sum(dW_vals)

    def gradient_descent(learning_rate, num_iterations, tol):
        print("Beggining Gradient Descent")

        # Perform the gradient descent iterations
        for i in range(num_iterations):
            print("Current Iteration: ", str(i+1))
            print("Computing Gradient")
            gradient = gradient_function()
            print("Current Gradient: ", gradient)
            print("Computing Error")
            curr_error = objective_function()
            print("Current Error: ", curr_error)
            if curr_error < tol:
                break
            print("Gamma: ", self.gamma)
            self.gamma = self.gamma - (gradient * learning_rate)
            self.gamma = self.gamma[0]
            print("Updated Gamma: ", self.gamma)
        print("Completed Gradient Descent")

    def generate_optimal_edge_weights(num_iterations):
        print("Generating Optimal Edge Weights")

        gradient_descent(.1, num_iterations, .01)

        generate_edge_weights()

    def edge_weight_computation(section):

        res = []

        for idx in section:
            point = slice(self.data_graph.indptr[idx], self.data_graph.indptr[idx+1])

            point1 = np.asarray(self.data.loc[[idx]])

            for vertex in self.data_graph.indices[point]:

                #point2 = np.asarray(self.data.loc[[vertex]])

                res.append((idx, vertex, self.similarity_function(idx, vertex)))

        return res

    def generate_edge_weights(self):
        print("Generating Edge Weights")

        split_data = split(range(self.data_graph.shape[0]), cpu_count())

        with Pool(processes=cpu_count()) as pool:
            edge_weight_res = [pool.apply_async(edge_weight_computation, (section)) for section in split_data]

            edge_weights = [edge_weight.get() for edge_weight in edge_weight_res]

        for section in edge_weights:
            for weight in section:
                self.similarity_matrix[weight[0]][weight[1]] = weight[2]

        laplacian_normalization()

        print("Edge Weight Generation Complete")

    def rewrite_edges(self):

        rows, cols = graph.nonzero()

        for idx in range(len(rows)):
            row = rows[idx]
            col = cols[idx]
            self.data_graph[row, col] = self.similarity_matrix[row, col]

