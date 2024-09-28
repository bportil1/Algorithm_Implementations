import numpy as np
from matplotlib import pyplot as plt

def generate_piece_wise_linear_data(lines: int, points: int) -> list[int]:

    idx = 0
    x = []
    y = []
    
    
    for line in range(lines):
        x_temp = np.linspace(idx, (int(points/lines))+idx, int(points/lines))
        y_temp = np.random.randint(-50,50) + x_temp + np.random.randint(-50,50) + np.random.normal(scale=1.0, size=int(points/lines))
        idx += (points/lines)

        x.extend(x_temp)
        y.extend(y_temp)
        
    return np.asarray(x), np.asarray(y)

def calculate_best_fit_line(x_vals, y_vals, section_length):
    if (section_length * sum(x_vals**2) - sum(x_vals)**2) == 0:
        return 0, sum(y_vals) / section_length
    m = ((section_length*sum(x_vals*y_vals)) - sum(x_vals)*sum(y_vals)) / (section_length * sum(x_vals**2) - sum(x_vals)**2)     
    b = (sum(y_vals) - m*sum(x_vals)) / section_length
    return m, b

def calculate_errors(x_vals, y_vals):

    n = len(x_vals)

    errors = np.zeros((n,n))
    for j in range(n):
        for i in range(j+1):
            section_length = j - i + 1
            if section_length > 1:
                m, b = calculate_best_fit_line(x_vals[i:i+section_length], y_vals[i:i+section_length], section_length)
                errors[i, j] = sum((y_vals[i:i+section_length] - m*x_vals[i:i+section_length] - b)**2)
    
    return errors

def rebuild_opt_path(errors, length, c):
    M = np.zeros(length)
    p = np.zeros(length)
    M_max = np.zeros(length)
    p_max = np.zeros(length)
    for j in range(1, length):
        cost = [errors[i, j] + c + M[i-1] for i in range(j)]
        M[j] = np.min(cost)
        p[j] = np.argmin(cost)
    return M, p

def find_k_fitted_lines(M, p, k):
    diffs = np.diff(M)

    split_points = []

    if k <= 1:
        return [0]

    for idx in range(k):
        split_points.append(np.argmax(diffs))
        diffs[np.argmax(diffs)] = 0


    return sorted(split_points)


def plot_split_data(x_vals, y_vals, split_points):
    plt.plot(x_vals, y_vals)
    idx = 0
    for point in split_points:
        if len(x_vals[idx:point]) > 1:
            m, b = calculate_best_fit_line(x_vals[idx:point], y_vals[idx:point], len(x_vals[idx:point]))
            plt.plot(x_vals[idx:point], m*x_vals[idx:point]+b, linewidth=3, 
                 label='line: ({:.2f}, {:.2f})'.format(m,b))
        idx = point 

    if len(x_vals[idx:]) > 1:
        m, b = calculate_best_fit_line(x_vals[point:], y_vals[point:], len(x_vals[point:]))
        plt.plot(x_vals[point:], m*x_vals[point:]+b, linewidth=3, 
            label='line: ({:.2f}, {:.2f})'.format(m,b))
    
    plt.show()

c = 50
x_vals, y_vals = generate_piece_wise_linear_data(15, 50)
error = calculate_errors(x_vals, y_vals)
M, p = rebuild_opt_path(error, len(x_vals), c)
k = 10
split_points = find_k_fitted_lines(M, p, k)
plot_split_data(x_vals, y_vals, split_points)

