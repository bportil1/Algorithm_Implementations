import numpy as np
from matplotlib import pyplot as plt

def generate_points(array_len: int) -> list[int]:

    '''
    #Generate random array with distinct values
    x = np.random.permutation(50)[:array_len]
    y1 = np.arange(1,30)
    y2 = np.arange(21,1, -1)
    y = np.concatenate((y1, y2), axis=0)
    print(y)
    points = []

    for idx in range(array_len):
        points.append((x[idx], y[idx]))
        
    points.sort()

    #points.sort(key=lambda x: x[1])
    '''

    # Set random seed for reproducibility
    np.random.seed(42)

    # Define number of points in each segment
    n_points = 50

    # Segment 1: y = 2x + 1 with some noise
    x1 = np.linspace(0, 5, n_points)
    y1 = 2 * x1 + 1 + np.random.normal(scale=1.0, size=n_points)

    # Segment 2: y = -1.5x + 15 with some noise
    x2 = np.linspace(5, 10, n_points)
    y2 = -1.5 * x2 + 15 + np.random.normal(scale=1.0, size=n_points)

    # Segment 3: y = 0.5x + 5 with some noise
    x3 = np.linspace(10, 15, n_points)
    y3 = 0.5 * x3 + 5 + np.random.normal(scale=1.0, size=n_points)

    # Combine the segments
    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])
    '''
    points = []

    for idx in range(array_len):
        points.append((x[idx], y[idx])) 
    '''
    #return points
    return x, y

def convert_tuples(points):
    x_vals = []
    y_vals = []
    for idx in range(len(points)):
        x_vals.append(points[idx][0])
        y_vals.append(points[idx][1])
    return np.asarray(x_vals), np.asarray(y_vals)

def calculate_best_fit_line(x_vals, y_vals, section_length):
    m = ((section_length*sum(x_vals*y_vals)) - sum(x_vals)*sum(y_vals)) / (section_length * sum(x_vals**2) - sum(x_vals)**2)     
    
    b = (sum(y_vals) - m*sum(x_vals)) / section_length
    
    return m, b

def calculate_errors(x_vals, y_vals):

    n = len(x_vals)

    errors = np.zeros((n,n))
    for j in range(n):
        for i in range(j+1):
            section_length = j - i + 1
            if j > 1:
                m, b = calculate_best_fit_line(x_vals[i:i+section_length], y_vals[i:i+section_length], section_length)
                errors[i, j] = sum((y_vals[i:i+section_length] - m*x_vals[i:i+section_length] - b)**2)
    
    return errors

def rebuild_opt_path(errors, length, c):
    M = np.zeros(length)
    p = np.zeros(length)

    for j in range(1, length):
        cost = [errors[i, j] + c + M[i-1] for i in range(j)]
        M[j] = np.min(cost)
        p[j] = np.argmin(cost)
    return M, p

c = 100

tol = .1

x_vals, y_vals = generate_points(25)
error = calculate_errors(x_vals, y_vals)
M, p = rebuild_opt_path(error, len(x_vals), c)

starts = np.unique(p)
starts = p
drawn = set([])
plt.plot(x_vals, y_vals)
for start in starts:
    indices = np.where(abs(p-start) < tol)[0]
    m, b = calculate_best_fit_line(x_vals[indices], y_vals[indices], len(indices))
    if not (m, b) in drawn:
        plt.plot([x_vals[min(indices)],x_vals[max(indices)]], [m*x_vals[min(indices)]+b, m*x_vals[max(indices)]+b], linewidth=3, 
                 label='line: ({:.2f}, {:.2f})'.format(m,b))
        drawn.add((m,b))
plt.legend()
plt.show()
