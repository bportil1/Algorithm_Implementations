import numpy as np
from matplotlib import pyplot as plt

def generate_piece_wise_linear_data(lines: int, points: int) -> list[int]:
    idx = 0
    x = []
    y = []
    
    #Generate piecewise linear components
    for line in range(lines):
        x_temp = np.linspace(idx, (int(points/lines))+idx, int(points/lines))
        y_temp = np.random.randint(-25,25) + x_temp + np.random.randint(-25,25) + np.random.normal(3, 2.5, size=int(points/lines))
        idx += (points/lines)
        x.extend(x_temp)
        y.extend(y_temp)
        
    return np.asarray(x), np.asarray(y)

def generate_non_linear_data(points: int) -> tuple[list[int], list[int]]:
    #Generate non-linear data
    de_linearize = lambda X: np.cos(1.5 * np.pi * X) + np.cos( 5 * np.pi * X )
    X = np.sort(np.random.rand(points)) * 2
    y = de_linearize(X) + np.random.randn(points) * 0.1

    return X, y

def error_coefs(x: list[int], y: list[int]) -> tuple[int, int, int]:
    n = len(x)
    
    if (n == 1):
        return (0.0, y[0], 0.0)

    #Compute terms for minimum error line calculation and error     
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    x_y_prod = np.sum(x * y)
    x_sq_sum = np.sum(x ** 2)
    if ((n*x_sq_sum - x_sum*x_sum) == 0):
        return (0.0, y[0], 0.0)
    a = (n*x_y_prod - x_sum*y_sum) / (n*x_sq_sum - x_sum*x_sum)
    b = (y_sum - a*x_sum) / n

    #Computer error for points around the minimum error line
    error = np.sum((y-a * x-b)**2)
    
    return a, b, error

def calculate_errors(x_vals: list[int], y_vals: list[int]) -> tuple[list[int], list[int], list[int]]:  
    n = len(x_vals)

    #Arrays for all errors and corresponding terms for backtracking
    errors = np.zeros((n,n))
    a_vals = np.zeros((n,n))
    b_vals = np.zeros((n,n))
    
    #Compute least squares error for segments between all pairs of points
    for j in range(n):
        for i in range(j+1):
            a_vals[i, j], b_vals[i, j], errors[i, j] = error_coefs(x_vals[i:j+1], y_vals[i:j+1])

    return a_vals, b_vals, errors

def rebuild_opt_path(a_vals: list[int], b_vals: list[int], x: list[int], y: list[int], errors: list[int], C: int) -> list[int]:
    n = len(errors[0])
    optimal_path = np.zeros(n)
    #Sum errors for all segments 
    for j in range(n):
        temp_path = np.zeros(j+1)
        temp_path[0] = errors[0,  j] + C

        #Find optimal solution using the recursive definition
        for i in range(1, j+1):
            temp_path[i] = optimal_path[i-1] + errors[i, j] + C 
        optimal_path[j] = np.min(temp_path)

    #Backtrack from end of dataset and rebuild optimal solution
    optimal_coeffs = []
    while j >= 0:
        temp_opt_coeffs = np.zeros(j+1)
        temp_opt_coeffs[0] = errors[0, j] + C
        for i in range(1, j+1):
            temp_opt_coeffs[i] = optimal_path[i-1] + errors[i, j] + C
        idx_opt = np.argmin(temp_opt_coeffs)
        a_optimal_section = a_vals[idx_opt,j]
        b_optimal_section = b_vals[idx_opt,j]
        if idx_opt <= 0:
            x_min = float('-inf')
        else:
            x_min = (x[idx_opt-1] + x[idx_opt])/2
        if j >= n-1:
            x_max = float('inf')
        else:
            x_max = (x[j] + x[j+1])/2   
        optimal_coeffs.insert(0, (x_min, x_max, a_optimal_section, b_optimal_section))
        j = idx_opt - 1
        
    return optimal_coeffs

def fit_data(x: list[int], optimal_coeffs: list[int]) -> tuple[list[int], list[int], list[int]]:
    n = len(x)
    y_approx = np.zeros(n)
    y_approx_segs = []
    indices = []
    #Approximate data using previously generated optimal path
    for opt_coeff in optimal_coeffs:
        ind = [i for i,elem in enumerate(x) if elem >= opt_coeff[0] and elem <= opt_coeff[1]]
        y_approx_val = x[ind] * opt_coeff[2] + opt_coeff[3]
        y_approx[ind] = y_approx_val
        indices.append(ind)
        y_approx_segs.append(y_approx_val)
    return y_approx, indices, y_approx_segs

def find_segments(x_vals: list[int], y_vals: list[int], max_segments: int, init_cost: float) -> list[int]:
    curr_C = init_cost
    curr_num_segs = float('inf')
    y_approx = []

    #Reapproximate data model by increasing error term until only the maximum
    #number of segments are used
    while curr_num_segs > max_segments:
        curr_C += .05
        a, b, errors = calculate_errors(x_vals, y_vals)
        optimal_coeffs = rebuild_opt_path(a, b, x_vals, y_vals, errors, curr_C)
        y_approx, indices, y_approx_segs = fit_data(x_vals, optimal_coeffs)
        curr_num_segs = len(optimal_coeffs)

    #print(optimal_coeffs)
    plot_approximation(x_vals, y_vals, y_approx, indices, y_approx_segs, curr_num_segs, curr_C)
    return y_approx

def plot_approximation(x_vals: list[int], y_vals: list[int], y_approx: list[int], indices: list[int], y_approx_segs: list[int], curr_num_segs: int, curr_C: float) -> None:
    fig = plt.figure(figsize=(8,4), constrained_layout=True)
    title_label = "SLS\n" + "(" + str(curr_num_segs) + " segments,  " + format(curr_C, ".2f") + " cost, " + str(len(x_vals)) + " points)\n"
    fig.suptitle(title_label, fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.82)
    #Plot actual data
    plot_1 = fig.add_subplot(131)
    plot_1.plot(x_vals, y_vals)
    plot_1.set_title("Original Data", loc="left")
    #Plot disconnected segments
    plot_2 = fig.add_subplot(132)
    for idx in range(len(y_approx_segs)):
        plot_2.plot(indices[idx], y_approx_segs[idx], '-', color='orange')
    plot_2.set_title('Unconnected Segments', loc="left")
    #Plot connected segments
    plot_3 = fig.add_subplot(133)
    plot_3.plot(x_vals, y_approx, 'o-', color="green")
    plot_3.set_title("Connected Segments")
    plt.show()

def test_driver() -> None:
    #Generate piecewise linear components for simple, traceable examples
    x_vals, y_vals = generate_piece_wise_linear_data(8, 50)
    number_of_segments = 15
    init_cost = .3
    y_approx = find_segments(x_vals, y_vals, number_of_segments, init_cost)

    #Generate non-linear data for more complicated examples
    x_vals, y_vals = generate_non_linear_data(50)
    number_of_segments = 15
    init_cost = .3
    y_approx = find_segments(x_vals, y_vals, number_of_segments, init_cost)

if __name__ == '__main__':
    test_driver()
