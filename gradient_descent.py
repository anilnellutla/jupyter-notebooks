import numpy as np
import math

def main():
    x_train = np.array([1.0, 2.0])
    y_train = np.array([300.0, 500.0])

    w_init = 0
    b_init = 0
    # some gradient descent settings
    iterations = 10000
    tmp_alpha = 1.0e-2
    # run gradient descent
    w_final, b_final = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, iterations)
    
    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost + cost_sum

    total_cost = (1/(2*m)) * cost_sum
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]

        dj_dw = dj_dw + dj_dw_i
        dj_db = dj_db + dj_db_i

    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
    """

    # An array to store cost J and w's at each iteration primarily for graphing later

    b = b_in
    w = w_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # Update Parameters using equation: 
        # w = w - alpha * derivative of cost function dj_dw J(w)
        # b = b - alpha * derivative of cost function dj_db J(w)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}", f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ", f"w: {w: 0.3e}, b:{b: 0.5e}")
    
    return w, b

main()
