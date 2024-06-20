import numpy as np

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w*x[i] + b
        cost += (f_wb - y[i])**2
    total_cost = (1/(2*m))*cost
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw = (1/m)*dj_dw
    dj_db = (1/m)*dj_db
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    dj_dw, dj_db = compute_gradient(x, y, w_in, b_in)
    w = w_in
    b = b_in
    for i in range(num_iters):
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

w_in = 0
b_in = 0
tmp_alpha = 1.0e-2
iterations = 1000
w_final, b_final = gradient_descent(x_train, y_train, w_in, b_in, tmp_alpha, iterations)
