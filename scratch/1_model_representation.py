import numpy as np

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

print(f"x_train.shape {x_train.shape}")
m = x_train.shape[0]
print(f"number of training examples {m}")
m = len(x_train)

i = 0
x_i = x_train[i]
y_i = y_train[i]

w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")

def compute_model(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

tmp_f_wb = compute_model(x_train, w, b)

x_i = 1.2
cost_1200sqft = w * x_i + b

print(f"${cost_1200sqft:.0f} thousand dollars")
