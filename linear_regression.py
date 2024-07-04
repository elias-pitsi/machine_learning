import numpy as np 

def predict_single_loop(x, w, b): 
    n = x.shape[0]
    p = 0 
    for i in range(n): 
        p_i = x[i]*w[i]
        p = p + p_i
    p = p + b 
    return p 

# Vectorized version  

def predict(x, w, b):
    return np.dot(x, w) + b 

def compute_cost(x,y,w,b): 
    m = x.shape[0]
    cost = 0.0 
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        cost = cost + (f_wb_i - y[i])**2 
    cost = cost / (2*m)
    return cost 



