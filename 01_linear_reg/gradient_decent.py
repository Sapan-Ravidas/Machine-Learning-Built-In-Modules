import numpy as np

def gradient_descent(x, y):
    m_curr = b_curr = 0
    rate = 0.08
    n = len(x)
    iterations = 10000
    
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val ** 2 for val in (y - y_predicted)])
        # print(m_curr, b_curr, y_predicted, i)
        md = -(2 / n) * sum(x * (y - y_predicted))
        yd = -(2 / n) * sum(y - y_predicted)
        m_curr = m_curr - rate * md
        b_curr = b_curr - rate * yd
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))

        
        
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])
gradient_descent(x, y)