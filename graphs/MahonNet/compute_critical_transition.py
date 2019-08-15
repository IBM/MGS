import numpy as np
import pandas as pd
import sys
import io
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import lag_plot
import matplotlib.pyplot as plt
from scipy.stats import skew

#%matplotlib inline
k = 1
M = 5
w_f = 10
w_id = 10
gradient = 5
theta_0 = -10
theta_1 = 50

def x_f(x,n):
    xf = 0
    #print(len(x))
    for i in range(n+k+M,n+k+M+w_f):
        #print(i)
        if(i > len(x)-1):
            break

        xf += x[i]

    print("x_f: ", xf/w_f)
    return xf/w_f

def x_p(x,n,p):

    xp = 0

    start_index = n+k+(p-1)*w_id
    if(n+k+(p-1)*w_id < 0):
        start_index = 0

    for i in range(start_index,n+k+p*w_id):
        #print(i)
        if(i > len(x)-1):
            break
        xp += x[i]
    return xp/w_id

def x_p_range(x,n,p):

    for i in range(p-1):
        if(x_p(x,n,i) <= theta_1):
            return False

    return True

def tracer_variable(x,index):
    p = 4
    if(x[index+k+M] - x[index] >= gradient and x_f(x,index) >= theta_1 \
       and x_p_range(x,index,p) == True and x_p(x,index,p) >= theta_0):
        return 1
    elif(x[index+k+M] - x[index] <= -1*gradient and x_f(x,index) <= theta_0 \
        and x_p_range(x,index,p) > theta_1 and x_p(x,index,p) <= theta_1):
        return 1
    else:
        return 0


if __name__ == '__main__':
    print(sys.argv[1])
    with open(sys.argv[1],'r') as f:
        next(f)
        data = pd.DataFrame(l.rstrip().split() for l in f)
        #data = f.read()
    data = data.astype(dtype=np.float64)

    print(data)

    #data = pd.Series(data, index=data[:,0])

    #print(data.iloc[:,1])
    print(len(data.columns))
    for i in range(len(data.columns)):
        data.iloc[:,i] = pd.Series(data.iloc[:,i])
    print(data)

    data = data.iloc[:,1]
    print(data.autocorr(lag=1))

    #lag_plot(data)
    #plot_acf(data,lags=1)
    #pyplot.show()

    #data = data.iloc[:,2]
    plt.subplot(2,1,1)
    plt.plot(data)
    #plt.figure(figsize=(5,5))
    ## Variance
    plt.subplot(2,1,2)
    window_len = len(data)/2
    print(window_len)
    window_len = int(window_len)
    data.rolling(100).var().plot(figsize=(5,5),color = 'red')
    #data.rolling(5).skew().plot()
    #plt.show()

    tracer = np.ones((len(data),1))
    #print(len(data))
    #print("Data_Index_0: ",data[0])
    for i in range(len(data)):
        if((len(data) - 1) < i+k+M):
            break

        #print(i)
        #tracer[i] = tracer_variable(data,i)
        #print(tracer[i])

    #plt.plot(tracer)

    plt.show()


