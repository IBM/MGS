import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import sys, math

#------------------------------------------------------------#
# Name:         lyarosenstein(x,m,tao,meanperiod,maxiter)
# Description:  Python implementation of Matlab function
#               for computing largest Lyapunov exponent
#               based on Rosenstein' algorithm
#
# Reference:    https://www.mathworks.com/matlabcentral/fileexchange/38424-largest-lyapunov-exponent-with-rosenstein-s-algorithm?s_tid=prof_contriblnk
#
# Parameters:   x -       time series data
#               m -       mimumum embedding dimension
#               tao -
#               maxiter - number of iterations s
#
#------------------------------------------------------------#
def lyarosenstein(x,m,tao,meanperiod,maxiter):
    N = len(x)
    print('N: ',N)
    M = N - (m-1)*tao
    print('M: ', M)
    Y = psr_deneme(x,m,tao,3,M)
    neardis = np.ones((M,1))
    nearpos = np.ones((M,1))

    for i in range(M):
        x0 = np.ones((M,1))
        x0 = pd.DataFrame(x0)

        x0 = (Y.loc[i,:].to_numpy() * x0.to_numpy())

        distance = np.sqrt(np.sum(np.power((Y - x0),2),axis=1))
        #print(distance)
        #print(type(distance))
        for j in range(M):
            if (np.abs(j-i) <= meanperiod):
                distance[j] = 1e10;

        neardis[i] = np.min(distance.to_numpy())
        nearpos[i] = np.argmin(distance.to_numpy())
        #print(distance[240])
        #print(distance[241])
        #print(distance[244])
        #print(nearpos[i])
        #index = int(nearpos[i])
        #print(neardis[i])

    d = np.zeros((maxiter,1))

    for k in range(maxiter):
        max_ind = M - k - 1
        evolve = 0
        pnt = 0

        for j in range(M):
            #print('nearpos_j: ', nearpos[j])
            #print(Y.iloc[j+k+1,:])
            #print(Y.iloc[int(nearpos[j]+k+1),:])
            if (j < max_ind and nearpos[j] < max_ind):
                dist_k = np.sqrt(np.sum(np.power((Y.iloc[j+k+1,:].values - Y.iloc[int(nearpos[j]+k+1),:].values),2)))
                print('Y.iloc[j+k+1]: ', Y.iloc[j+k+1,:].values)
                print('Y.iloc[nearpos(j)+k]: ', Y.iloc[int(nearpos[j]+k+1),:].values)
                print('Dist_k: ',dist_k)

                if dist_k != 0:
                    evolve = evolve + np.log(dist_k)
                    pnt = pnt + 1
        if pnt > 0:
            d[k] = evolve / pnt
        else:
            d[k] = 0
    return d


#------------------------------------------------------------#
# Name:         psr_deneme(x,m,tao)
# Description:  Python implementation of Matlab function
#               for phase space reconstruction
#
# Reference:
#
# Parameters:   x -   time series data
#               m -   embedding dimension
#               tao - time delay
#
#------------------------------------------------------------#
def psr_deneme(x,m,tao,nargin,npoint):

    N = len(x)
    if nargin == 4:
        M = npoint
    else:
        M = N - (m - 1) * tao

    Y = np.zeros((M,m))
    Y = pd.DataFrame(Y)
    #print(len(Y))

    for i in range(m):
        #print(i*tao)
        Y.loc[:,i] = x[(i*tao):(M+i*tao)].values
        #print(Y.iloc[:,i])

    return Y

#------------------------------------------------------------#
# Name:         knn_deneme(x,tao,nmax,rtol,atol)
# Description:  Python implementation of Matlab function
#               for finding minimum embedding dimension
#               using false nearest neighbors method
#
# Reference:    https://www.mathworks.com/matlabcentral/fileexchange/37239-minimum-embedding-dimension
#
# Parameters:   x -    time series data
#               tao -
#               nmax -
#               rtol - number of iterations s
#               atol -
#
#------------------------------------------------------------#
def knn_deneme(x,tao,nmax,rtol,atol):
    N = len(x)
    Ra = np.std(x)
    print('Ra: ', Ra)
    print('N: ', N)
    FNN = np.zeros((nmax,1))
    for m in range(nmax):
        M = N - (m + 1) * tao
        Y = psr_deneme(x,m+1,tao,4,M)
        #FNN = np.zeros((m+1,1))
        #print('Y: ', Y)
        for n in range(M):
            y0 = np.ones((M,1))*Y.iloc[n,:].values
            #print(Y.iloc[n,:])
            #print("Y_size: ", len(Y))
            #print("y0_size: ", len(y0))
            distance = np.sqrt(np.sum(np.power((Y - y0),2),axis=1))
            neardis = np.sort(distance)
            nearpos = np.argsort(distance)
            #print('x: ',x[n+(m+1)*tao-1])
            #print('x: ',x[nearpos[1] + (m+1)*tao-1])
            D = np.abs(x[n+(m+1)*tao-1] - x[nearpos[1] + (m+1)*tao - 1])
            R = np.sqrt(np.power(D,2)+np.power(neardis[1],2))

            if (math.isnan(D/neardis[1]) != True and (D/neardis[1] > rtol or R/Ra > atol)):
                FNN[m,0] = FNN[m,0] + 1
    print('FNN: ', FNN)
    print('FNN_0: ', FNN[0,0])
    FNN = (FNN/(FNN[0,0]))*100
    return np.argmin(FNN)

#------------------------------------------------------------#
# Name:         meanfreq(x)
# Description:  computes mean frequency of power spectrum of
#               a time-domain signal
#
# Parameters:   x - time series data
#
#------------------------------------------------------------#
def meanfreq(x):
    fs = 1/0.0002 # time in between observations is 0.2ms = 0.0002s
    f, PSD = signal.periodogram(x,fs)

    #PSD = 10*np.log10(PSD)
    #print('PSD: ',PSD)
    #print('f: ',f)
    #print('PSD*f: ',PSD*f)
    plt.figure()
    plt.plot(f,PSD)

    mean_freq = np.sum(PSD*f)/np.sum(PSD)

    return mean_freq

if __name__ == '__main__':

    with open(sys.argv[1],'r') as file:
        data = pd.DataFrame(l.rstrip().split() for l in file)

    #meanfreqs = np.zeros((100,1))
    #with open(sys.argv[2],'r') as file:
    #    meanfreqs = pd.DataFrame(l.rstrip().split() for l in file)

    data = data.astype(dtype=np.float64)
    #meanfreqs = meanfreqs.to_numpy(); #meanfreqs.astype(dtype=np.float64)
    #meanfreqs = meanfreqs.astype(dtype=np.float64);

    #print(data)
    #print('Mean_freqs: ',meanfreqs)

    isChaotic = np.zeros((100,1))
    max_lyap = np.zeros((100,1))
    component = 0;  # component/cell to compute Lyapunov exponent on
    num_iters = 10

    print(len(data)/10000)

    for i in range(2): #int(len(data)/10000)): # simulation conducted w/ 10,000 iterations
        inputs = data.iloc[i*10000:(i+1)*10000,component]
        print('Input: ',inputs)
        embed_dim = knn_deneme(inputs,5,25,15,2)
        #print('embed_dim: ', embed_dim)
        lyp_exps = lyarosenstein(inputs,int(embed_dim),5,1/meanfreq(inputs),num_iters)
        max_lyap[i] = np.max(lyp_exps)
        if (max_lyap[i] > 0):
            isChaotic[i] = 1
        #print('Input: ', inputs)
        print('Mean Frequency: ',meanfreq(inputs))
    #freqs,psd = signal.welch(data.iloc[:,16])

    #plt.figure()
    #plt.semilogx(freqs,psd)

    np.savetxt('max_lyaps.txt',max_lyap,fmt='%4.4f')
    plt.figure()
    plt.plot(np.arange(0,30,5),max_lyap) #plots largest Lyapunov exponent across gi for specific ge

    #plt.figure()
    #plt.plot(np.arange(0.005,0.05,0.005),isChaotic)

    #plt.figure()
    #plt.plot(lyp_exps)

    plt.show()
