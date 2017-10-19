#!/usr/bin/python
import numpy as np
import matplotlib.pyplot
import pylab
import sys
from scipy import optimize #Leastsq Levenberg-Marquadt Algorithm
import matplotlib.pyplot as plt

def log_scatter_plot(name):

    #name = 'part-r-000001'
    data = np.loadtxt(str(name), delimiter='\t')
    data = sorted(data, key=getKey)

    x = np.zeros((1,len(data)))
    y = np.zeros((1,len(data)))
    x_log = []
    y_log = []
    for i,_ in enumerate(data):
        x[0][i] = data[i][0]
        y[0][i] = data[i][1]
        if i == 0:
            x_log.append(float(np.log10(data[i][0])))
            y_log.append(float(np.log10(data[i][1])))
        else:
            if np.log10(data[i][1]) not in y_log:
                x_log.append(float(np.log10(data[i][0])))
                y_log.append(float(np.log10(data[i][1])))

    x_log_arr = np.array(x_log)
    y_log_arr = np.array(y_log)

    p0 = [0.1,-1]
    p1, success = optimize.leastsq(errfunc, p0[:], args=(x_log_arr, y_log_arr))

    x_log2 = np.linspace(min(x_log), max(x_log), 80)

    print("estimated parameters: ", p1)
    print("observed parameters: ", p0)

    plt.plot(x_log, y_log,"o", x_log2, fitfunc(p1, x_log2),'-')
    plt.legend(['data', 'est. par.: %.2f %.2f' % (p1[0],p1[1])], loc='best')
    plt.title('plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def fitfunc (p,x):
    #return p[0] * (x ** p[1])
    return p[0] + p[1] * x

def errfunc (p,x,y):
    err =  y - fitfunc(p,x)
    return err

def getKey(item):
    return item[0]

if __name__ == "__main__":
    log_scatter_plot(sys.argv[1])
