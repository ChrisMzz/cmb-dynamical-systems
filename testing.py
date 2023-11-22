import numpy as np


# sinko streifer model

# u is a 2-variable function
# g is a 2-variable function
#d[u(x,t)]/dt + d[u(x,t)g(x,t)]/dx = -mu*u(x,t) is a PDE


# we'll be using the 3 mortality types model as it can be expressed with a simple ODE :
# x'(t) = -[pdir(t) + pdelay(t) + mu(t)] x(t).

# to do this, we'll express each of the rates with piecewise linear splines :
def l(j):
    if j == 0: return lambda t:t==0
    if j == n-1: return lambda t:t==T
    return       lambda t : (t-span[j-1])/(span[j]-span[j-1]) if (span[j-1] < t and t < span[j]) \
            else lambda t : (span[j+1]-t)/(span[j+1]-span[j]) if (span[j] < t and t < span[j+1]) \
            else lambda t:0
    

T = 20
n = 40
span = [dt for dt in np.linspace(0,20,n)]


q = lambda t,a : sum([a[j]*l(j)(t) for j in range(n)])






