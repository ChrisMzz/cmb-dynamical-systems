import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
from model_errors import *



IGNORE_MODEL_ERRORS = False


def split_time_view():
    fig, ((axQ0,axQ1,axA1),(axQ2,axQ3,axA2)) = plt.subplots(2,3)
    axQ0.plot(t,Q0), axQ1.plot(t,Q1), axQ2.plot(t,Q2), axQ3.plot(t,Q3)
    axA1.plot(t,A1),axA2.plot(t,A2)
    axQ0.set_title("Q0"), axQ1.set_title("Q1"), axQ2.set_title("Q2"), axQ3.set_title("Q3")
    axA1.set_title("A1"), axA2.set_title("A2")
    return fig, ((axQ0,axQ1,axA1),(axQ2,axQ3,axA2))

def time_view():
    fig, (axQ,axA) = plt.subplots(1,2)
    axQ.plot(t,Q0), axQ.plot(t,Q1), axQ.plot(t,Q2), axQ.plot(t,Q3)
    axA.plot(t,A1),axA.plot(t,A2)
    axQ.set_title("Q")
    axA.set_title("A")
    return fig, (axQ,axA)



# Model ---------------------------------------------------------------

def f(Y,t):
    """The function describing the ODE represented in the model.
    Here Y is a vector containing information [Q0,Q1,Q2,Q3,A1,A2].
    This function returns dynamics of Y, resulting in Y' = f(Y,t).
    """
    Q,A = Y[:4], Y[3:] # Y[3] is Q[3] but I want to write A[1] as A1 so I use A = [Q3, A1, A2]
    return np.array([
        -f0(Q[2],Q[3])*Q[0],
        f0(Q[2],Q[3])*Q[0] - f1(A[1])*Q[1],
        gamma2*Q[2]*(1-(Q[2]+Q[3])/tauC + A[1]/tauCA1 + A[2]/tauCA2) + f1(A[1])*Q[1] - f2(A[1], A[2])*Q[2],
        gamma3*Q[3]*(1-(Q[2]+Q[3])/tauC + A[1]/tauCA1 + A[2]/tauCA2) + f2(A[1], A[2])*Q[2],
        
        (alpha1*Q[1]+alpha2*Q[2]+alpha3*Q[3])*(1-A[1]/tauA1)*(1+A[1]/tauA1),
        (alpha2b*Q[2]+alpha3b*Q[3])*A[2]*(1-A[2]/tauA2)
    ])

eps = 0.1
rho = lambda x : (1+x/np.sqrt(x**2+eps))/2 # multiplicative regularisation term


pi0, pi1, pi2 = 0.5, 0.3, 0.4 # transfer terms
alpha1, alpha2, alpha3, alpha2b, alpha3b = 0.1, 0.2, 0.3, 0.2, 0.1 # axon growth dynamics
beta1, beta2 = 2.5, 3       # inhibition terms
gamma2, gamma3 = 0.1, 0.2   # proliferation terms
delta0, delta2 = 0.2, 0.2   # activation terms

# transfer dynamics (with inhibition/activation)
f0 = lambda Q2, Q3 : pi0*(1+delta0*(Q2+Q3)/(1+Q2+Q3))
f1 = lambda A1 : pi1*(1-beta1*A1*rho(A1))
f2 = lambda A1, A2 : pi2*(beta2*A1*rho(A1)+delta2*A2)


# saturation term and thresholds limiting tumor growth factors
tauC, [[tauA1,  tauA2], 
       [tauCA1, tauCA2]] = 0.1, [[0.5,0.3],
                                 [1,0.5]]


# Check hypotheses
if not IGNORE_MODEL_ERRORS:
    # H1
    if not (1 < beta1*tauA1) or not (1 < beta2*tauA1) : 
        raise HypothesisError("We don't have 1 < min(beta1*tauA1, beta2*tauA1)", 1)
    # H2
    if not (gamma2 < gamma3) : 
        raise HypothesisError("We don't have gamma2 < gamma3", 2)
    # H3
    if not (tauA1 < tauCA1):
        raise HypothesisError("We don't have tauA1 < tauCA1", 3)


# -------------------------------------------------


Y0 = [2,0,0,0]+[0,0.05]

t = np.linspace(0,60,50)

Yest = spi.odeint(f,Y0,t)
Q0,Q1,Q2,Q3 = Yest[:,0], Yest[:,1], Yest[:,2], Yest[:,3]
A1, A2 = Yest[:,4], Yest[:,5]


# Check for errors in the solution
if not IGNORE_MODEL_ERRORS:
    # Nonnegativity of Qi
    if np.any(Yest[:,:4]<0): 
        raise BehaviouralError("Nonnegativity of Qi is not respected")
    # Bounds of Ai
    if np.any(~ (np.abs(A1)<tauA1)    ): 
        raise BehaviouralError("|A1|<tauA1 is not respected")
    if np.any(~ ((0<=A2)*(A2<=tauA2)) ): 
        raise BehaviouralError("0<A2<tauA1 is not respected")
    


time_view()
plt.show()









