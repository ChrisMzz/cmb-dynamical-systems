import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
from model_utility import *
import json


IGNORE_MODEL_ERRORS = True


def split_time_view(scale=False):
    """Graph all relevant curves (Q0,Q1,Q2,Q3,A1,A2) wrt time separately.
    """
    fig, ((axQ0,axQ1,axA1),(axQ2,axQ3,axA2)) = plt.subplots(2,3)
    if not scale:
        axQ0.plot(t,Q0, color=(0,0,1)), axQ1.plot(t,Q1, color=(1,0.6,0.2))
        axQ2.plot(t,Q2, color=(0,0.6,0)), axQ3.plot(t,Q3, color=(1,0,0))
    else:
        sQ = Q0+Q1+Q2+Q3
        axQ0.plot(t,Q0/sQ, color=(0,0,1)), axQ1.plot(t,Q1/sQ, color=(1,0.6,0.2))
        axQ2.plot(t,Q2/sQ, color=(0,0.6,0)), axQ3.plot(t,Q3/sQ, color=(1,0,0))
    axA1.plot(t,A1, color=(1,0,1)), axA2.plot(t,A2, color=(0.2,1,0.5))
    axQ0.set_title("Q0"), axQ1.set_title("Q1"), axQ2.set_title("Q2"), axQ3.set_title("Q3")
    axA1.set_title("A1"), axA2.set_title("A2")
    fig.set_figheight(6)
    fig.set_figwidth(9)
    return fig, ((axQ0,axQ1,axA1),(axQ2,axQ3,axA2))

def split_time_view_reuse(fig, axes, scale=False):
    """Graph all relevant curves (Q0,Q1,Q2,Q3,A1,A2) wrt time separately.
    
    Takes figure and axes as argument so it can be reused with different data.
    """
    fig, ((axQ0,axQ1,axA1),(axQ2,axQ3,axA2)) = fig, axes
    if not scale:
        axQ0.plot(t,Q0, color='black'), axQ1.plot(t,Q1, color='black')
        axQ2.plot(t,Q2, color='black'), axQ3.plot(t,Q3, color='black')
    else:
        sQ = Q0+Q1+Q2+Q3
        axQ0.plot(t,Q0/sQ, color='black'), axQ1.plot(t,Q1/sQ, color='black')
        axQ2.plot(t,Q2/sQ, color='black'), axQ3.plot(t,Q3/sQ, color='black')
    axA1.plot(t,A1, color='black'), axA2.plot(t,A2, color='black')
    axQ0.set_title("Q0"), axQ1.set_title("Q1"), axQ2.set_title("Q2"), axQ3.set_title("Q3")
    axA1.set_title("A1"), axA2.set_title("A2")
    return fig, ((axQ0,axQ1,axA1),(axQ2,axQ3,axA2))



def time_view(scale=False):
    """Graph Q and A in time (Q superposed in one graph). Scale option displays proportion of Q (easier to compare).
    """
    fig, (axQ,axA) = plt.subplots(1,2)
    if not scale:
        axQ.plot(t,Q0, color=(0,0,1), label='Q0'), axQ.plot(t,Q1, color=(1,0.6,0.2), label='Q1')
        axQ.plot(t,Q2, color=(0,0.6,0), label='Q2'), axQ.plot(t,Q3, color=(1,0,0), label='Q3')
    else:
        sQ = Q0+Q1+Q2+Q3
        axQ.plot(t,Q0/sQ, color=(0,0,1), label='Q0'), axQ.plot(t,Q1/sQ, color=(1,0.6,0.2), label='Q1')
        axQ.plot(t,Q2/sQ, color=(0,0.6,0), label='Q2'), axQ.plot(t,Q3/sQ, color=(1,0,0), label='Q3')
    axA.plot(t,A1, color=(1,0,1), label='A1'),axA.plot(t,A2, color=(0.2,1,0.5), label='A2')
    axQ.set_title("Q")
    axA.set_title("A")
    axQ.legend(), axA.legend()
    fig.set_figheight(6)
    fig.set_figwidth(9)
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
        
        (alpha1*Q[1]+alpha2*Q[2]-alpha3*Q[3])*(1-A[1]/tauA1)*(1+A[1]/tauA1),
        (alpha2b*Q[2]+alpha3b*Q[3])*A[2]*(1-A[2]/tauA2)
    ])

eps = 0.001
rho = lambda x : (1+x/np.sqrt(x**2+eps))/2 # multiplicative regularisation term


# Parameters explanation (example) ---------------------------


pi0, pi1, pi2 = 1.5e-3, 1.9e-5, 1.1e-3 # transfer terms
alpha1, alpha2, alpha3, alpha2b, alpha3b = 1.6e-5, 8.4e-2, 4.7e-4, 6.0e-1, 1.6e-5 # axon growth dynamics
beta1, beta2 = 3.2, 2.2             # inhibition terms
gamma2, gamma3 = 4.2e-1, 9.3e-1     # proliferation terms
delta0, delta2 = 8.4, 3.2           # activation terms

# transfer dynamics (with inhibition/activation)
f0 = lambda Q2, Q3 : pi0*(1+delta0*(Q2+Q3)/(1+Q2+Q3))
f1 = lambda A1 : pi1*(1-beta1*A1*rho(A1))
f2 = lambda A1, A2 : pi2*(beta2*A1*rho(A1)+delta2*A2)


# saturation term and thresholds limiting tumor growth factors
tauC, [[tauA1,  tauA2], 
       [tauCA1, tauCA2]] = 8.5e1, [[0.3,1.9],
                                   [3.6e-1,1.6]]


# ------------------------------------------------------------




# try to recreate Figure 3 from section 4.3.1
fig3, ((axQ0,axQ1,axA1),(axQ2,axQ3,axA2)) = plt.subplots(2,3)
fig3.set_figheight(6)
fig3.set_figwidth(9)
with open('mice.json') as fp:
    mice = json.load(fp)
    for mouse in mice["mice"]: 
        try: # note that some mice parameters don't respect some hypotheses
            
            log_mouse(mouse)

            # Check ranges
            #params = (pi0, pi1, pi2, alpha1, alpha2, alpha3, alpha2b, alpha3b, beta1, beta2, gamma2, gamma3, delta0, delta2, tauC, tauA1, tauA2, tauCA1, tauCA2)
            params = load_mouse(mouse)
            check_in_range(params)
            (pi0, pi1, pi2, alpha1, alpha2, alpha3, alpha2b, alpha3b, beta1, beta2, gamma2, gamma3, delta0, delta2, tauC, tauA1, tauA2, tauCA1, tauCA2) = params

            # Check hypotheses
            if not IGNORE_MODEL_ERRORS:
                # H1
                if not (1 > beta1*tauA1) or not (1 > beta2*tauA1) : 
                    raise HypothesisError("We don't have 1 > max(beta1*tauA1, beta2*tauA1)", 1)
                # H2
                if not (gamma2 < gamma3) : 
                    raise HypothesisError("We don't have gamma2 < gamma3", 2)
                # H3
                if not (tauA1 < tauCA1):
                    raise HypothesisError("We don't have tauA1 < tauCA1", 3)


            # -------------------------------------------------


            Y0 = [2,0,0,0]+[0,0.05]

            t = np.linspace(0,60,500)

            Yest = spi.odeint(f,Y0,t) # use Euler explicit ??
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
            

            fig3, ((axQ0,axQ1,axA1),(axQ2,axQ3,axA2)) = split_time_view_reuse(fig3, ((axQ0,axQ1,axA1),(axQ2,axQ3,axA2)), scale=True)
            fig, axes = split_time_view()
            fig.savefig(f'sim_dump/{mouse["name"]}-split_time_view.png', dpi=300, format='png')
            plt.close(fig)
            fig, axes = time_view()
            fig.savefig(f'sim_dump/{mouse["name"]}-time_view.png', dpi=300, format='png')
            plt.close(fig)
            fig, axes = time_view(scale=True)
            fig.savefig(f'sim_dump/{mouse["name"]}-time_view_scaled.png', dpi=300, format='png')
            plt.close(fig)
            #plt.show()
        except Exception as e: print(e)
        print("\n")

fig3.savefig('sim_dump/allmice-split_time_view_scaled.png', dpi=300, format='png')






