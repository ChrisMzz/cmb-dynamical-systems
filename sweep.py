import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
from scipy import interpolate
from model_utility import *
import json
import sys

param_name = "tauCA1"
if len(sys.argv) > 1: param_name = sys.argv[1]

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

def f(Y,t):
    """The function describing the ODE represented in the model.
    Here Y is a vector containing information [Q0,Q1,Q2,Q3,A1,A2].
    This function returns dynamics of Y, resulting in Y' = f(Y,t).
    """
    global pi0, pi1, pi2, alpha1, alpha2, alpha3, alpha2b, alpha3b, beta1, beta2, gamma2, gamma3, delta0, delta2, tauC, tauA1, tauA2, tauCA1, tauCA2
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

# transfer dynamics (with inhibition/activation)
f0 = lambda Q2, Q3 : pi0*(1+delta0*(Q2+Q3)/(1+Q2+Q3))
f1 = lambda A1 : pi1*(1-beta1*A1*rho(A1))
f2 = lambda A1, A2 : pi2*(beta2*A1*rho(A1)+delta2*A2)

t = np.linspace(0,60,500)

# the graphs of ranges marked by an x have a problem that will need fixing
sweep_ranges = {
    "pi0":[1e-5, 1],
    "pi1":[1e-5, 1],
    "pi2":[1e-5, 1],
    "gamma2":[1e-6, 0.99], # x
    "gamma3":[1e-6, 1],
    "beta1":[1e-6, 3.33],
    "beta2":[1e-6, 3.33],
    "delta0":[0.01, 10],
    "delta2":[0.01, 10],
    "tauC":[50, 1e3],
    "tauA2":[0.01, 2],
    "tauCA1":[0.3, 2],
    "tauCA2":[0.01,2],
    "alpha1":[1e-5,0.99], # x
    "alpha2":[1e-5,1],
    "alpha3":[1e-5,0.99], # x
    "alpha2b":[1e-5,1],
    "alpha3b":[1e-5,1] 
}
latex = {
    "pi0":"$\\pi_0$",
    "pi1":"$\\pi_1$",
    "pi2":"$\\pi_2$",
    "gamma2":"$\\gamma_2$",
    "gamma3":"$\\gamma_3$",
    "beta1":"$\\beta_1$",
    "beta2":"$\\beta_2$",
    "delta0":"$\\delta_0$",
    "delta2":"$\\delta_2$",
    "tauC":"$\\tau^C$",
    "tauA2":"$\\tau_{A2}$",
    "tauCA1":"$\\tau^C_{A1}$",
    "tauCA2":"$\\tau^C_{A2}$",
    "alpha1":"$\\alpha_1$",
    "alpha2":"$\\alpha_2$",
    "alpha3":"$\\alpha_3$",
    "alpha2b":"$\\overline{\\alpha_2}$",
    "alpha3b":"$\\overline{\\alpha_3}$"
}

with open('mice.json') as fp:
    mice = json.load(fp)
    for mouse in mice["sweep_mice"]:
        log_mouse(mouse)
        params = load_mouse(mouse)
        check_in_range(params)
        (pi0, pi1, pi2, alpha1, alpha2, alpha3, alpha2b, alpha3b, beta1, beta2, gamma2, gamma3, delta0, delta2, tauC, tauA1, tauA2, tauCA1, tauCA2) = params
        Y0 = [2,0,0,0]+[0,0.05]

        
        x = np.linspace(0, 1, 20)
        param_space = interpolate.interp1d([0, 1], sweep_ranges[param_name])
        
        fig, AX = plt.subplots(2,3)
        fig.set_figheight(8)
        fig.set_figwidth(10)
        
        for current in param_space(x):
            mouse[param_name] = current
            (pi0, pi1, pi2, alpha1, alpha2, alpha3, alpha2b, alpha3b, beta1, beta2, gamma2, gamma3, delta0, delta2, tauC, tauA1, tauA2, tauCA1, tauCA2) = load_mouse(mouse)
            Yest = spi.odeint(f,Y0,t)
            Q0,Q1,Q2,Q3 = Yest[:,0], Yest[:,1], Yest[:,2], Yest[:,3]
            A1, A2 = Yest[:,4], Yest[:,5]
            fig, AX = split_time_view_reuse(fig, AX)
        ((axQ0,axQ1,axA1),(axQ2,axQ3,axA2)) = AX
        # plot last one again in red
        axQ0.plot(t,Q0, color='red'), axQ1.plot(t,Q1, color='red')
        axQ2.plot(t,Q2, color='red'), axQ3.plot(t,Q3, color='red')
        axA1.plot(t,A1, color='red'), axA2.plot(t,A2, color='red')
        mouse[param_name] = sweep_ranges[param_name][0]
        (pi0, pi1, pi2, alpha1, alpha2, alpha3, alpha2b, alpha3b, beta1, beta2, gamma2, gamma3, delta0, delta2, tauC, tauA1, tauA2, tauCA1, tauCA2) = load_mouse(mouse)
        Yest = spi.odeint(f,Y0,t)
        Q0,Q1,Q2,Q3 = Yest[:,0], Yest[:,1], Yest[:,2], Yest[:,3]
        A1, A2 = Yest[:,4], Yest[:,5]
        # plot first one again in blue
        axQ0.plot(t,Q0, color='blue'), axQ1.plot(t,Q1, color='blue')
        axQ2.plot(t,Q2, color='blue'), axQ3.plot(t,Q3, color='blue')
        axA1.plot(t,A1, color='blue'), axA2.plot(t,A2, color='blue')
        fig.supxlabel("Time in days")
        fig.suptitle(f"{latex[param_name]} sweep from {sweep_ranges[param_name][0]} (blue) to {sweep_ranges[param_name][1]} (red) ; linear interpolation")
        if len(sys.argv) > 2:
            if sys.argv[2] == "linear":
                if len(sys.argv) > 3: fig.savefig(f"sim_dump/sweep/{mouse['name']}_{sys.argv[1]}_linear_sweep.png", dpi=300, format='png')
                #plt.show()
        else: plt.show()
        plt.close(fig)
        
        fig, AX = plt.subplots(2,3)
        fig.set_figheight(8)
        fig.set_figwidth(10)
        for current in param_space(x**2):
            mouse[param_name] = current
            (pi0, pi1, pi2, alpha1, alpha2, alpha3, alpha2b, alpha3b, beta1, beta2, gamma2, gamma3, delta0, delta2, tauC, tauA1, tauA2, tauCA1, tauCA2) = load_mouse(mouse)
            Yest = spi.odeint(f,Y0,t)
            Q0,Q1,Q2,Q3 = Yest[:,0], Yest[:,1], Yest[:,2], Yest[:,3]
            A1, A2 = Yest[:,4], Yest[:,5]
            fig, AX = split_time_view_reuse(fig, AX)
        ((axQ0,axQ1,axA1),(axQ2,axQ3,axA2)) = AX
        # plot last one again in red
        axQ0.plot(t,Q0, color='red'), axQ1.plot(t,Q1, color='red')
        axQ2.plot(t,Q2, color='red'), axQ3.plot(t,Q3, color='red')
        axA1.plot(t,A1, color='red'), axA2.plot(t,A2, color='red')
        mouse[param_name] = sweep_ranges[param_name][0]
        (pi0, pi1, pi2, alpha1, alpha2, alpha3, alpha2b, alpha3b, beta1, beta2, gamma2, gamma3, delta0, delta2, tauC, tauA1, tauA2, tauCA1, tauCA2) = load_mouse(mouse)
        Yest = spi.odeint(f,Y0,t)
        Q0,Q1,Q2,Q3 = Yest[:,0], Yest[:,1], Yest[:,2], Yest[:,3]
        A1, A2 = Yest[:,4], Yest[:,5]
        # plot first one again in blue
        axQ0.plot(t,Q0, color='blue'), axQ1.plot(t,Q1, color='blue')
        axQ2.plot(t,Q2, color='blue'), axQ3.plot(t,Q3, color='blue')
        axA1.plot(t,A1, color='blue'), axA2.plot(t,A2, color='blue')
        fig.suptitle(f"{latex[param_name]} sweep from {sweep_ranges[param_name][0]} (blue) to {sweep_ranges[param_name][1]} (red) ; quadratic interpolation")
        fig.supxlabel("Time in days")
        if len(sys.argv) > 2:
            if sys.argv[2] == "quadratic":
                if len(sys.argv) > 3: fig.savefig(f"sim_dump/sweep/{mouse['name']}_{sys.argv[1]}_quadratic_sweep.png", dpi=300, format='png')
                #plt.show()
        plt.close(fig)
        


            
        
        
        
        

