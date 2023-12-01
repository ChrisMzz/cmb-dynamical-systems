import warnings
import sys

class HypothesisError(Exception):
    """Raise when one of the hypotheses is not satisfied.
    """
    def __init__(self, message, H_number, *args: object) -> None:
        super().__init__(*args)
        self.H = f'(H{H_number})'
        self.message = self.H + f' {message}'
    def __str__(self): return self.message
    
class BehaviouralError(Exception):
    """Raise when a solution behaves in unexpected / unintended ways.
    """
    

class ParameterRangeWarning(UserWarning):
    """Warn when a parameter is not in the preferred range.
    """

def check_in_range(params):
    pi0, pi1, pi2, alpha1, alpha2, alpha3, alpha2b, alpha3b, beta1, beta2, \
        gamma2, gamma3, delta0, delta2, tauC, tauA1, tauA2, tauCA1, tauCA2 = params
    if not (tauA1 == 0.3): warnings.warn('tauA1 is not 0.3', ParameterRangeWarning, stacklevel=2)
    if not (1e-5 <= pi0 <= 1): warnings.warn('pi0 not in range', ParameterRangeWarning, stacklevel=2)
    if not (1e-5 <= pi1 <= 1): warnings.warn('pi1 not in range', ParameterRangeWarning, stacklevel=2)
    if not (1e-5 <= pi2 <= 1): warnings.warn('pi2 not in range', ParameterRangeWarning, stacklevel=2)
    if not (1e-6 <= gamma2 <= 1): warnings.warn('gamma2 not in range', ParameterRangeWarning, stacklevel=2)
    if not (1e-6 <= gamma3 <= 1): warnings.warn('gamma3 not in range', ParameterRangeWarning, stacklevel=2)
    if not (0 <= beta1 <= 1/tauA1): warnings.warn('beta1 not in range', ParameterRangeWarning, stacklevel=2)
    if not (0 <= beta2 <= 1/tauA1): warnings.warn('beta2 not in range', ParameterRangeWarning, stacklevel=2)
    if not (0 <= delta0 <= 10): warnings.warn('delta0 not in range', ParameterRangeWarning, stacklevel=2)
    if not (0 <= delta2 <= 10): warnings.warn('delta2 not in range', ParameterRangeWarning, stacklevel=2)
    if not (50 <= tauC <= 10e3): warnings.warn('tauC not in range', ParameterRangeWarning, stacklevel=2)
    if not (0 <= tauA2 <= 2): warnings.warn('tauA2 not in range', ParameterRangeWarning, stacklevel=2)
    if not (0.3 <= tauCA1 <= 2): warnings.warn('tauCA1 not in range', ParameterRangeWarning, stacklevel=2)
    if not (0 <= tauCA2 <= 2): warnings.warn('tauCA2 not in range', ParameterRangeWarning, stacklevel=2)
    if not (1e-5 <= alpha1 <= 1): warnings.warn('alpha1 not in range', ParameterRangeWarning, stacklevel=2)
    if not (1e-5 <= alpha2 <= 1): warnings.warn('alpha2 not in range', ParameterRangeWarning, stacklevel=2)
    if not (1e-5 <= alpha3 <= 1): warnings.warn('alpha3 not in range', ParameterRangeWarning, stacklevel=2)
    if not (1e-5 <= alpha2b <= 1): warnings.warn('alpha2b not in range', ParameterRangeWarning, stacklevel=2)
    if not (1e-5 <= alpha3b <= 1): warnings.warn('alpha3b not in range', ParameterRangeWarning, stacklevel=2)
    
default_hook=sys.excepthook
def exception_handler(exception_type, exception, traceback): 
    [print(f'{exception_type.__name__}: {exception}') if exception_type.__module__ == __name__ 
     else default_hook(exception_type, exception, traceback)]
sys.excepthook = exception_handler

