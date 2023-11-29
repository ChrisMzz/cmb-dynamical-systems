


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


