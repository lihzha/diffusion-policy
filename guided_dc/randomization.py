import random

def randomize_continuous(min, max, size: int):
    '''
    Return a list with length 'size', with each element's value within the interval [min, max].
    '''
    if size > 1:
        return [random.uniform(min, max) for _ in range(size)]
    else:
        return random.uniform(min, max)

def randomize_by_percentage(value, low_percentage=0.9, high_percentage=1.1):
    """Randomize a parameter value by a percentage."""
    return value * random.uniform(low_percentage, high_percentage)