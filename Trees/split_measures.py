import numpy as np


def evaluate_measures(sample):
    """Calculate measure of split quality (each node separately).

    Please use natural logarithm (e.g. np.log) to evaluate value of entropy measure.

    Parameters
    ----------
    sample : a list of integers. The size of the sample equals to the number of objects in the current node. The integer
    values are equal to the class labels of the objects in the node.

    Returns
    -------
    measures - a dictionary which contains three values of the split quality.
    Example of output:

    {
        'gini': 0.1,
        'entropy': 1.0,
        'error': 0.6
    }

    """
    gini, entropy, error = 0, 0, 0

    def p1(s, k):
        tmp = np.array(s)
        return 1 / len(s) * len(tmp[tmp == k])

    for i in np.unique(sample):
        gini += p1(sample, i) * (1 - p1(sample, i))

    for i in np.unique(sample):
        entropy += p1(sample, i) * np.log(p1(sample, i))
    entropy *= -1

    for i in np.unique(sample):
        if p1(sample, i) > error:
            error = p1(sample, i)
    error = 1 - error
    measures = {'gini': float(gini), 'entropy': float(entropy), 'error': float(error)}
    return measures
