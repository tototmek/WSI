def f(x):
    if len(x) != 1:
        raise ValueError("f(x) jest funkcją jednego parametru!")
    x = x[0]
    return 10*x**4 + 3*x**3 - 30*x**2 + 10*x


def df(x):
    # gradient funkcji f
    if len(x) != 1:
        raise ValueError("f(x) jest funkcją jednego parametru!")
    x = x[0]
    return [40*x**3 + 9*x**2 - 60*x + 10]


def g(x):
    if len(x) != 2:
        raise ValueError("g(x) jest funkcją dwóch parametrów!")
    return 10*x[1]**4 + 10*x[0]**4 + 3*x[0]**3 - 30*x[0]**2 + 10*x[0]


def dg(x):
    # gradient funkcji g
    if len(x) != 2:
        raise ValueError("g(x) jest funkcją dwóch parametrów!")
    return [
        40*x[0]**3 + 9*x[0]**2 - 60*x[0] + 10,
        40*x[1]**3]
