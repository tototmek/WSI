from matplotlib import projections
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from pylab import meshgrid
from functions import f, g


def plot():
    # Rysowanie wykresu f(x)
    x = np.linspace(-2, 2, 100)
    y = [f([x[i]]) for i in range(len(x))]
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Wykres funkcji f(x)")
    plt.show()

    # Rysowanie wyrkesu g(x)
    x = np.linspace(-2.4, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = meshgrid(x, y)
    Z = g([X, Y])
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)
    plt.xlabel("x1")
    plt.ylabel("x2")
    ax.set_zlabel("g(x)")
    plt.title("Wykres funkcji g(x)")
    plt.show()
    plt.contourf(X, Y, Z, np.linspace(-50, 60, 15), cmap=plt.cm.coolwarm)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Wykres poziomicowy funkcji g(x)")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    plot()
