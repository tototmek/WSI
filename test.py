from gradient_descent import gradient_descent
from functions import f, df, dg

print(gradient_descent([1, 1], dg, 1e-5, 1e-8, 1e8))
