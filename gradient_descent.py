def gradient_descent(x, df, h=1e-4, e=1e-4, max_iters=1e6):
    iters = 0
    while (iters < max_iters and sum([i**2 for i in df(x)]) > e):
        x = [x[i] - h * df(x)[i] for i in range(len(x))]
        iters += 1
    return x, iters
