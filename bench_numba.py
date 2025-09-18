import time

import numpy as np
from numba import njit, prange


def mandelbrot_py(w, h, max_iter):
    out = np.zeros((h, w), dtype=np.uint16)
    for y in range(h):
        cy = -1.5 + (y / (h - 1)) * 3.0
        for x in range(w):
            cx = -2.0 + (x / (w - 1)) * 3.0
            zx = 0.0
            zy = 0.0
            i = 0
            while zx * zx + zy * zy <= 4.0 and i < max_iter:
                xt = zx * zx - zy * zy + cx
                zy = 2.0 * zx * zy + cy
                zx = xt
                i += 1
            out[y, x] = i
    return out


@njit(cache=True)
def mandelbrot_nb(w, h, max_iter):
    out = np.zeros((h, w), dtype=np.uint16)
    for y in range(h):
        cy = -1.5 + (y / (h - 1)) * 3.0
        for x in range(w):
            cx = -2.0 + (x / (w - 1)) * 3.0
            zx = 0.0
            zy = 0.0
            i = 0
            while zx * zx + zy * zy <= 4.0 and i < max_iter:
                xt = zx * zx - zy * zy + cx
                zy = 2.0 * zx * zy + cy
                zx = xt
                i += 1
            out[y, x] = i
    return out


@njit(parallel=True, fastmath=True, cache=True)
def mandelbrot_nbp(w, h, max_iter):
    out = np.zeros((h, w), dtype=np.uint16)
    for y in prange(h):
        cy = -1.5 + (y / (h - 1)) * 3.0
        for x in range(w):
            cx = -2.0 + (x / (w - 1)) * 3.0
            zx = 0.0
            zy = 0.0
            i = 0
            while zx * zx + zy * zy <= 4.0 and i < max_iter:
                xt = zx * zx - zy * zy + cx
                zy = 2.0 * zx * zy + cy
                zx = xt
                i += 1
            out[y, x] = i
    return out


def bench(func, *args, warmup=True, repeat=1):
    if warmup:
        func(*args)
    t0 = time.perf_counter()
    for _ in range(repeat):
        func(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / repeat


if __name__ == "__main__":
    w, h, it = 800, 600, 300
    t_py = bench(mandelbrot_py, w, h, it, warmup=False)
    t_nb = bench(mandelbrot_nb, w, h, it, warmup=True)
    t_nbp = bench(mandelbrot_nbp, w, h, it, warmup=True)
    print(f"Python:        {t_py:.3f}s")
    print(f"Numba:         {t_nb:.3f}s  ({t_py/t_nb:0.1f}×)")
    print(f"Numba parallel:{t_nbp:.3f}s  ({t_py/t_nbp:0.1f}×)")
