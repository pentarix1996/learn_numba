import time

import numpy as np
import pandas as pd
import streamlit as st
from numba import njit, prange

# ---------------------------
# Kernels: Mandelbrot
# ---------------------------


def mandelbrot_py(
    width: int, height: int, max_iter: int, xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5
) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.uint16)
    for y in range(height):
        cy = ymin + (y / (height - 1)) * (ymax - ymin)
        for x in range(width):
            cx = xmin + (x / (width - 1)) * (xmax - xmin)
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
def mandelbrot_numba(
    width: int, height: int, max_iter: int, xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5
) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.uint16)
    for y in range(height):
        cy = ymin + (y / (height - 1)) * (ymax - ymin)
        for x in range(width):
            cx = xmin + (x / (width - 1)) * (xmax - xmin)
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
def mandelbrot_numba_parallel(
    width: int, height: int, max_iter: int, xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5
) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.uint16)
    for y in prange(height):
        cy = ymin + (y / (height - 1)) * (ymax - ymin)
        for x in range(width):
            cx = xmin + (x / (width - 1)) * (xmax - xmin)
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


# ---------------------------
# Utils
# ---------------------------


def timeit(func, *args, repeat=1, include_compile=False):
    """
    Devuelve (elapsed_seconds, result_array). Si include_compile=False,
    hace una llamada de calentamiento (warm-up) antes de medir.
    """
    res = None
    if not include_compile:
        _ = func(*args)  # warm-up (compila la primera vez)
    t0 = time.perf_counter()
    for _ in range(repeat):
        res = func(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / repeat, res


def to_uint8_image(Z: np.ndarray, max_iter: int) -> np.ndarray:
    # Normalización simple a 0..255
    return (255 * (Z / max_iter)).astype(np.uint8)


# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Numba vs Python — Benchmark UI", layout="wide")
st.title("Numba vs Python — Demo visual con Mandelbrot")

with st.sidebar:
    st.header("Parámetros")
    width = st.slider("Ancho (px)", 200, 2000, 800, step=100)
    height = st.slider("Alto (px)", 200, 2000, 600, step=100)
    max_iter = st.slider("Iteraciones máximas", 50, 1500, 300, step=50)
    repeat = st.slider("Repeticiones para media", 1, 5, 1)
    include_compile = st.checkbox("Incluir tiempo de compilación (warm-up)", value=False)
    run_btn = st.button("Run")

    st.divider()
    st.subheader("Barrido de tamaños (opcional)")
    sweep = st.checkbox("Activar barrido")
    sizes = st.multiselect(
        "Tamaños (ancho=alto)", [300, 500, 700, 900, 1100, 1300], default=[300, 500, 700, 900]
    )
    sweep_repeat = st.slider("Repeticiones barrido", 1, 3, 1)

col1, col2 = st.columns([1, 1])

if run_btn:
    with st.spinner("Ejecutando…"):
        t_py, img_py = timeit(
            mandelbrot_py, width, height, max_iter, include_compile=include_compile, repeat=repeat
        )
        t_nb, img_nb = timeit(
            mandelbrot_numba,
            width,
            height,
            max_iter,
            include_compile=include_compile,
            repeat=repeat,
        )
        t_nbp, img_nbp = timeit(
            mandelbrot_numba_parallel,
            width,
            height,
            max_iter,
            include_compile=include_compile,
            repeat=repeat,
        )

    # Métricas
    base = t_py
    speed_nb = base / t_nb if t_nb > 0 else np.nan
    speed_nbp = base / t_nbp if t_nbp > 0 else np.nan

    with col1:
        st.subheader("Fractal (Numba paralelo)")
        st.image(to_uint8_image(img_nbp, max_iter), clamp=True, width="stretch")

    with col2:
        st.subheader("Métricas")
        st.metric("Python puro (s)", f"{t_py:0.3f}")
        st.metric("Numba (s)", f"{t_nb:0.3f}", f"{speed_nb:0.1f}× más rápido")
        st.metric("Numba paralelo (s)", f"{t_nbp:0.3f}", f"{speed_nbp:0.1f}× más rápido")
        df = pd.DataFrame(
            {
                "Método": ["Python", "Numba", "Numba paralelo"],
                "Tiempo (s)": [t_py, t_nb, t_nbp],
            }
        )
        st.bar_chart(df.set_index("Método"))

    if sweep and sizes:
        st.subheader("Barrido de tamaños (ancho=alto)")
        rows = []
        for s in sizes:
            # Para barrido usamos menos iteraciones para que no tarde mucho
            iters = max(100, min(max_iter, 400))
            t_py_s, _ = timeit(
                mandelbrot_py, s, s, iters, include_compile=False, repeat=sweep_repeat
            )
            t_nb_s, _ = timeit(
                mandelbrot_numba, s, s, iters, include_compile=False, repeat=sweep_repeat
            )
            t_nbp_s, _ = timeit(
                mandelbrot_numba_parallel, s, s, iters, include_compile=False, repeat=sweep_repeat
            )
            rows.append(
                {"Tamaño": s, "Python": t_py_s, "Numba": t_nb_s, "Numba paralelo": t_nbp_s}
            )
        dff = pd.DataFrame(rows).set_index("Tamaño")
        st.line_chart(dff)
        st.caption(
            "Consejo: mide siempre sin warm-up para comparar de forma justa. "
            "La 1.ª ejecución incluye la compilación."
        )
else:
    st.info("Configura parámetros en la barra lateral y pulsa **Run**.")
