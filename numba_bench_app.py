import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from numba import njit, prange

# =========================================
# Utils comunes
# =========================================


def timeit(func, *args, repeat=1, include_compile=False):
    """
    Devuelve (elapsed_seconds, result). Si include_compile=False,
    hace una llamada de calentamiento (warm-up) antes de medir.
    """
    res = None
    if not include_compile:
        _ = func(*args)  # warm-up (compila la 1ª vez si es Numba)
    t0 = time.perf_counter()
    for _ in range(repeat):
        res = func(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / repeat, res


def bars_time_chart(df_methods: pd.DataFrame):
    # df_methods: columns ["Método","Tiempo (s)"]
    chart = (
        alt.Chart(df_methods)
        .mark_bar()
        .encode(
            x=alt.X("Método:N", title="Método"),
            y=alt.Y("Tiempo (s):Q", title="Tiempo (s)"),
            tooltip=["Método", alt.Tooltip("Tiempo (s):Q", format=".3f")],
        )
    )
    return chart


def sweep_line_chart(df_long: pd.DataFrame, x_label: str, log_y: bool):
    # df_long: columns [X, "Método","Tiempo (s)"]
    y_enc = alt.Y(
        "Tiempo (s):Q", title="Tiempo (s)", scale=alt.Scale(type="log") if log_y else alt.Scale()
    )
    x_col = df_long.columns[0]
    chart = (
        alt.Chart(df_long)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_label),
            y=y_enc,
            color="Método:N",
            tooltip=[x_col, "Método", alt.Tooltip("Tiempo (s):Q", format=".3f")],
        )
    )
    return chart


# =========================================
# TEST 1: Mandelbrot
# =========================================


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


def to_uint8_image(Z: np.ndarray, max_iter: int) -> np.ndarray:
    # Normalización simple a 0..255
    return (255 * (Z / max_iter)).astype(np.uint8)


# =========================================
# TEST 2: DataFrame sintético (loops por fila)
# =========================================


def gen_synthetic_df(n_rows: int, spec: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    spec: DataFrame con columnas: name (str), dtype (one of 'float','int','bool')
    Genera un DataFrame con datos aleatorios según dtype.
    *La generación NO se incluye en la medición.*
    """
    rng = np.random.default_rng(seed)
    data = {}
    for _, row in spec.iterrows():
        name = str(row["name"])
        dtype = str(row["dtype"]).lower().strip()
        if dtype == "float":
            data[name] = rng.random(n_rows) * 100.0
        elif dtype == "int":
            data[name] = rng.integers(0, 100, size=n_rows)
        elif dtype == "bool":
            data[name] = rng.integers(0, 2, size=n_rows).astype(bool)
        else:
            data[name] = rng.random(n_rows) * 100.0
    return pd.DataFrame(data)


def to_numeric_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Convierte columnas numéricas y booleanas a una matriz float64 (bool->0/1).
    Ignora columnas no numéricas si las hubiese.
    """
    df_num = df.select_dtypes(include=["number", "bool"]).copy()
    for c in df_num.columns:
        if df_num[c].dtype == bool:
            df_num[c] = df_num[c].astype(np.uint8)
    return df_num.to_numpy(dtype=np.float64, copy=False), list(df_num.columns)


def row_score_py(X: np.ndarray) -> np.ndarray:
    """
    Versión Python pura: bucles por fila y columna.
    score_i = sum_j sqrt(v^2 + 3) / (1 + 0.1 * v^2)
    """
    n_rows, n_cols = X.shape
    out = np.empty(n_rows, dtype=np.float64)
    for i in range(n_rows):
        s = 0.0
        row = X[i]
        for j in range(n_cols):
            v = row[j]
            s += ((v * v + 3.0) ** 0.5) / (1.0 + 0.1 * v * v)
        out[i] = s
    return out


@njit(cache=True)
def row_score_nb(X: np.ndarray) -> np.ndarray:
    n_rows, n_cols = X.shape
    out = np.empty(n_rows, dtype=np.float64)
    for i in range(n_rows):
        s = 0.0
        for j in range(n_cols):
            v = X[i, j]
            s += ((v * v + 3.0) ** 0.5) / (1.0 + 0.1 * v * v)
        out[i] = s
    return out


@njit(parallel=True, fastmath=True, cache=True)
def row_score_nbp(X: np.ndarray) -> np.ndarray:
    n_rows, n_cols = X.shape
    out = np.empty(n_rows, dtype=np.float64)
    for i in prange(n_rows):
        s = 0.0
        for j in range(n_cols):
            v = X[i, j]
            s += ((v * v + 3.0) ** 0.5) / (1.0 + 0.1 * v * v)
        out[i] = s
    return out


def summarize_scores(arr: np.ndarray) -> pd.DataFrame:
    q50 = float(np.percentile(arr, 50))
    q95 = float(np.percentile(arr, 95))
    summary = {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": q50,
        "p95": q95,
        "max": float(np.max(arr)),
    }
    return pd.DataFrame([summary])


# =========================================
# UI
# =========================================

st.set_page_config(page_title="Numba vs Python — Benchmark UI", layout="wide")
st.title("Numba vs Python — Demo visual (Mandelbrot & DataFrame sintético)")

with st.sidebar:
    st.header("Tipo de test")
    test_type = st.selectbox(
        "Selecciona el test",
        ["Fractal de Mandelbrot", "DataFrame sintético (loops por fila)"],
        index=0,
    )

    st.header("Parámetros comunes")
    repeat = st.number_input(
        "Repeticiones para media", min_value=1, max_value=10, value=1, step=1, format="%d"
    )
    include_compile = st.checkbox("Incluir tiempo de compilación (warm-up)", value=False)
    run_btn = st.button("Run")

    st.divider()

    if test_type == "Fractal de Mandelbrot":
        st.subheader("Parámetros — Mandelbrot")
        width = st.slider("Ancho (px)", 200, 2000, 800, step=100)
        height = st.slider("Alto (px)", 200, 2000, 600, step=100)
        max_iter = st.slider("Iteraciones máximas", 50, 1500, 300, step=50)

        st.subheader("Barrido (opcional)")
        sweep = st.checkbox("Activar barrido de tamaños")
        sizes = st.multiselect(
            "Tamaños (ancho=alto)",
            [300, 500, 700, 900, 1100, 1300],
            default=[300, 500, 700, 900],
        )
        sweep_repeat = st.number_input(
            "Repeticiones barrido", min_value=1, max_value=5, value=1, step=1, format="%d"
        )
        log_y = st.checkbox("Escala logarítmica (Y)", value=True)

    else:
        st.subheader("Parámetros — DF sintético")
        n_rows = st.number_input(
            "Nº de filas",
            min_value=1_000,
            max_value=5_000_000,
            value=200_000,
            step=10_000,
            format="%d",
        )

        st.caption(
            "Define columnas (nombre + tipo: float, int, bool). Puedes añadir/eliminar filas."
        )
        default_spec = pd.DataFrame(
            {"name": ["x1", "x2", "x3"], "dtype": ["float", "float", "int"]}
        )
        spec = st.data_editor(
            default_spec,
            num_rows="dynamic",
            column_config={
                "name": st.column_config.TextColumn("Columna"),
                "dtype": st.column_config.SelectboxColumn(
                    "Tipo", options=["float", "int", "bool"], required=True
                ),
            },
            key="spec_editor",
        )

        st.subheader("Barrido (opcional)")
        sweep = st.checkbox("Activar barrido de filas")
        sizes = st.multiselect(
            "Filas a probar",
            [50_000, 100_000, 200_000, 300_000, 500_000, 800_000],
            default=[50_000, 100_000, 200_000, 300_000],
        )
        sweep_repeat = st.number_input(
            "Repeticiones barrido", min_value=1, max_value=5, value=1, step=1, format="%d"
        )
        log_y = st.checkbox("Escala logarítmica (Y)", value=True)

col1, col2 = st.columns([1, 1])

if run_btn:
    with st.spinner("Ejecutando…"):
        if test_type == "Fractal de Mandelbrot":
            # --- medir mandelbrot ---
            t_py, img_py = timeit(
                mandelbrot_py,
                width,
                height,
                max_iter,
                include_compile=include_compile,
                repeat=repeat,
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
                dfm = pd.DataFrame(
                    {
                        "Método": ["Python", "Numba", "Numba paralelo"],
                        "Tiempo (s)": [t_py, t_nb, t_nbp],
                    }
                )
                st.altair_chart(bars_time_chart(dfm), use_container_width=True)

            if sweep and sizes:
                st.subheader("Barrido de tamaños (ancho=alto)")
                rows = []
                for s in sizes:
                    iters = max(100, min(max_iter, 400))
                    t_py_s, _ = timeit(
                        mandelbrot_py, s, s, iters, include_compile=False, repeat=sweep_repeat
                    )
                    t_nb_s, _ = timeit(
                        mandelbrot_numba, s, s, iters, include_compile=False, repeat=sweep_repeat
                    )
                    t_nbp_s, _ = timeit(
                        mandelbrot_numba_parallel,
                        s,
                        s,
                        iters,
                        include_compile=False,
                        repeat=sweep_repeat,
                    )
                    rows.append(
                        {"Tamaño": s, "Python": t_py_s, "Numba": t_nb_s, "Numba paralelo": t_nbp_s}
                    )
                dff = pd.DataFrame(rows)
                df_long = dff.melt(id_vars=["Tamaño"], var_name="Método", value_name="Tiempo (s)")
                st.altair_chart(
                    sweep_line_chart(df_long, x_label="Tamaño (px)", log_y=log_y),
                    use_container_width=True,
                )
                st.caption(
                    "Nota: el tiempo mostrado excluye la compilación y usa iteraciones "
                    "limitadas para el barrido."
                )

        else:
            # --- generar DF fuera de la medición ---
            df = gen_synthetic_df(n_rows, spec)
            X, used_cols = to_numeric_matrix(df)  # bool->0/1, todo a float64

            # --- medir scoring por filas ---
            t_py, res_py = timeit(row_score_py, X, include_compile=include_compile, repeat=repeat)
            t_nb, res_nb = timeit(row_score_nb, X, include_compile=include_compile, repeat=repeat)
            t_nbp, res_nbp = timeit(
                row_score_nbp, X, include_compile=include_compile, repeat=repeat
            )

            base = t_py
            speed_nb = base / t_nb if t_nb > 0 else np.nan
            speed_nbp = base / t_nbp if t_nbp > 0 else np.nan

            with col1:
                st.subheader("Preview DF")
                st.write(f"Columnas usadas para el cálculo: {used_cols}")
                st.dataframe(df.head(8))
                st.caption("La generación del DF no se incluye en el tiempo medido.")

                st.subheader("Resumen de puntuaciones (Numba paralelo)")
                st.dataframe(summarize_scores(res_nbp))

            with col2:
                st.subheader("Métricas")
                st.metric("Python puro (s)", f"{t_py:0.3f}")
                st.metric("Numba (s)", f"{t_nb:0.3f}", f"{speed_nb:0.1f}× más rápido")
                st.metric("Numba paralelo (s)", f"{t_nbp:0.3f}", f"{speed_nbp:0.1f}× más rápido")
                dfm = pd.DataFrame(
                    {
                        "Método": ["Python", "Numba", "Numba paralelo"],
                        "Tiempo (s)": [t_py, t_nb, t_nbp],
                    }
                )
                st.altair_chart(bars_time_chart(dfm), use_container_width=True)

            if sweep and sizes:
                st.subheader("Barrido de filas")
                rows = []
                for n in sizes:
                    df_s = gen_synthetic_df(n, spec, seed=12345)
                    X_s, _ = to_numeric_matrix(df_s)
                    t_py_s, _ = timeit(
                        row_score_py, X_s, include_compile=False, repeat=sweep_repeat
                    )
                    t_nb_s, _ = timeit(
                        row_score_nb, X_s, include_compile=False, repeat=sweep_repeat
                    )
                    t_nbp_s, _ = timeit(
                        row_score_nbp, X_s, include_compile=False, repeat=sweep_repeat
                    )
                    rows.append(
                        {"Filas": n, "Python": t_py_s, "Numba": t_nb_s, "Numba paralelo": t_nbp_s}
                    )
                dff = pd.DataFrame(rows)
                df_long = dff.melt(id_vars=["Filas"], var_name="Método", value_name="Tiempo (s)")
                st.altair_chart(
                    sweep_line_chart(df_long, x_label="Filas", log_y=log_y),
                    use_container_width=True,
                )
                st.caption(
                    "Tiempo de cálculo por filas (excluye generación de datos y compilación)."
                )

else:
    st.info("Configura parámetros en la barra lateral, elige el tipo de test y pulsa **Run**.")
