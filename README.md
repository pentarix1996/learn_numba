# Numba vs Python

> **Objetivo**: enseñar de forma clara y visual **qué es Numba**, **cuándo utilizarlo** y **cuánto acelera** un algoritmo con bucles numéricos sobre `NumPy`, con una **UI en Streamlit** y un **CLI de benchmark**.

---

## TL;DR (arranque rápido)

```bash
# 1) Instala dependencias (Python 3.9–3.12 recomendado)
pip install -U numba numpy pandas streamlit

# 2) Lanza la app visual
streamlit run numba_bench_app.py

# 3) (Opcional) Benchmark en CLI
python bench_numba.py
```

---

## Índice

1. [¿Qué es Numba?](#qué-es-numba)
2. [Cuándo usarlo (y cuándo no)](#cuándo-usarlo-y-cuándo-no)
3. [Contenido del repo](#contenido-del-repo)
4. [Instalación](#instalación)
5. [Script CLI](#script-cli)
6. [Kernels y patrón de uso](#kernels-y-patrón-de-uso)
7. [Metodología de medición](#metodología-de-medición)
8. [Troubleshooting](#troubleshooting)
9. [Extensiones y ideas futuras](#extensiones-y-ideas-futuras)
10. [Licencia](#licencia)

---

## ¿Qué es Numba?

**Numba** es un **compilador JIT** (just-in-time) para Python que convierte funciones numéricas escritas con **bucles + `NumPy`** en **código máquina** mediante LLVM (`Low-Level-Virtual-Machine`).

Puntos clave:
- `@njit` (a.k.a. *nopython mode*) → todo corre en nativo; si usas objetos Python no soportados, cae a *object mode* (lento).
- `parallel=True` + `prange` → paraleliza bucles a nivel de CPU (multi-core).
- `fastmath=True` → permite optimizaciones agresivas en operaciones de coma flotante (útil en kernels bien condicionados).
- La **primera ejecución** compila (es más lenta). A partir de la segunda, obtienes el **speedup real**.

**Casos típicos de mejora**: 10× a 1000× en kernels bien adaptados (bucles intensivos sobre arrays numéricos).

---

## Cuándo usarlo (y cuándo no)

**Úsalo cuando…**
- Hay **bucles** que operan sobre arrays `NumPy` y **no** tienen vectorización clara.
- El cálculo es **numérico puro** (floats/ints), con acceso por índice y sin estructuras Python complejas.
- Necesitas **multi-hilo** en CPU sin abandonar Python.

**Evítalo cuando…**
- Ya tienes una **vectorización `NumPy` limpia** y sin cuellos (Numba no va a mejorar mucho).
- El problema es de **I/O** o manipulación de objetos **pandas/strings/dicts**.
- La lógica es muy dinámica y “pythónica” (tipos cambiantes, objetos, etc.).

---

## Contenido del repo

```
.
├── README.md                    # este documento
├── requirements.txt             # Dependencias
├── numba_bench_app.py           # UI Streamlit: comparación visual y métricas
└── bench_numba.py               # CLI: benchmark rápido en consola

```

---

## Instalación

- Requiere **Python 3.9–3.12** (recomendado).
- En macOS/Windows/Linux funciona con `pip` normal.

```bash
pip -m install -r requirements.txt
```

> Sugerencia: ejecuta en un **entorno virtual** (`python -m venv .venv && source .venv/bin/activate` en Unix/macOS, o `.venv\Scripts\activate` en Windows).

---

## App UI (Streamlit)

Lanza la app:

```bash
streamlit run numba_bench_app.py
```

### ¿Qué verás?

- **Imagen del fractal de Mandelbrot** (generado por el kernel paralelo con Numba).
- **Métricas**: tiempo (Python puro / Numba / Numba paralelo), y **speedup**.
- **Gráficas**: barras comparativas y **curva de escalado** por tamaño (barrido).
- Opción **“Incluir tiempo de compilación (warm-up)”** para mostrar el coste de la 1.ª ejecución.

### Controles recomendados para la demo
- Tamaño: 800×600, `max_iter=300`.
- Marca **“Excluir tiempo de compilación”** al principio para una comparación justa.
- Activa el **barrido** (p. ej. 300, 500, 700, 900) para enseñar el escalado.

---

## Script CLI

Ejecuta:
```bash
python bench_numba.py
```

---

## Kernels y patrón de uso

**Patrón mínimo** para convertir bucles Python a nativo:

1. Escribe la función “limpia” sobre `np.ndarray` (sin listas/dicts/objetos dinámicos).
2. Añade `@njit(cache=True)`; si procede, usa `@njit(parallel=True, fastmath=True)`.
3. Cambia el bucle exterior a `for i in prange(n)` cuando tenga sentido paralelizar.
4. **Mide sin incluir el warm-up y posteriormente incluye el warm-up** (la 1.ª ejecución compila).

> Nota: el ejemplo usa el fractal de **Mandelbrot** porque es un kernel con **mucho bucle** y sin I/O, ideal para enseñar speedup.

---

**FAQ rápidas**
- **¿Libera GIL?** Sí en *nopython mode* y con `parallel=True` puede usar varios hilos.
- **¿Con pandas?** No directamente; extrae `.values` y opera en `np.ndarray`.
- **¿GPU?** Existe `numba.cuda` para NVIDIA (da para otra mini-review).
- **¿Random?** Subconjunto de `np.random` soportado.

---

## Metodología de medición

- Reloj: `time.perf_counter()`.
- **Warm-up**: no cuenta la 1.ª llamada (compilación). En la app puedes incluir/excluirlo.
- **Repeat**: promedia varias repeticiones si quieres estabilidad.
- **Paralelismo**: `parallel=True` + `prange` en el bucle exterior si cada iteración es independiente.
- **Comparabilidad**: misma entrada, mismo `dtype` y mismos límites/iteraciones.

---

## Troubleshooting

- **“llvmlite/Numba mismatch”**: reinstala `numba` que ya trae la versión de `llvmlite` adecuada (`pip install -U numba`).
- **Python no soportado**: usa 3.9–3.12. Si usas una versión más nueva y falla, cambia a una soportada.
- **Apple Silicon (M1/M2/M3)**: funciona con `pip` normal; si ves errores, actualiza `pip` y prueba a reinstalar en un venv limpio.
- **Caída a object-mode**: suele indicar que hay objetos Python no soportados dentro del hot-loop; pasa todo a `np.ndarray` y tipos primitivos.
- **Resultados distintos con `fastmath`**: desactívalo si necesitas reproducibilidad bit a bit (usa el decorador sin `fastmath=True`).

---

## Extensiones y ideas futuras

- **GPU**: portar el kernel a `numba.cuda` (NVIDIA) para comparar CPU vs GPU.
- **Otros kernels**: convoluciones 2D, filtros (Sobel/Gauss), simulaciones Monte Carlo, autómatas celulares.
- **Benchmarks sistemáticos**: integrar `pytest-benchmark` o `asv` (airspeed velocity).
- **Polars/DuckDB**: comparar preprocesado de datos + kernel numérico con Numba.

---

## Licencia

Este repositorio se publica bajo licencia **MIT**. Si lo reutilizas para formaciones internas, ¡bienvenido sea!

---

### Créditos

Demo y materiales preparados con mera intención de divulgación técnica. **Numba en Python — de bucles a nativo**.
