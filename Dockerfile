FROM python:3.5-slim

# Old Debian repos for this legacy base
RUN echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99no-check-valid-until \
 && printf 'deb http://archive.debian.org/debian buster main contrib non-free\n\
deb http://archive.debian.org/debian buster-updates main contrib non-free\n\
deb http://archive.debian.org/debian-security buster/updates main contrib non-free\n' > /etc/apt/sources.list

# System deps (BLAS/LAPACK); no GUI libs needed
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gfortran pkg-config \
      libopenblas-base liblapack3 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Locale + avoid CPU oversubscription + headless MPL
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app
COPY . /app

# Tooling that still supports Python 3.5
RUN python -m pip install --upgrade pip==20.3.4 setuptools==50.3.2 wheel==0.37.1

# Pin a Py3.5-compatible scientific + plotting stack
RUN printf '%s\n' \
  'numpy==1.17.5' \
  'scipy==1.3.3' \
  'scikit-learn==0.21.3' \
  'joblib==0.14.1' \
  'pandas==0.25.3' \
  'matplotlib==2.2.5' \
  'cycler==0.10.0' \
  'kiwisolver==1.1.0' \
  'python-dateutil==2.8.1' \
  'pyparsing==2.4.7' \
  > /tmp/constraints.txt

# 1) Install the pinned stack first (pulls wheels; no compiling)
RUN pip install -c /tmp/constraints.txt numpy scipy scikit-learn joblib pandas matplotlib cycler kiwisolver python-dateutil pyparsing

# 2) Install the rest of your deps but IGNORE pins to numpy/scipy/sklearn/pandas/joblib/matplotlib
RUN grep -viE '^(numpy|scipy|scikit-learn|pandas|joblib|matplotlib)(\s|[=<>].*)?$' requirements.txt > /tmp/requirements.no-sci.txt || true \
 && pip install -r /tmp/requirements.no-sci.txt -c /tmp/constraints.txt \
 && pip check

ENTRYPOINT ["python", "run.py"]
CMD ["test1"]