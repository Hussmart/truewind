# Container image for the TrueWind Streamlit dashboard.
#
#   docker build -t truewind .
#   docker run --rm -p 8501:8501 truewind
#
# then open http://localhost:8501

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the package and its dashboard extra first, so code-only changes
# to the dashboard don't invalidate the dependency layer.
COPY pyproject.toml README.md ./
COPY truewind ./truewind
RUN pip install ".[dashboard]"

COPY dashboard ./dashboard

RUN useradd --create-home --uid 1000 truewind
USER truewind

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)" || exit 1

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
