# EPM Chat — standalone image. Build from the REPOSITORY ROOT (it reuses src/ and docs/):
#   docker build -f epm_chatbot/Dockerfile -t epm-chat .
#   docker run -p 8765:8765 -e EPM_CHAT_SECRET_KEY=... -e EPM_CHAT_ALLOWED_ORIGINS=https://your-site.com \
#              -v epm_chat_var:/app/epm_chatbot/var epm-chat
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    EPM_CHAT_HOST=0.0.0.0 EPM_CHAT_PORT=8765 MPLBACKEND=Agg

WORKDIR /app
# Runtime dependencies only (CPU PyTorch wheel keeps the image smaller).
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
      "numpy>=1.24" "pandas>=2.0" "scikit-learn>=1.3" "xgboost>=2.0" "torch>=2.0" "scipy>=1.10" "pyyaml>=6.0" "matplotlib>=3.7"

COPY src/ src/
COPY docs/ docs/
COPY configs/ configs/
COPY data/README.md data/README.md
COPY epm_chatbot/ epm_chatbot/

RUN useradd --create-home --uid 10001 epm && mkdir -p /app/epm_chatbot/var /app/data && chown -R epm /app
USER epm

# Explicitly train the shared starter models at build time (operator decision; set QUICK=1 to skip neural models).
ARG QUICK=0
RUN if [ "$QUICK" = "1" ]; then python -m epm_chatbot.bootstrap --quick; else python -m epm_chatbot.bootstrap; fi

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/api/health',timeout=4).status==200 else 1)"
CMD ["python", "-m", "epm_chatbot"]
