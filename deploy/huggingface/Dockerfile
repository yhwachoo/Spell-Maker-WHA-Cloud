# Hugging Face Space (Docker) — WHA · Reconocedor de Sellos
# Clona el repo (que ya incluye el clasificador compacto models/sign_clf.joblib),
# instala solo las dependencias de runtime (sin torch) y sirve la app Flask con
# gunicorn en el puerto 7860 que espera Hugging Face.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# HF ejecuta el contenedor como uid 1000
RUN useradd -m -u 1000 user

WORKDIR /app

# Ajusta la rama cuando fusiones a main
ARG REPO_URL=https://github.com/yhwachoo/Spell-Maker-WHA-Cloud
ARG REPO_BRANCH=claude/github-summary-review-l0si88
RUN git clone --depth 1 --branch ${REPO_BRANCH} ${REPO_URL} .

RUN pip install --no-cache-dir -r deploy/huggingface/requirements.txt

# El repo trae models/sign_clf.joblib (~6 MB). Si faltara, se entrena en el build.
RUN test -f models/sign_clf.joblib || python train_sign_clf.py --synth-seals 150

RUN chown -R user:user /app
USER user

EXPOSE 7860
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "--timeout", "180", "webapp.app:app"]
