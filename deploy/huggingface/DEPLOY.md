# Desplegar en Hugging Face Spaces

La app es Flask + scikit-learn, así que se despliega como **Space tipo Docker**
(no hay límite de tamaño como en serverless). El contenedor clona este repo —que ya
incluye el clasificador compacto `models/sign_clf.joblib` (~6 MB)— y lo sirve con
gunicorn en el puerto 7860.

## Opción A — Space que clona el repo (recomendada, 2 archivos)

1. Crea un Space nuevo en https://huggingface.co/new-space
   - **SDK: Docker** → *Blank*.
   - Visibilidad: pública o privada (a tu gusto).
2. En el Space, sube estos **dos archivos** (de esta carpeta):
   - `Dockerfile`
   - `README.md`  (el frontmatter de arriba configura `sdk: docker` y `app_port: 7860`)
3. El build clona el repo, instala dependencias y arranca. En ~2-4 min tendrás la URL
   pública (`https://<usuario>-<space>.hf.space`).

> **Repo privado:** si `yhwachoo/Spell-Maker-WHA-Cloud` es privado, el `git clone` del
> build fallará. Hazlo público, o en Settings → Secrets del Space añade `GIT_TOKEN` y
> cambia la línea de clone del Dockerfile a:
> `RUN git clone --depth 1 --branch ${REPO_BRANCH} https://x-access-token:${GIT_TOKEN}@github.com/yhwachoo/Spell-Maker-WHA-Cloud .`
>
> **Rama:** el Dockerfile clona `claude/github-summary-review-l0si88`. Cuando fusiones a
> `main`, cambia `REPO_BRANCH` a `main`.

## Opción B — Subir el repo entero al Space

Si prefieres no clonar en el build, empuja todo el repo al Space (los Spaces son repos git):

```bash
git remote add space https://huggingface.co/spaces/<usuario>/<space>
git push space claude/github-summary-review-l0si88:main
```

Y en el Space usa un `Dockerfile` sin el `git clone` (copiando el contexto local):

```dockerfile
FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r deploy/huggingface/requirements.txt
RUN test -f models/sign_clf.joblib || python train_sign_clf.py --synth-seals 150
RUN chown -R user:user /app
USER user
EXPOSE 7860
CMD ["gunicorn","-w","2","-b","0.0.0.0:7860","--timeout","180","webapp.app:app"]
```

Coloca ese `Dockerfile` y el `README.md` (con frontmatter) en la raíz del Space.

## Probar localmente el contenedor

```bash
docker build -f deploy/huggingface/Dockerfile -t wha-sellos .
docker run -p 7860:7860 wha-sellos      # http://localhost:7860
```
