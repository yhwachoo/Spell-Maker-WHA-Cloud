"""
Descriptor de forma para un recorte de signo — ROTACIONALMENTE INVARIANTE.

Motivacion: el mismo signo aparece girado segun su posicion en el anillo, y a baja
resolucion (catalogo escaneado) el template matching por IoU se estanca (~37% cross
-domain). En vez de comparar pixeles, se describe cada signo con rasgos geometricos
invariantes a rotacion/escala/traslacion y se entrena un clasificador (train_sign_clf.py).

IMPORTANTE (lore): la invariancia es a ROTACION, no a ESPEJO. Espejar un signo cambia
su significado, asi que el descriptor no se fuerza a ser invariante a reflexion (y el
aumento de datos nunca voltea).

Vector de rasgos (~60 dims):
  - Histograma polar (radio x angulo) con |FFT| sobre el angulo -> invariante a rotacion.
  - Perfil radial de densidad de tinta (naturalmente invariante a rotacion).
  - 7 momentos de Hu en escala log.
  - Elongacion (razon de autovalores de inercia), fill ratio, densidad, nº de huecos.
"""
import numpy as np
from scipy import ndimage

R_BINS = 6          # anillos radiales
A_BINS = 24         # sectores angulares
N_FREQ = 6          # componentes de frecuencia angular que se conservan


def _largest_component(mask):
    lbl, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    return lbl == (int(np.argmax(sizes)) + 1)


def _hu_log(binary):
    ys, xs = np.nonzero(binary)
    if len(xs) < 5:
        return np.zeros(7)
    xb, yb = xs.mean(), ys.mean()
    x, y = xs - xb, ys - yb
    m00 = float(len(xs))

    def mu(p, q):
        return np.sum((x ** p) * (y ** q))

    def eta(p, q):
        return mu(p, q) / (m00 ** (1 + (p + q) / 2.0))

    n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
    n30, n12, n21, n03 = eta(3, 0), eta(1, 2), eta(2, 1), eta(0, 3)
    h = [0.0] * 7
    h[0] = n20 + n02
    h[1] = (n20 - n02) ** 2 + 4 * n11 ** 2
    h[2] = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h[3] = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h[4] = (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) + \
           (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    h[5] = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + \
           4 * n11 * (n30 + n12) * (n21 + n03)
    h[6] = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) - \
           (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    return np.array([-np.sign(v) * np.log10(abs(v) + 1e-30) for v in h])


def _polar_fft(binary, cy, cx, rmax):
    """Histograma polar (R_BINS x A_BINS) -> |FFT| angular -> (R_BINS x N_FREQ)."""
    ys, xs = np.nonzero(binary)
    if len(xs) < 5:
        return np.zeros(R_BINS * N_FREQ), np.zeros(R_BINS)
    r = np.hypot(xs - cx, ys - cy) / (rmax + 1e-9)
    th = np.arctan2(ys - cy, xs - cx)
    rb = np.clip((r * R_BINS).astype(int), 0, R_BINS - 1)
    ab = np.clip(((th + np.pi) / (2 * np.pi) * A_BINS).astype(int), 0, A_BINS - 1)
    hist = np.zeros((R_BINS, A_BINS))
    np.add.at(hist, (rb, ab), 1.0)
    ring_density = hist.sum(axis=1)
    ring_density = ring_density / (ring_density.sum() + 1e-9)
    # |FFT| sobre el eje angular: la rotacion es un corrimiento -> magnitud invariante.
    mag = np.abs(np.fft.rfft(hist, axis=1))[:, :N_FREQ]
    # normaliza cada anillo por su componente 0 (densidad) para robustez de escala
    mag = mag / (mag[:, :1] + 1e-9)
    return mag.flatten(), ring_density


def _inertia_elongation(binary):
    ys, xs = np.nonzero(binary)
    if len(xs) < 5:
        return 0.0
    x, y = xs - xs.mean(), ys - ys.mean()
    cov = np.array([[np.mean(x * x), np.mean(x * y)], [np.mean(x * y), np.mean(y * y)]])
    w = np.linalg.eigvalsh(cov)
    w = np.sort(np.abs(w))
    return float(w[0] / (w[1] + 1e-9))   # 0=linea, 1=isotropo (rot-invariante)


def features(binary):
    """binary: np.bool_ 2D (tinta=True). Devuelve vector de rasgos (float64)."""
    binary = np.asarray(binary, bool)
    ys, xs = np.nonzero(binary)
    if len(xs) < 5:
        return np.zeros(R_BINS * N_FREQ + R_BINS + 7 + 4)
    # recorte al bbox y centrado por centroide
    cy, cx = ys.mean(), xs.mean()
    rmax = np.hypot(xs - cx, ys - cy).max()

    pf, ring = _polar_fft(binary, cy, cx, rmax)
    hu = _hu_log(binary)

    area = float(binary.sum())
    h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
    fill = area / (h * w + 1e-9)                       # densidad dentro del bbox
    elong = _inertia_elongation(binary)
    # nº de huecos (Euler): agujeros internos discriminan Orb, Eye, Window...
    filled = ndimage.binary_fill_holes(binary)
    holes = float((filled & ~binary).sum()) / (area + 1e-9)
    density = area / (np.pi * rmax ** 2 + 1e-9)         # tinta / disco envolvente

    extra = np.array([fill, elong, holes, density])
    return np.concatenate([pf, ring, hu, extra]).astype(np.float64)


FEATURE_DIM = R_BINS * N_FREQ + R_BINS + 7 + 4
