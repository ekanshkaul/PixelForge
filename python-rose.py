import argparse
import time
from functools import lru_cache

import numpy as np

# --------------------------------------------------------------------------
#  Timeline
# --------------------------------------------------------------------------
T_STEM = (0.0, 4.5)
T_BUD = (3.5, 5.5)
T_BLOOM = (5.5, 15.0)
T_END = T_BLOOM[1]
T_DROPS = (13.0, 16.0)
T_FALL0, FALL_DUR = 16.5, 6.0
T_LOOP = T_FALL0 + FALL_DUR + 4.0
SPIN_DEG_PER_SEC = 14.0

H = 2.6          # stem length (stem runs from z=-H up to z=0, flower sits at origin)
GOLDEN = np.deg2rad(137.50776)


# --------------------------------------------------------------------------
#  Small helpers
# --------------------------------------------------------------------------
def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def lerp(a, b, t):
    return a + (b - a) * t


def span(t, a, b):
    """0..1 progress of time t inside window [a, b]."""
    return float(np.clip((t - a) / (b - a), 0.0, 1.0))


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def mix(c0, c1, t):
    """Blend two RGB colours; t is an array broadcastable to (...,1)."""
    c0, c1, t = np.asarray(c0, float), np.asarray(c1, float), np.asarray(t, float)
    return c0 + (c1 - c0) * t[..., None]


# A "surface" is a tuple (P, C): P has shape (nS, nT, 3) points, C has shape
# (nS, nT, 3) RGB in 0..1.  Everything in the scene is built from these.


# --------------------------------------------------------------------------
#  Generic tube  (stem, thorns, leaf stalks)
# --------------------------------------------------------------------------
def tube(centres, radii, colour_fn, n_around=14):
    centres = np.asarray(centres, float)
    nS = len(centres)
    T = unit(np.gradient(centres, axis=0))
    ref = np.where(np.abs(T[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    e1 = unit(np.cross(T, ref))
    e2 = np.cross(T, e1)
    a = np.linspace(0, 2 * np.pi, n_around)[None, :, None]
    P = (centres[:, None, :]
         + radii[:, None, None] * (np.cos(a) * e1[:, None, :] + np.sin(a) * e2[:, None, :]))
    s = np.linspace(0, 1, nS)[:, None] * np.ones((1, n_around))
    return P, colour_fn(s, a[..., 0] * np.ones((nS, 1)))


# --------------------------------------------------------------------------
#  Stem
# --------------------------------------------------------------------------
def stem_xy(z):
    # gentle bow that is vertical at both ends -> the spin axis passes through base AND flower
    return 0.11 * (1 - np.cos(2 * np.pi * z / H)), 0.06 * np.sin(np.pi * z / H)


def stem_radius(z):
    taper = 0.054 - 0.013 * (z + H) / H
    hip = 0.115 * np.exp(-((z + 0.075) / 0.12) ** 2)          # rose hip under the flower
    return taper + hip


STEM_TOP = 0.03


def stem_z_top(p):
    return -H + (H + STEM_TOP) * p


def stem_surface(p, nZ=70):
    s = np.linspace(0, 1, nZ)
    z = -H + (H + STEM_TOP) * p * s
    cx, cy = stem_xy(z)
    e = 0.03
    cap = np.sqrt(np.clip(1 - ((s - (1 - e)) / e) ** 2, 0, 1)) * (s > 1 - e) + (s <= 1 - e)
    r = stem_radius(z) * cap
    centres = np.stack([cx, cy, z], -1)

    def col(sv, av):
        base = mix((0.10, 0.25, 0.08), (0.20, 0.40, 0.13), np.clip(sv * 1.2, 0, 1))
        redden = np.clip((sv - 0.6) * 1.3, 0, 1) * 0.35 * (0.5 + 0.5 * np.sin(av * 1.0 + 0.5))
        c = mix(base, (0.38, 0.12, 0.08), redden)
        return c * (0.8 + 0.2 * np.cos(av - 0.8))[..., None]   # cheap form shading

    return tube(centres, r, col, n_around=18)


def thorn_surfaces(p):
    out = []
    ztop = stem_z_top(p)
    for k in range(9):
        zk = -H + 0.35 + k * 0.27
        phi = k * GOLDEN + 0.7
        g = smoothstep((ztop - zk - 0.05) / 0.4)
        if g <= 0.01:
            continue
        cx, cy = stem_xy(zk)
        rad = stem_radius(zk)
        out_dir = np.array([np.cos(phi), np.sin(phi), 0.0])
        d = unit(out_dir * 0.85 + np.array([0, 0, -0.5]))
        base = np.array([cx, cy, zk]) + out_dir * rad * 0.7
        h = 0.12 * g
        s = np.linspace(0, 1, 5)
        centres = base + d * (h * s)[:, None] + np.array([0, 0, -1.0]) * (0.05 * h * s ** 2)[:, None]
        radii = 0.032 * g * (1 - s) ** 1.2 + 1e-4
        out.append(tube(centres, radii,
                        lambda sv, av: mix((0.42, 0.10, 0.07), (0.62, 0.22, 0.12), sv),
                        n_around=8))
    return out


# --------------------------------------------------------------------------
#  Leaves  (compound rose leaves: stalk + 5 serrated leaflets)
# --------------------------------------------------------------------------
def leaflet(O, d, L, W, droop, fold, nS=30, nT=11, phase=0.0):
    zhat = np.array([0, 0, 1.0])
    d = unit(d)
    side = unit(np.cross(zhat, d))
    n = np.cross(d, side)
    s = np.linspace(0, 1, nS)[:, None]
    t = np.linspace(-1, 1, nT)[None, :]
    w = W * np.sin(np.pi * s ** 0.8) ** 0.85
    w = w * (1 + 0.07 * np.abs(np.sin(s * np.pi * 9 + phase)) * (s > 0.08))   # serrated edge
    u = t * w
    P = (np.asarray(O)[None, None, :]
         + (L * s)[..., None] * d[None, None, :]
         + u[..., None] * side[None, None, :]
         + (fold * np.abs(t) * w)[..., None] * n[None, None, :]
         - (droop * (L * s) ** 2)[..., None] * zhat[None, None, :])
    vein = np.exp(-(t * 5.0) ** 2) * 0.10
    c = mix((0.06, 0.22, 0.07), (0.16, 0.42, 0.14), np.clip(s * 0.8 + 0.2, 0, 1) * np.ones_like(t))
    c = c + vein[..., None] * np.array([0.5, 0.8, 0.3])
    c = mix(c, (0.35, 0.10, 0.08), 0.12 * (np.abs(t) ** 3 * np.ones_like(s)))   # reddish rim
    return P, np.clip(c, 0, 1)


LEAVES = [(-2.05, 0.3), (-1.60, 0.3 + GOLDEN), (-1.15, 0.3 + 2 * GOLDEN), (-0.72, 0.3 + 3 * GOLDEN)]


def leaf_surfaces(p):
    out = []
    ztop = stem_z_top(p)
    for z_attach, phi in LEAVES:
        g = smoothstep((ztop - z_attach - 0.05) / 0.9)
        if g <= 0.01:
            continue
        cx, cy = stem_xy(z_attach)
        rad = stem_radius(z_attach)
        radial = np.array([np.cos(phi), np.sin(phi), 0.0])
        O = np.array([cx, cy, z_attach]) + radial * rad * 0.8
        el = np.deg2rad(lerp(82, 28, g))
        d0 = unit(radial * np.cos(el) + np.array([0, 0, np.sin(el)]))
        sc = 0.25 + 0.75 * g
        droop = lerp(0.0, 0.32, g)
        Lr = 0.62 * sc

        def rach(k):
            return O + d0 * (k * Lr) - np.array([0, 0, 1.0]) * droop * (k * Lr) ** 2

        # stalk
        ks = np.linspace(0, 1, 8)
        pts = np.stack([rach(k) for k in ks])
        out.append(tube(pts, 0.014 * sc * (1 - 0.4 * ks) + 1e-4,
                        lambda sv, av: mix((0.12, 0.28, 0.09), (0.18, 0.36, 0.12), sv), n_around=6))

        zax = np.array([0, 0, 1.0])
        rot = lambda v, ang: (v * np.cos(ang) + np.cross(zax, v) * np.sin(ang))
        tang = unit(d0 - zax * 2 * droop * Lr * 0.5)
        for k, size, ang in [(0.42, 0.75, 0.95), (0.42, 0.75, -0.95), (0.72, 0.9, 0.8),
                             (0.72, 0.9, -0.8), (1.0, 1.0, 0.0)]:
            dl = rot(tang, ang)
            dl = unit(dl + zax * 0.15)
            out.append(leaflet(rach(k), dl, L=0.54 * size * sc, W=0.18 * size * sc,
                               droop=droop * 1.6, fold=0.55, phase=k * 7 + ang))
    return out


# --------------------------------------------------------------------------
#  Petal / sepal surface
# --------------------------------------------------------------------------
def petal(L, W, phi, base_r, base_z, th0, bend, tip, cup, ruffle=0.0, phase=0.0,
          profile=(0.6, 5.0, 0.5), nS=26, nT=19, lip=1.7, asym=0.0):
    """
    One curved, cupped, wrapped petal.

    Centre line lives in the (radius, height) half-plane:  its angle from
    vertical is  th0 + bend*s^1.6 + tip*smoothstep(s)  (positive = leaning
    outward).  Across the width the sheet is cupped towards the flower axis,
    flares back out near the lip, and wraps around the bud (arc-length based,
    so petals hug the bud when closed and lie flat when open).
    Grid winding: the concave (inner) face is the front face.
    """
    s = np.linspace(0, 1, nS)[:, None]
    t = np.linspace(-1, 1, nT)[None, :]
    a, p, b = profile
    w = W * s ** a * np.clip(1 - s ** p, 0, None) ** b

    sc = s[:, 0]
    theta = th0 + bend * sc ** 1.6 + tip * smoothstep((sc - 0.65) / 0.35)
    ds = L / (nS - 1)
    dr, dz = np.sin(theta) * ds, np.cos(theta) * ds
    rc = base_r + np.concatenate([[0], np.cumsum(0.5 * (dr[1:] + dr[:-1]))])
    zc = base_z + np.concatenate([[0], np.cumsum(0.5 * (dz[1:] + dz[:-1]))])
    rc, zc, th = rc[:, None], zc[:, None], theta[:, None]

    u = t * w * (1 + asym * t)                                   # asymmetric petals look hand-made
    cup_s = cup * (1 - lip * smoothstep((s - 0.62) / 0.38))      # lip flares outward only when open
    inward = cup_s * u ** 2 / max(W, 1e-6)
    inward = inward + ruffle * np.abs(t) ** 2 * np.sin(6.5 * np.pi * s + phase) * w * smoothstep((s - 0.3) / 0.4)

    r = np.maximum(rc - np.cos(th) * inward, 0.015)
    z = zc + np.sin(th) * inward

    rho = np.maximum(rc, 0.14)
    amax = 1.5
    alpha = amax * np.tanh(u / (rho * amax))
    ang = phi + alpha
    P = np.stack([r * np.cos(ang), r * np.sin(ang), z], -1)
    return P, s * np.ones_like(t), t * np.ones_like(s)


def petal_colour(s, t, frac, jitter, brown, back=False):
    """Inner face = rich velvety red.  Outer face = paler, more matte and a bit pinker."""
    deep = np.array([0.20, 0.0, 0.03])
    mid = np.array([0.66, 0.02, 0.08])
    light = np.array([0.90, 0.10, 0.16])
    c = mix(deep, mid, smoothstep(s / 0.55))
    edge = smoothstep((s - 0.55) / 0.45) * (0.35 + 0.65 * np.abs(t) ** 2)
    c = mix(c, light, 0.75 * edge)
    veins = 1 + 0.06 * np.sin(22 * t * (0.25 + s) + jitter * 9)           # radial veins
    ao = 0.50 + 0.50 * frac ** 0.6                                        # deep inside = darker
    c = mix(c, (0.55, 0.35, 0.30), 0.25 * np.exp(-(s / 0.06) ** 2))       # pale nick at the base
    # slightly browned, papery rim on some petals
    rim = smoothstep((s - 0.84) / 0.16) * (0.3 + 0.7 * np.abs(t) ** 2) * brown
    c = mix(c, (0.30, 0.07, 0.04), rim)
    if back:
        c = mix(c, (0.84, 0.30, 0.36), 0.32)
        c = mix(c, (0.50, 0.10, 0.12), 0.35 * np.exp(-(s / 0.25) ** 2))   # still darker at the base
        ao = 0.55 + 0.45 * ao
    return np.clip(c * veins[..., None] * ao * (0.94 + 0.12 * jitter), 0, 1)


N_PETALS = 38
_rng = np.random.default_rng(7)
_PJ = _rng.random((N_PETALS, 6))           # per-petal randomness: phase, jitter, size, tilt, asym, azimuth
_PETAL_PHASE = _PJ[:, 0] * 6.28
FS = 0.78                                  # overall flower size vs. leaves


def petal_frac(j):
    return (j / (N_PETALS - 1)) ** 1.3     # denser inner layers -> tight rolled centre


def petal_shape(frac, b, scale, jit):
    """All shape parameters of a petal at layer `frac` (0 = innermost, 1 = outermost), open amount b."""
    size = (1 + 0.16 * (jit[2] - 0.5)) * scale * FS
    return dict(
        L=lerp(0.40, 1.10, frac ** 0.8) * size,
        W=lerp(0.30, 0.60, frac ** 0.7) * size * (1 + 0.1 * (jit[1] - 0.5)),
        base_r=lerp(0.05, 0.17, frac) * scale * FS,
        base_z=lerp(0.10, 0.0, frac) * scale * FS - 0.02,
        th0=lerp(lerp(0.04, 0.24, frac), lerp(0.14, 0.80, frac ** 0.9), b) + 0.10 * (jit[3] - 0.5) * (0.3 + frac),
        bend=lerp(lerp(-0.50, -0.42, frac), lerp(0.10, 0.40, frac), b),
        tip=lerp(0.0, lerp(0.05, 0.50, frac), b),
        cup=lerp(1.7, lerp(1.2, 0.85, frac), b),
        ruffle=0.07 * b * frac,
        lip=lerp(0.45, 1.7, b),
        asym=0.30 * (jit[4] - 0.5),
    )


def flower_surfaces(growth, bloom):
    """Petals (inner-face + outer-face colours) and sepals for bud growth (0..1) and bloom (0..1)."""
    petals, sepals = [], []
    if growth <= 0.005:
        return petals, sepals
    g = smoothstep(growth)
    scale = 0.10 + 0.90 * g

    for j in range(N_PETALS):
        frac = petal_frac(j)
        delay = lerp(0.62, 0.0, frac ** 0.8)
        b = smoothstep((bloom - delay) / 0.38)
        prm = petal_shape(frac, b, scale, _PJ[j])
        P, s, t = petal(phi=j * GOLDEN + 0.25 * (_PJ[j, 5] - 0.5), phase=_PETAL_PHASE[j], **prm)
        brown = 0.55 * _PJ[j, 1] * frac ** 1.5
        petals.append((P, petal_colour(s, t, frac, _PJ[j, 0], brown),
                       petal_colour(s, t, frac, _PJ[j, 0], brown, back=True)))

    for k in range(5):                           # sepals: green, fold back as the flower opens
        phi = k * 2 * np.pi / 5 + 0.3
        b = smoothstep(bloom / 0.45)
        P, s, t = petal(0.66 * scale, 0.095 * scale, phi, 0.19 * scale, -0.05 * scale,
                        th0=lerp(0.42, 1.55, b), bend=lerp(-0.12, 0.55, b), tip=lerp(0, 0.5, b),
                        cup=0.5, profile=(0.5, 1.0, 0.85), nS=24, nT=9)
        c = mix((0.07, 0.24, 0.07), (0.28, 0.44, 0.14), np.clip(s * 1.1, 0, 1))
        c = mix(c, (0.40, 0.14, 0.08), 0.35 * smoothstep((s - 0.6) / 0.4))
        sepals.append((P, np.clip(c, 0, 1)))
    return petals, sepals


# --------------------------------------------------------------------------
#  Scene = function of time
# --------------------------------------------------------------------------
def scene(t):
    """Returns (petals_inner_face, petals_outer_face, plant_surfaces) at time t."""
    p = smoothstep(span(t, *T_STEM))
    growth = span(t, *T_BUD)
    bloom = smoothstep(span(t, *T_BLOOM))

    plant = []
    if p > 0.003:
        plant.append(stem_surface(p))
        plant += thorn_surfaces(p)
        plant += leaf_surfaces(p)
    petals, sepals = flower_surfaces(growth, bloom)
    p_in = [(P, ci) for P, ci, co in petals]
    p_out = [(P, co) for P, ci, co in petals]
    return p_in, p_out, plant + sepals


# --------------------------------------------------------------------------
#  Static extras: dew drops and the petal that falls
# --------------------------------------------------------------------------
def _sphere(c, r, nS=9, nT=13):
    th = np.linspace(0, np.pi, nS)[:, None]
    ph = np.linspace(0, 2 * np.pi, nT)[None, :]
    P = np.asarray(c) + r * np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                                      np.cos(th) * np.ones_like(ph)], -1)
    col = np.ones((nS, nT, 3)) * np.array([0.36, 0.46, 0.55])
    return P, col


def dew_drops(n_petal=7, n_leaf=7):
    """Water drops resting on the upper faces of the finished petals and leaves."""
    rng = np.random.default_rng(11)
    petals, _ = flower_surfaces(1.0, 1.0)
    leaves = leaf_surfaces(1.0)
    out = []

    def place(P, s_rng, t_rng, radius):
        nS, nT = P.shape[:2]
        i = int(rng.uniform(*s_rng) * (nS - 1))
        j = int(rng.uniform(*t_rng) * (nT - 1))
        i, j = np.clip(i, 1, nS - 2), np.clip(j, 1, nT - 2)
        n = np.cross(P[i + 1, j] - P[i - 1, j], P[i, j + 1] - P[i, j - 1])
        n = unit(n)
        if n[2] < 0:
            n = -n
        if n[2] < 0.35:                      # only on surfaces that face upwards
            return None
        return _sphere(P[i, j] + n * radius * 0.55, radius)

    for _ in range(400):
        if len(out) >= n_petal + n_leaf:
            break
        if len(out) < n_petal:
            P = petals[int(rng.integers(N_PETALS // 2, N_PETALS))][0]
            d = place(P, (0.35, 0.65), (0.30, 0.70), rng.uniform(0.016, 0.028))
        else:
            P = leaves[int(rng.integers(0, len(leaves)))][0]
            if P.shape[0] < 20:              # skip stalks
                continue
            d = place(P, (0.3, 0.7), (0.25, 0.75), rng.uniform(0.014, 0.024))
        if d is not None:
            out.append(d)
    return out


def fallen_petal():
    """A single open petal, centred on its own centroid so it can tumble."""
    frac = 0.93
    prm = petal_shape(frac, 1.0, 1.0, np.array([0.5, 0.7, 0.5, 0.5, 0.8, 0.5]))
    P, s, t = petal(phi=0.0, phase=1.3, **prm)
    P = P - P.reshape(-1, 3).mean(0)
    ci = petal_colour(s, t, frac, 0.4, 0.5)
    co = petal_colour(s, t, frac, 0.4, 0.5, back=True)
    return [(P, ci)], [(P, co)]


def fall_pose(t):
    """(position, orientation_xyz_deg, opacity) of the falling petal at time t, or None if not visible yet."""
    if t < T_FALL0:
        return None
    k = np.clip((t - T_FALL0) / FALL_DUR, 0, 1)
    z0, zg = -0.10, -H + 0.035
    e = k ** 1.15
    flutter = (1 - k) ** 0.6
    x = 0.62 + 0.95 * k + 0.32 * np.sin(2 * np.pi * 1.4 * k) * flutter
    y = 0.15 + 0.55 * k + 0.32 * np.cos(2 * np.pi * 1.1 * k + 0.6) * flutter
    z = z0 + (zg - z0) * e
    amp = 150 * (1 - k) ** 1.2
    ori = (80 + amp * np.sin(2 * np.pi * 1.3 * k + 0.5),
           25 * np.sin(2 * np.pi * 0.9 * k) * flutter,
           40 + 300 * (1 - (1 - k) ** 2) + amp * 0.6 * np.sin(2 * np.pi * 1.7 * k))
    opacity = float(np.clip((t - T_FALL0) / 0.8, 0, 1))
    return (x, y, z), ori, opacity


# --------------------------------------------------------------------------
#  Surfaces -> single PolyData  (vectorised, topology cached per grid size)
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _quads(nS, nT):
    i, j = np.meshgrid(np.arange(nS - 1), np.arange(nT - 1), indexing="ij")
    a = (i * nT + j).ravel()
    return np.stack([a, a + nT, a + nT + 1, a + 1], -1)


def assemble(surfs, flip=False):
    """List of (P, C) grids -> (points, uint8 rgb, VTK face array).  flip=True reverses winding."""
    if not surfs:
        return None, None, None
    pts, cols, faces, off = [], [], [], 0
    for P, C in surfs:
        nS, nT = P.shape[:2]
        pts.append(P.reshape(-1, 3))
        cols.append(C.reshape(-1, 3))
        q = _quads(nS, nT) + off
        if flip:
            q = q[:, [0, 3, 2, 1]]
        faces.append(np.hstack([np.full((len(q), 1), 4), q]))
        off += nS * nT
    return (np.concatenate(pts), (np.concatenate(cols) * 255).astype(np.uint8),
            np.concatenate(faces).ravel())


# --------------------------------------------------------------------------
#  Renderer (PyVista / VTK)
# --------------------------------------------------------------------------
class Stage:
    SPIN = ("plant", "petals_in", "petals_out", "drops")

    def __init__(self, pv, args):
        self.pv = pv
        self.args = args
        off = bool(args.save)
        pl = self.pl = pv.Plotter(window_size=(1280, 960), off_screen=off)
        pl.set_background("#050507", top="#1c1c26")
        pl.remove_all_lights()
        for pos, col, inten in (((4, -4, 6), (1.0, 0.93, 0.82), 1.0),     # warm key
                                ((-5, 4, 3), (0.55, 0.65, 1.0), 0.55),     # cool rim
                                ((0, 5, -2), (1.0, 0.80, 0.80), 0.25)):    # bounce from below
            pl.add_light(pv.Light(position=pos, focal_point=(0, 0, -0.3), color=col,
                                  intensity=inten, light_type="scene light"))

        q = args.quality
        self._try(lambda: pl.enable_anti_aliasing("ssaa" if q == "high" else "fxaa"))
        if q in ("medium", "high"):
            self._try(lambda: pl.enable_ssao(radius=0.35, bias=0.02, kernel_size=128, blur=True))
        if q == "high":
            self._try(lambda: pl.enable_shadows())
            self._try(lambda: pl.enable_depth_of_field())

        self.mat_petal = dict(ambient=0.28, diffuse=0.95, specular=0.12, specular_power=8)
        if args.pbr:                                 # experimental; look depends on VTK version
            self.mat_petal = dict(pbr=True, metallic=0.0, roughness=0.55)
        self.mat_plant = dict(ambient=0.25, diffuse=0.85, specular=0.25, specular_power=25)

        self.actors = {}
        self.final = False
        self._build_ground()
        fall_in, fall_out = fallen_petal()
        self.set_mesh("fall_in", fall_in, culling="back", **self.mat_petal)
        self.set_mesh("fall_out", fall_out, flip=True, culling="back", **self.mat_petal)
        self.set_mesh("drops", dew_drops(), specular=1.0, specular_power=120, ambient=0.1,
                      diffuse=0.4, opacity=0.8)

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _try(fn):
        try:
            fn()
        except Exception as exc:                      # optional effect not supported -> carry on
            print(f"[info] optional effect skipped: {exc}")

    def _mesh(self, surfs, flip=False):
        pts, rgb, faces = assemble(surfs, flip)
        if pts is None:
            return None
        m = self.pv.PolyData(pts, faces)
        m.point_data["rgb"] = rgb
        return m

    def set_mesh(self, name, surfs, flip=False, **mat):
        m = self._mesh(surfs, flip)
        if m is None:
            self.pl.remove_actor(name, render=False)
            self.actors.pop(name, None)
            return
        self.actors[name] = self.pl.add_mesh(m, scalars="rgb", rgb=True, smooth_shading=True, name=name,
                                             reset_camera=False, render=False, **mat)

    def _build_ground(self):
        pv = self.pv
        disc = pv.Disc(center=(0, 0, -H - 0.004), inner=0.02, outer=3.4, normal=(0, 0, 1), r_res=28, c_res=72)
        r = np.linalg.norm(disc.points[:, :2], axis=1) / 3.4
        f = smoothstep(r) ** 0.6
        col = np.array([0.17, 0.16, 0.18])[None] * (1 - f[:, None]) + np.array([0.02, 0.02, 0.03])[None] * f[:, None]
        disc.point_data["rgb"] = (col * 255).astype(np.uint8)
        self.actors["ground"] = self.pl.add_mesh(disc, scalars="rgb", rgb=True, specular=0.5, specular_power=40,
                                                 reset_camera=False, render=False, name="ground")

    # ---- per-frame -------------------------------------------------------
    def update(self, t):
        pl = self.pl
        if t <= T_END + 0.1 or not self.final:
            p_in, p_out, plant = scene(t)
            self.set_mesh("plant", plant, **self.mat_plant)
            self.set_mesh("petals_in", p_in, culling="back", **self.mat_petal)
            self.set_mesh("petals_out", p_out, flip=True, culling="back", **self.mat_petal)
            self.final = t > T_END

        # spin + gentle wind sway (pivoting at the base) = actor transforms only, essentially free
        grow = smoothstep(span(t, 1.0, 5.5))
        tx = grow * (1.1 * np.sin(2 * np.pi * t / 6.5) + 0.5 * np.sin(2 * np.pi * t / 2.9 + 1.0))
        ty = grow * (0.9 * np.sin(2 * np.pi * t / 5.3 + 2.0) + 0.4 * np.sin(2 * np.pi * t / 3.3))
        yaw = t * SPIN_DEG_PER_SEC
        for n in self.SPIN:
            a = self.actors.get(n)
            if a is not None:
                a.SetOrigin(0, 0, -H)
                a.SetOrientation(float(tx), float(ty), float(yaw))
        a = self.actors.get("drops")
        if a is not None:
            vis = float(smoothstep(span(t, *T_DROPS)))
            a.SetVisibility(bool(vis > 0.02))           # VTK wants a plain Python bool / float,
            a.GetProperty().SetOpacity(float(0.8 * vis))  # not NumPy scalars

        pose = fall_pose(t)
        for n in ("fall_in", "fall_out"):
            a = self.actors.get(n)
            if a is None:
                continue
            if pose is None:
                a.SetVisibility(False)
                continue
            pos, ori, op = pose
            a.SetVisibility(True)
            a.SetPosition(*[float(v) for v in pos])
            a.SetOrientation(*[float(v) for v in ori])
            a.GetProperty().SetOpacity(float(op))

        self.camera(t)

    def camera(self, t):
        pl = self.pl
        k = smoothstep(span(t, 2.0, T_END))
        target = np.array([0.0, 0.0, lerp(-1.1, -0.25, k)])
        dist = lerp(10.5, 6.2, k)
        el = np.deg2rad(lerp(10, 24, k))
        pos = target + dist * np.array([np.cos(el), 0.0, np.sin(el)])
        pl.camera_position = [tuple(pos), tuple(target), (0, 0, 1)]
        pl.camera.view_angle = 30
        pl.renderer.ResetCameraClippingRange()


def run(args):
    import pyvista as pv

    st = Stage(pv, args)
    pl = st.pl

    # ---------------- offline render -------------------------------------
    if args.save:
        fps = 30
        if args.save.lower().endswith(".gif"):
            pl.open_gif(args.save, fps=fps)
        else:
            pl.open_movie(args.save, framerate=fps, quality=8)
        pl.show(auto_close=False)
        total = T_END + args.hold
        for i in range(int(total * fps)):
            st.update(i / fps)
            pl.write_frame()
        pl.close()
        print("saved", args.save)
        return

    # ---------------- live window ----------------------------------------
    clock = {"t": 0.0, "paused": False, "loop": args.loop}

    def toggle_pause():
        clock["paused"] = not clock["paused"]

    def restart():
        clock["t"] = 0.0
        st.final = False

    def toggle_loop():
        clock["loop"] = not clock["loop"]

    pl.add_key_event("space", toggle_pause)
    pl.add_key_event("n", restart)
    pl.add_key_event("l", toggle_loop)

    st.update(0.0)
    pl.show(auto_close=False, interactive_update=True)
    last = time.perf_counter()
    while not getattr(pl, "_closed", False):
        now = time.perf_counter()
        dt, last = now - last, now
        if not clock["paused"]:
            clock["t"] += dt * args.speed
        if clock["loop"] and clock["t"] > T_LOOP:
            restart()
        try:
            st.update(clock["t"])
            pl.update()
        except Exception:
            break                                   # window was closed
        time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - now)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Growing & blooming rose")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--quality", choices=["low", "medium", "high"], default="medium",
                    help="low: fastest | medium: + ambient occlusion | high: + shadows, depth of field, supersampling")
    ap.add_argument("--loop", action="store_true", help="restart automatically when the animation ends")
    ap.add_argument("--pbr", action="store_true", help="experimental physically-based petal shading")
    ap.add_argument("--save", default=None, help="write .mp4 or .gif instead of opening a window")
    ap.add_argument("--hold", type=float, default=9.0,
                    help="seconds to keep recording after the bloom (with --save); 9 covers the falling petal")
    run(ap.parse_args())