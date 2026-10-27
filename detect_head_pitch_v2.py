#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_head_pitch.py  (version 2)
=================================
Analyse la vidéo de la caméra scène d'un eye-tracker (lunettes) et, si on le
fournit, le fichier de regard, pour repérer les situations où l'eye-tracker
risque de perdre le regard.

Événements détectés (une image peut porter plusieurs étiquettes à la fois)
---------------------------------------------------------------------------
MESURE_GAZE_MANQUANTE        coordonnées absentes, invalides (NaN, inf, hors
                             image, confiance trop basse) ou égales à (0, 0),
                             ou absence totale d'échantillon.
PERTE_MESURE_PROLONGEE       sous-ensemble des précédentes : absence de mesures
                             pendant au moins --prolonged-min secondes.
VISAGE_ABSENT                aucun visage détecté alors que la caméra filme
                             effectivement un interlocuteur (voir --face-mode).
TETE_BAISSEE / REGARD_HAUT   mouvement vertical de la caméra (angle de tangage
                             estimé par flux optique) compatible avec une tête
                             inclinée vers le bas / orientée vers le haut.
MOUVEMENT_HORIZONTAL_GAUCHE  mouvement horizontal de la caméra (angle de lacet
MOUVEMENT_HORIZONTAL_DROITE  estimé par flux optique) compatible avec une
                             rotation de la tête vers la gauche / la droite.

Pour chaque segment de perte de mesure, une colonne « causes_probables »
indique quels autres événements le recouvrent (avec une marge temporelle).

Usage
-----
    python detect_head_pitch.py video.mp4 --vfov 60
    python detect_head_pitch.py video.mp4 --gaze gaze.csv --time-col timestamp \
        --x-col gaze_x --y-col gaze_y --time-unit ms --annotate

Dépendances : pip install opencv-python numpy pandas matplotlib
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
EVENT_ORDER = [
    "MESURE_GAZE_MANQUANTE",
    "PERTE_MESURE_PROLONGEE",
    "VISAGE_ABSENT",
    "TETE_BAISSEE",
    "REGARD_HAUT",
    "MOUVEMENT_HORIZONTAL_GAUCHE",
    "MOUVEMENT_HORIZONTAL_DROITE",
]
CAUSE_EVENTS = [
    "TETE_BAISSEE",
    "REGARD_HAUT",
    "MOUVEMENT_HORIZONTAL_GAUCHE",
    "MOUVEMENT_HORIZONTAL_DROITE",
    "VISAGE_ABSENT",
]
COLORS_RGB = {
    "MESURE_GAZE_MANQUANTE": (200, 30, 30),
    "PERTE_MESURE_PROLONGEE": (120, 0, 0),
    "VISAGE_ABSENT": (150, 60, 200),
    "TETE_BAISSEE": (30, 110, 220),
    "REGARD_HAUT": (255, 140, 0),
    "MOUVEMENT_HORIZONTAL_GAUCHE": (0, 160, 130),
    "MOUVEMENT_HORIZONTAL_DROITE": (120, 190, 40),
}
UNIT_FACTOR = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


# --------------------------------------------------------------------------- #
# Outils sur des masques booléens (un booléen par image)
# --------------------------------------------------------------------------- #
def runs_of(mask):
    """Plages consécutives de True : liste de (début, fin) inclus."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def clean_mask(mask, fps, min_dur, merge_gap):
    """Fusionne les plages séparées de moins de merge_gap s,
    puis supprime celles de moins de min_dur s."""
    m = np.asarray(mask, dtype=bool).copy()
    gap_n = int(round(merge_gap * fps))
    if gap_n > 0:
        runs = runs_of(m)
        for (_, e0), (s1, _) in zip(runs, runs[1:]):
            if s1 - e0 - 1 <= gap_n:
                m[e0 + 1:s1] = True
    min_n = max(1, int(round(min_dur * fps)))
    for s, e in runs_of(m):
        if e - s + 1 < min_n:
            m[s:e + 1] = False
    return m


def dilate(mask, n):
    """Élargit chaque plage de True de n images de chaque côté."""
    m = np.asarray(mask, dtype=bool)
    if n <= 0 or not m.any():
        return m.copy()
    c = np.r_[0, np.cumsum(m)]
    idx = np.arange(len(m))
    lo = np.clip(idx - n, 0, len(m))
    hi = np.clip(idx + n + 1, 0, len(m))
    return (c[hi] - c[lo]) > 0


# --------------------------------------------------------------------------- #
# Détection de visages
# --------------------------------------------------------------------------- #
class FaceDetector:
    """Haar (intégré à OpenCV) par défaut ; YuNet si un modèle .onnx est fourni
    (plus fiable : https://github.com/opencv/opencv_zoo, dossier face_detection_yunet)."""

    def __init__(self, yunet_model=None, width=480):
        self.width = width
        self.yunet = None
        if yunet_model:
            self.yunet = cv2.FaceDetectorYN.create(
                str(yunet_model), "", (width, 10), 0.6, 0.3, 5000)
        else:
            if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
                raise RuntimeError("cette version d'OpenCV n'a pas CascadeClassifier "
                                   f"(version {cv2.__version__})")
            base = cv2.data.haarcascades
            self.front = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
            self.prof = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
            if self.front.empty():
                raise RuntimeError("cascade Haar introuvable dans l'installation d'OpenCV")

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        img = cv2.resize(frame_bgr, (self.width, int(round(h * self.width / w))),
                         interpolation=cv2.INTER_AREA)
        if self.yunet is not None:
            self.yunet.setInputSize((img.shape[1], img.shape[0]))
            _, faces = self.yunet.detect(img)
            return faces is not None and len(faces) > 0
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        ms = (max(24, img.shape[1] // 20),) * 2
        if len(self.front.detectMultiScale(gray, 1.1, 5, minSize=ms)):
            return True
        if not self.prof.empty():
            if len(self.prof.detectMultiScale(gray, 1.1, 5, minSize=ms)):
                return True
            if len(self.prof.detectMultiScale(cv2.flip(gray, 1), 1.1, 5, minSize=ms)):
                return True
        return False


# --------------------------------------------------------------------------- #
# 1. Analyse vidéo : mouvement de la caméra + visages (une seule lecture)
# --------------------------------------------------------------------------- #
def analyze_video(video_path, detector, face_step=3, motion_width=320, min_inliers=15):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir la vidéo : {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    dys, dxs, inls, texs, faces = [], [], [], [], []
    prev = None
    W = H = sw = sh = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if W is None:
            H, W = frame.shape[:2]
            sw = motion_width
            sh = int(round(H * sw / W))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        texs.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        dx = dy = 0.0
        n_inl = 0
        if prev is not None:
            pts = cv2.goodFeaturesToTrack(prev, maxCorners=250, qualityLevel=0.01,
                                          minDistance=8)
            if pts is not None and len(pts) >= 10:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None,
                                                      winSize=(21, 21), maxLevel=3)
                good = st.ravel() == 1
                if good.sum() >= 10:
                    p0, p1 = pts[good], nxt[good]
                    M, inl = cv2.estimateAffinePartial2D(
                        p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if M is not None and inl is not None:
                        inl = inl.ravel().astype(bool)
                        n_inl = int(inl.sum())
                        if n_inl >= min_inliers:
                            disp = (p1 - p0).reshape(-1, 2)[inl]
                            dx = float(np.median(disp[:, 0]))
                            dy = float(np.median(disp[:, 1]))
        dxs.append(dx)
        dys.append(dy)
        inls.append(n_inl)
        prev = gray

        if detector is not None and i % face_step == 0:
            faces.append(bool(detector.detect(frame)))

        i += 1
        if i % 300 == 0:
            print(f"  {i}/{n_total}" if n_total else f"  {i} images", end="\r")
    cap.release()
    print(f"  {i} images analysées.            ")
    if i == 0:
        raise SystemExit("Aucune image lue dans la vidéo.")

    face_raw = None
    if detector is not None:
        face_raw = np.repeat(np.array(faces, dtype=bool), face_step)[:i]
    return dict(fps=fps, W=W, H=H, sw=sw, sh=sh,
                dx=np.array(dxs), dy=np.array(dys),
                inliers=np.array(inls), texture=np.array(texs),
                face_raw=face_raw, min_inliers=min_inliers)


# --------------------------------------------------------------------------- #
# 2. Angles de la caméra
# --------------------------------------------------------------------------- #
def compute_angle(delta_px, deg_per_px, sign, fps, baseline, window_s):
    """Intègre le déplacement (px) en angle (°) puis retire la posture de référence."""
    raw = sign * np.cumsum(delta_px) * deg_per_px
    s = pd.Series(raw)
    if baseline == "global":
        base = np.full(len(s), s.median())
    elif baseline == "rolling":
        win = max(int(window_s * fps), 3)
        base = s.rolling(win, center=True, min_periods=1).median().values
    else:
        base = np.zeros(len(s))
    return raw, raw - base


# --------------------------------------------------------------------------- #
# 3. Données de regard
# --------------------------------------------------------------------------- #
def load_gaze(args, W, H):
    df = pd.read_csv(args.gaze, sep=None, engine="python")
    for c in (args.time_col, args.x_col, args.y_col):
        if c not in df.columns:
            raise SystemExit(f"Colonne '{c}' introuvable dans {args.gaze}.\n"
                             f"Colonnes disponibles : {list(df.columns)}")
    t = pd.to_numeric(df[args.time_col], errors="coerce")
    x = pd.to_numeric(df[args.x_col], errors="coerce")
    y = pd.to_numeric(df[args.y_col], errors="coerce")
    conf = None
    if args.conf_col:
        if args.conf_col not in df.columns:
            raise SystemExit(f"Colonne de confiance '{args.conf_col}' introuvable.")
        conf = pd.to_numeric(df[args.conf_col], errors="coerce")

    keep = t.notna()
    t, x, y = t[keep], x[keep], y[keep]
    conf = conf[keep] if conf is not None else None
    order = np.argsort(t.values, kind="stable")
    t, x, y = t.values[order], x.values[order], y.values[order]
    conf = conf.values[order] if conf is not None else None
    if len(t) == 0:
        raise SystemExit("Le fichier de regard ne contient aucun horodatage valide.")

    t0_raw = t[0] if args.gaze_t0 is None else args.gaze_t0
    t_s = (t - t0_raw) * UNIT_FACTOR[args.time_unit]

    # Validité d'un échantillon
    valid = np.isfinite(x) & np.isfinite(y)                  # absent / invalide
    valid &= ~((x == 0) & (y == 0))                          # (0, 0)
    if conf is not None and args.min_conf is not None:
        valid &= np.nan_to_num(conf, nan=-1.0) >= args.min_conf
    mode = args.gaze_coords
    if mode == "auto":
        finite = np.abs(np.r_[x[np.isfinite(x)], y[np.isfinite(y)]])
        mode = "normalized" if len(finite) and np.percentile(finite, 99) <= 1.5 else "pixels"
    if mode == "normalized":
        valid &= (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    elif mode == "pixels":
        valid &= (x >= 0) & (x <= W) & (y >= 0) & (y <= H)
    print(f"   Regard : {len(t_s)} échantillons, coordonnées interprétées en '{mode}', "
          f"{100 * (~valid).mean():.1f} % invalides.")
    return t_s, valid


def gaze_missing_per_frame(t_s, valid, n, fps, valid_ratio, gap_tol):
    """True pour chaque image où la mesure du regard est manquante."""
    fi = np.floor(t_s * fps).astype(int)
    inside = (fi >= 0) & (fi < n)
    tot = np.bincount(fi[inside], minlength=n)
    nok = np.bincount(fi[inside], weights=valid[inside].astype(float), minlength=n)

    missing = np.ones(n, dtype=bool)               # par défaut : aucune donnée
    has = tot > 0
    missing[has] = (nok[has] / tot[has]) < valid_ratio

    empty = np.where(~has)[0]                      # images sans échantillon
    if len(empty) and len(t_s):
        tc = (empty + 0.5) / fps
        idx = np.searchsorted(t_s, tc)
        i0 = np.clip(idx - 1, 0, len(t_s) - 1)
        i1 = np.clip(idx, 0, len(t_s) - 1)
        pick = np.where(np.abs(t_s[i0] - tc) <= np.abs(t_s[i1] - tc), i0, i1)
        near = np.abs(t_s[pick] - tc) <= gap_tol
        missing[empty[near]] = ~valid[pick[near]]
    return missing


# --------------------------------------------------------------------------- #
# 4. Événements
# --------------------------------------------------------------------------- #
def build_events(mask, etype, fps, angle_rel=None, direction=0):
    events = []
    for s, e in runs_of(mask):
        peak = np.nan
        if angle_rel is not None:
            seg = angle_rel[s:e + 1]
            peak = round(float(seg.min() if direction < 0 else seg.max()), 1)
        events.append({
            "type": etype,
            "start_frame": s, "end_frame": e,
            "start_s": round(s / fps, 3), "end_s": round((e + 1) / fps, 3),
            "duration_s": round((e - s + 1) / fps, 3),
            "peak_angle_deg": peak,
        })
    return events


# --------------------------------------------------------------------------- #
# 5. Sorties : graphique et vidéo annotée
# --------------------------------------------------------------------------- #
def plot_result(t, fps, pitch_rel, yaw_rel, masks, args, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = {k: tuple(c / 255 for c in v) for k, v in COLORS_RGB.items()}
    fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                            gridspec_kw={"height_ratios": [2, 2, 3]})
    axs[0].plot(t, pitch_rel, lw=0.8, color="k")
    axs[0].axhline(args.thr_up, color=rgb["REGARD_HAUT"], ls="--", lw=0.9)
    axs[0].axhline(-args.thr_down, color=rgb["TETE_BAISSEE"], ls="--", lw=0.9)
    axs[0].set_ylabel("Vertical (°)\nhaut + / bas −")

    axs[1].plot(t, yaw_rel, lw=0.8, color="k")
    axs[1].axhline(args.thr_right, color=rgb["MOUVEMENT_HORIZONTAL_DROITE"], ls="--", lw=0.9)
    axs[1].axhline(-args.thr_left, color=rgb["MOUVEMENT_HORIZONTAL_GAUCHE"], ls="--", lw=0.9)
    axs[1].set_ylabel("Horizontal (°)\ndroite + / gauche −")

    for k, name in enumerate(EVENT_ORDER):
        bars = [(s / fps, (e - s + 1) / fps) for s, e in runs_of(masks[name])]
        if bars:
            axs[2].broken_barh(bars, (k - 0.4, 0.8), color=rgb[name])
    axs[2].set_yticks(range(len(EVENT_ORDER)))
    axs[2].set_yticklabels(EVENT_ORDER, fontsize=8)
    axs[2].invert_yaxis()
    axs[2].set_xlabel("Temps (s)")
    fig.suptitle("Événements susceptibles de faire perdre le regard à l'eye-tracker")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def write_annotated_video(video_path, out_path, masks, pitch_rel, yaw_rel, fps):
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = len(pitch_rel)
    scale = max(0.5, w / 1280)
    i = 0
    while i < n:
        ok, frame = cap.read()
        if not ok:
            break
        active = [k for k in EVENT_ORDER if masks[k][i]]
        if active:
            bgr = COLORS_RGB[active[0]][::-1]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), bgr, int(10 * scale))
        for j, name in enumerate(active):
            cv2.putText(frame, name, (int(20 * scale), int((45 + 38 * j) * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale,
                        COLORS_RGB[name][::-1], max(2, int(2 * scale)))
        cv2.putText(frame, f"t={i / fps:6.2f}s  V:{pitch_rel[i]:+5.1f}  H:{yaw_rel[i]:+5.1f} deg",
                    (int(20 * scale), h - int(20 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * scale, (255, 255, 255), max(2, int(2 * scale)))
        vw.write(frame)
        i += 1
    cap.release()
    vw.release()


# --------------------------------------------------------------------------- #
# Programme principal
# --------------------------------------------------------------------------- #
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="Vidéo de la caméra scène de l'eye-tracker")
    ap.add_argument("--out-dir", default=None, help="Dossier de sortie (défaut : à côté de la vidéo)")
    ap.add_argument("--annotate", action="store_true", help="Écrit une vidéo annotée")

    g = ap.add_argument_group("Mouvements de la caméra")
    g.add_argument("--vfov", type=float, default=60.0, help="Champ de vision vertical (°), défaut 60")
    g.add_argument("--hfov", type=float, default=None,
                   help="Champ de vision horizontal (°). Défaut : déduit de --vfov et du format d'image")
    g.add_argument("--thr-down", type=float, default=20.0, help="Seuil TETE_BAISSEE (°)")
    g.add_argument("--thr-up", type=float, default=20.0, help="Seuil REGARD_HAUT (°)")
    g.add_argument("--thr-left", type=float, default=30.0, help="Seuil MOUVEMENT_HORIZONTAL_GAUCHE (°)")
    g.add_argument("--thr-right", type=float, default=30.0, help="Seuil MOUVEMENT_HORIZONTAL_DROITE (°)")
    g.add_argument("--min-dur", type=float, default=0.4, help="Durée min. d'un mouvement (s)")
    g.add_argument("--merge-gap", type=float, default=0.4, help="Fusionne les mouvements séparés de moins de X s")
    g.add_argument("--baseline", choices=["rolling", "global", "none"], default="rolling",
                   help="Référence de la posture 'normale' (défaut : médiane glissante)")
    g.add_argument("--window", type=float, default=45.0, help="Fenêtre de la médiane glissante (s)")

    g = ap.add_argument_group("Visage")
    g.add_argument("--face-mode", choices=["auto", "on", "off"], default="auto",
                   help="auto : VISAGE_ABSENT n'est évalué que si des visages sont vus sur au moins "
                        "--min-face-rate de la vidéo ; on : toujours ; off : désactivé (plus rapide)")
    g.add_argument("--min-face-rate", type=float, default=0.10,
                   help="Part min. d'images avec visage pour considérer qu'un interlocuteur est filmé (défaut 0.10)")
    g.add_argument("--face-step", type=int, default=3, help="Détection de visage toutes les N images")
    g.add_argument("--face-width", type=int, default=480, help="Largeur de travail de la détection (px)")
    g.add_argument("--face-hold", type=float, default=0.3,
                   help="Un visage détecté reste 'présent' X s (évite le scintillement), défaut 0.3")
    g.add_argument("--min-dur-face", type=float, default=0.5, help="Durée min. de VISAGE_ABSENT (s)")
    g.add_argument("--yunet-model", default=None, help="Modèle YuNet .onnx (optionnel, plus fiable que Haar)")

    g = ap.add_argument_group("Fichier de regard")
    g.add_argument("--gaze", default=None, help="CSV de regard (optionnel mais nécessaire pour les 2 événements de mesure)")
    g.add_argument("--time-col", default="timestamp")
    g.add_argument("--x-col", default="x")
    g.add_argument("--y-col", default="y")
    g.add_argument("--conf-col", default=None, help="Colonne de confiance (optionnel)")
    g.add_argument("--min-conf", type=float, default=None, help="Confiance minimale (avec --conf-col)")
    g.add_argument("--time-unit", choices=list(UNIT_FACTOR), default="s")
    g.add_argument("--gaze-t0", type=float, default=None,
                   help="Instant du CSV (dans l'unité --time-unit) correspondant au début de la vidéo "
                        "(défaut : premier échantillon)")
    g.add_argument("--gaze-coords", choices=["auto", "normalized", "pixels", "none"], default="auto",
                   help="Repère des coordonnées pour le test 'hors image' (none : pas de test)")
    g.add_argument("--valid-ratio", type=float, default=0.5,
                   help="Une image est 'manquante' si moins de cette part de ses échantillons est valide")
    g.add_argument("--gap-tol", type=float, default=0.1,
                   help="Sans échantillon dans l'image, on accepte l'échantillon le plus proche à ≤ X s")
    g.add_argument("--min-dur-missing", type=float, default=0.1, help="Durée min. de MESURE_GAZE_MANQUANTE (s)")
    g.add_argument("--merge-gap-missing", type=float, default=0.1,
                   help="Fusionne les manques séparés de moins de X s")
    g.add_argument("--prolonged-min", type=float, default=1.0,
                   help="Durée min. de PERTE_MESURE_PROLONGEE (s), défaut 1.0")

    g = ap.add_argument_group("Attribution des causes")
    g.add_argument("--cause-margin", type=float, default=0.3,
                   help="Marge (s) autour des événements-causes pour le recouvrement")
    g.add_argument("--cause-overlap", type=float, default=0.5,
                   help="Part min. d'un segment de perte recouverte par une cause pour la retenir")
    return ap


def main():
    args = build_parser().parse_args()
    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"Vidéo introuvable : {video}")
    out_dir = Path(args.out_dir) if args.out_dir else video.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem

    # ---------------- 1. Vidéo ----------------
    print(f"[1/5] Analyse de la vidéo : {video.name}")
    detector = None
    if args.face_mode != "off":
        try:
            detector = FaceDetector(args.yunet_model, args.face_width)
        except Exception as exc:  # OpenCV 5 : plus de CascadeClassifier, etc.
            print(f"   Détection de visage indisponible : {exc}.\n"
                  f"   VISAGE_ABSENT sera ignoré (solutions : pip install \"opencv-python<5\", "
                  f"ou fournir --yunet-model).")
    v = analyze_video(video, detector, args.face_step)
    fps, W, H = v["fps"], v["W"], v["H"]
    n = len(v["dy"])
    t = np.arange(n) / fps
    tracking_lost = v["inliers"] < v["min_inliers"]
    tracking_lost[0] = False

    # ---------------- 2. Angles ----------------
    print("[2/5] Estimation des angles (vertical / horizontal)")
    hfov = args.hfov or math.degrees(
        2 * math.atan(math.tan(math.radians(args.vfov / 2)) * W / H))
    # scène qui descend dans l'image (dy>0) = caméra qui se lève ; 
    # scène qui va vers la gauche (dx<0) = caméra qui tourne à droite
    pitch_raw, pitch_rel = compute_angle(v["dy"], args.vfov / v["sh"], +1, fps,
                                         args.baseline, args.window)
    yaw_raw, yaw_rel = compute_angle(v["dx"], hfov / v["sw"], -1, fps,
                                     args.baseline, args.window)

    zeros = np.zeros(n, dtype=bool)
    masks = {k: zeros.copy() for k in EVENT_ORDER}
    available = {k: True for k in EVENT_ORDER}

    masks["TETE_BAISSEE"] = clean_mask(pitch_rel <= -abs(args.thr_down), fps, args.min_dur, args.merge_gap)
    masks["REGARD_HAUT"] = clean_mask(pitch_rel >= abs(args.thr_up), fps, args.min_dur, args.merge_gap)
    masks["MOUVEMENT_HORIZONTAL_GAUCHE"] = clean_mask(yaw_rel <= -abs(args.thr_left), fps, args.min_dur, args.merge_gap)
    masks["MOUVEMENT_HORIZONTAL_DROITE"] = clean_mask(yaw_rel >= abs(args.thr_right), fps, args.min_dur, args.merge_gap)

    # ---------------- 3. Visage ----------------
    print("[3/5] Présence de visage")
    face_present = None
    if v["face_raw"] is None:
        available["VISAGE_ABSENT"] = False
        print("   Détection de visage désactivée ou indisponible : VISAGE_ABSENT non évalué.")
    else:
        face_rate = float(v["face_raw"].mean())
        face_present = dilate(v["face_raw"], int(round(args.face_hold * fps)))
        if args.face_mode == "auto" and face_rate < args.min_face_rate:
            available["VISAGE_ABSENT"] = False
            print(f"   Visage détecté sur {100 * face_rate:.1f} % des images (< "
                  f"{100 * args.min_face_rate:.0f} %) : la caméra ne semble pas filmer d'interlocuteur,\n"
                  f"   VISAGE_ABSENT n'est pas évalué (forcer avec --face-mode on).")
        else:
            masks["VISAGE_ABSENT"] = clean_mask(~face_present, fps, args.min_dur_face, 0.3)
            print(f"   Visage détecté sur {100 * face_rate:.1f} % des images.")

    # ---------------- 4. Regard ----------------
    print("[4/5] Mesures de regard")
    gaze_missing = None
    if args.gaze:
        t_s, valid = load_gaze(args, W, H)
        gaze_missing = gaze_missing_per_frame(t_s, valid, n, fps, args.valid_ratio, args.gap_tol)
        masks["MESURE_GAZE_MANQUANTE"] = clean_mask(gaze_missing, fps, args.min_dur_missing,
                                                    args.merge_gap_missing)
        masks["PERTE_MESURE_PROLONGEE"] = clean_mask(masks["MESURE_GAZE_MANQUANTE"], fps,
                                                     args.prolonged_min, 0.0)
    else:
        available["MESURE_GAZE_MANQUANTE"] = available["PERTE_MESURE_PROLONGEE"] = False
        print("   Pas de fichier de regard (--gaze) : MESURE_GAZE_MANQUANTE et "
              "PERTE_MESURE_PROLONGEE ne sont pas évalués.")

    # ---------------- 5. Segments, causes, résumé ----------------
    print("[5/5] Segments et résumé")
    events = []
    events += build_events(masks["TETE_BAISSEE"], "TETE_BAISSEE", fps, pitch_rel, -1)
    events += build_events(masks["REGARD_HAUT"], "REGARD_HAUT", fps, pitch_rel, +1)
    events += build_events(masks["MOUVEMENT_HORIZONTAL_GAUCHE"], "MOUVEMENT_HORIZONTAL_GAUCHE", fps, yaw_rel, -1)
    events += build_events(masks["MOUVEMENT_HORIZONTAL_DROITE"], "MOUVEMENT_HORIZONTAL_DROITE", fps, yaw_rel, +1)
    for name in ("VISAGE_ABSENT", "MESURE_GAZE_MANQUANTE", "PERTE_MESURE_PROLONGEE"):
        events += build_events(masks[name], name, fps)

    margin_n = int(round(args.cause_margin * fps))
    cause_masks = {c: dilate(masks[c], margin_n) for c in CAUSE_EVENTS}
    for ev in events:
        s, e = ev["start_frame"], ev["end_frame"]
        ev["tracking_lost_pct"] = round(100 * float(tracking_lost[s:e + 1].mean()), 1)
        ev["gaze_missing_pct"] = (round(100 * float(gaze_missing[s:e + 1].mean()), 1)
                                  if gaze_missing is not None else np.nan)
        ev["causes_probables"] = ""
        if ev["type"] in ("MESURE_GAZE_MANQUANTE", "PERTE_MESURE_PROLONGEE"):
            parts = []
            for c in CAUSE_EVENTS:
                if not available[c]:
                    continue
                frac = float(cause_masks[c][s:e + 1].mean())
                if frac >= args.cause_overlap:
                    parts.append(f"{c}({100 * frac:.0f}%)")
            ev["causes_probables"] = " + ".join(parts) if parts else "NON_EXPLIQUE"

    rank = {k: i for i, k in enumerate(EVENT_ORDER)}
    events.sort(key=lambda d: (d["start_frame"], rank[d["type"]]))
    cols = ["type", "start_frame", "end_frame", "start_s", "end_s", "duration_s",
            "peak_angle_deg", "gaze_missing_pct", "tracking_lost_pct", "causes_probables"]
    seg_df = pd.DataFrame(events, columns=cols)

    summary = {
        "video": video.name, "fps": fps, "duree_s": round(n / fps, 2), "n_images": n,
        "resolution": [W, H], "vfov_deg": args.vfov, "hfov_deg": round(hfov, 1),
        "images_suivi_perdu_pct": round(100 * float(tracking_lost.mean()), 1),
        "evenements": {},
    }
    for name in EVENT_ORDER:
        summary["evenements"][name] = {
            "evalue": available[name],
            "n_segments": int(sum(1 for e in events if e["type"] == name)),
            "duree_totale_s": round(float(masks[name].sum() / fps), 2),
            "pct_video": round(100 * float(masks[name].mean()), 1),
        }
    miss = masks["MESURE_GAZE_MANQUANTE"]
    if available["MESURE_GAZE_MANQUANTE"] and miss.any():
        usable = [c for c in CAUSE_EVENTS if available[c]]
        expl = {c: round(100 * float((miss & cause_masks[c]).sum() / miss.sum()), 1) for c in usable}
        any_c = np.any([cause_masks[c] for c in usable], axis=0) if usable else zeros
        expl["AU_MOINS_UNE_CAUSE"] = round(100 * float((miss & any_c).sum() / miss.sum()), 1)
        expl["NON_EXPLIQUE"] = round(100 * float((miss & ~any_c).sum() / miss.sum()), 1)
        summary["part_des_mesures_manquantes_expliquee_par_pct"] = expl

    # Fichiers
    frames_df = pd.DataFrame({
        "frame": np.arange(n), "time_s": t,
        "pitch_raw_deg": pitch_raw, "pitch_rel_deg": pitch_rel,
        "yaw_raw_deg": yaw_raw, "yaw_rel_deg": yaw_rel,
        "tracking_inliers": v["inliers"], "texture": v["texture"],
        "face_detected": (face_present.astype(int) if face_present is not None else np.nan),
        "gaze_missing_raw": (gaze_missing.astype(int) if gaze_missing is not None else np.nan),
    })
    for name in EVENT_ORDER:
        frames_df[name] = masks[name].astype(int)
    frames_df["labels"] = ["|".join(k for k in EVENT_ORDER if masks[k][i]) for i in range(n)]

    f_frames = out_dir / f"{stem}_frames.csv"
    f_segs = out_dir / f"{stem}_segments.csv"
    f_sum = out_dir / f"{stem}_summary.json"
    f_png = out_dir / f"{stem}_events.png"
    frames_df.to_csv(f_frames, index=False)
    seg_df.to_csv(f_segs, index=False)
    f_sum.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_result(t, fps, pitch_rel, yaw_rel, masks, args, f_png)

    # Affichage
    print("\n=== Résumé ===")
    for name in EVENT_ORDER:
        d = summary["evenements"][name]
        if not d["evalue"]:
            print(f"  {name:30s} non évalué")
        else:
            print(f"  {name:30s} {d['n_segments']:4d} segments  {d['duree_totale_s']:8.1f} s  "
                  f"({d['pct_video']:5.1f} % de la vidéo)")
    if "part_des_mesures_manquantes_expliquee_par_pct" in summary:
        print("\n  Part des mesures manquantes recouverte par :")
        for k, val in summary["part_des_mesures_manquantes_expliquee_par_pct"].items():
            print(f"    {k:30s} {val:5.1f} %")
    if summary["images_suivi_perdu_pct"] > 20:
        print(f"\n  Attention : suivi du mouvement perdu sur {summary['images_suivi_perdu_pct']} % des "
              f"images (scène peu texturée / floue) : les angles sont peu fiables.")

    if args.annotate:
        ann = out_dir / f"{stem}_annotated.mp4"
        print(f"\nÉcriture de la vidéo annotée : {ann.name}")
        write_annotated_video(video, ann, masks, pitch_rel, yaw_rel, fps)

    print(f"\nFichiers : {f_segs.name}, {f_frames.name}, {f_sum.name}, {f_png.name}")


if __name__ == "__main__":
    main()
