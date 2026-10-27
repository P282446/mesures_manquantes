#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_absences.py
==================
Explique les mesures de regard manquantes d'un eye-tracker à partir de la vidéo
(caméra FIXE filmant le haut du corps du sujet).

Principe
--------
1. Dans le fichier de l'eye-tracker, une mesure est MANQUANTE quand les
   coordonnées sont absentes (vide / NaN) ou égales à (0, 0).
2. Pour chaque instant, on regarde dans la vidéo ce que fait la personne :
   la position et la taille de son visage sont comparées à une POSTURE DE
   RÉFÉRENCE (la posture médiane pendant les instants où l'eye-tracker
   mesurait correctement, donc « bien assise, tête droite, face à l'appareil »).
3. Si la posture est normale pendant l'absence de mesure, l'absence est
   attribuée à une erreur de l'eye-tracker. Sinon, on nomme le comportement
   probablement responsable.
4. Une colonne « type_absence » est ajoutée au fichier : "NA" quand la mesure
   est présente, sinon le type d'absence.

Valeurs de type_absence
-----------------------
NA                     mesure présente
ERREUR_EYE_TRACKER     posture normale dans la vidéo : défaut de l'appareil
TETE_BAISSEE           visage nettement plus bas que la référence
TETE_LEVEE             visage nettement plus haut que la référence
TETE_TOURNEE           visage de profil détecté
DEPLACEMENT_LATERAL    visage décalé à gauche / à droite
RAPPROCHEMENT_CAMERA   visage plus grand (buste penché vers l'avant)
ELOIGNEMENT_CAMERA     visage plus petit (recul)
VISAGE_ABSENT          aucun visage visible (sorti du cadre, main devant...)
HORS_VIDEO             instant situé en dehors de la durée de la vidéo

Exécution simple
----------------
    python3 detect_absences.py video.wmv fichier_eyetracker.txt

Les noms de colonnes par défaut sont ceux de votre fichier (Capture Time,
Lft X Pos, Lft Y Pos, Rt X Pos, Rt Y Pos) et le temps est en microsecondes.

Dépendances : pip install "opencv-python<5" numpy pandas matplotlib
(OpenCV 5 ne fournit plus le détecteur de visage utilisé ici.)
"""
import argparse
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
NA_LABEL = "NA"                       # valeur écrite quand la mesure est présente
NORMAL = "NORMAL"                     # posture normale (interne)
T_EYE_ERROR = "ERREUR_EYE_TRACKER"
T_OUT = "HORS_VIDEO"
T_ABSENT = "VISAGE_ABSENT"
T_TURNED = "TETE_TOURNEE"

ALL_TYPES = [T_EYE_ERROR, "TETE_BAISSEE", "TETE_LEVEE", T_TURNED,
             "DEPLACEMENT_LATERAL", "RAPPROCHEMENT_CAMERA", "ELOIGNEMENT_CAMERA",
             T_ABSENT, T_OUT]

COLORS_RGB = {
    T_EYE_ERROR: (200, 30, 30), "TETE_BAISSEE": (30, 110, 220),
    "TETE_LEVEE": (255, 140, 0), T_TURNED: (150, 60, 200),
    "DEPLACEMENT_LATERAL": (0, 160, 130), "RAPPROCHEMENT_CAMERA": (120, 190, 40),
    "ELOIGNEMENT_CAMERA": (190, 190, 40), T_ABSENT: (90, 90, 90), T_OUT: (0, 0, 0),
}
UNIT_FACTOR = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


# --------------------------------------------------------------------------- #
# Petits outils
# --------------------------------------------------------------------------- #
def runs_of_true(mask):
    """Plages consécutives de True : liste de (début, fin) inclus."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    return list(zip(np.where(d == 1)[0].tolist(), (np.where(d == -1)[0] - 1).tolist()))


def label_runs(labels):
    """Plages de valeurs identiques : liste de [début, fin, valeur]."""
    runs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append([start, i - 1, labels[start]])
            start = i
    return runs


# --------------------------------------------------------------------------- #
# 1. Lecture du fichier de l'eye-tracker
# --------------------------------------------------------------------------- #
def sniff_encoding(path):
    """Encodage utilisé pour lire ET réécrire le fichier."""
    with open(path, "rb") as f:
        head = f.read(1 << 20)
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        head.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def detect_separator(path, enc):
    """Séparateur de colonnes d'après la ligne d'en-tête."""
    with open(path, "r", encoding=enc, errors="replace") as f:
        header = f.readline()
    counts = {s: header.count(s) for s in ("\t", ";", ",", "|")}
    sep = max(counts, key=counts.get)
    if counts[sep] == 0:
        raise SystemExit("Séparateur de colonnes introuvable dans l'en-tête du fichier.")
    return sep


def resolve_column(df, name):
    """Trouve une colonne même si la casse ou les espaces diffèrent."""
    if name in df.columns:
        return name
    wanted = name.strip().lower()
    for c in df.columns:
        if c.strip().lower() == wanted:
            return c
    raise SystemExit(f"Colonne '{name}' introuvable.\nColonnes disponibles : {list(df.columns)}")


def to_num(series, sep):
    """Texte -> nombre (vide / 'NaN' / illisible -> NaN ; virgule décimale acceptée)."""
    s = series.str.strip()
    if sep != ",":
        s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def eye_missing(x, y):
    """Mesure manquante : coordonnée absente/invalide OU couple égal à (0, 0)."""
    return ~np.isfinite(x) | ~np.isfinite(y) | ((x == 0) & (y == 0))


def read_gaze(args):
    """Retourne (df texte complet, séparateur, encodage, temps en s, masque manquant)."""
    path = Path(args.gaze)
    enc = sniff_encoding(path)
    sep = detect_separator(path, enc)
    # dtype=str : on relit tout en texte pour réécrire le fichier à l'identique.
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding=enc)

    tcol = resolve_column(df, args.time_col)
    t_raw = to_num(df[tcol], sep)
    if not np.isfinite(t_raw).any():
        raise SystemExit(f"Colonne de temps '{tcol}' : aucun nombre lisible.")
    t0 = np.nanmin(t_raw) if args.gaze_t0 is None else args.gaze_t0
    t_s = (t_raw - t0) * UNIT_FACTOR[args.time_unit]

    pairs = []
    if args.eyes in ("left", "both"):
        pairs.append((args.left_x, args.left_y))
    if args.eyes in ("right", "both"):
        pairs.append((args.right_x, args.right_y))
    per_eye = []
    for cx_name, cy_name in pairs:
        x = to_num(df[resolve_column(df, cx_name)], sep)
        y = to_num(df[resolve_column(df, cy_name)], sep)
        per_eye.append(eye_missing(x, y))
    reducer = np.logical_and if args.missing_rule == "all" else np.logical_or
    missing = reducer.reduce(per_eye)

    print(f"   {len(df)} lignes, séparateur {sep!r}, encodage {enc}.")
    for (cx_name, _), m in zip(pairs, per_eye):
        print(f"   Mesures manquantes ({cx_name.split()[0]}) : {100 * m.mean():.1f} %")
    print(f"   Mesures manquantes retenues (règle '{args.missing_rule}', yeux '{args.eyes}') : "
          f"{100 * missing.mean():.1f} % des lignes.")
    return df, sep, enc, t_s, missing


# --------------------------------------------------------------------------- #
# 2. Détection du visage dans la vidéo
# --------------------------------------------------------------------------- #
class FaceDetector:
    """detect(image) -> (genre, (x, y, w, h)) avec genre : 1 = face, 2 = profil, 0 = rien.
    Haar (intégré à OpenCV 4) par défaut ; YuNet si un modèle .onnx est fourni."""

    def __init__(self, width, yunet_model=None):
        self.width = width
        self.yunet = None
        if yunet_model:
            self.yunet = cv2.FaceDetectorYN.create(str(yunet_model), "", (width, 10),
                                                   0.6, 0.3, 5000)
            return
        if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
            raise SystemExit(
                f"OpenCV {cv2.__version__} ne fournit pas CascadeClassifier.\n"
                "Installez OpenCV 4 :  pip uninstall -y opencv-python && "
                "pip install \"opencv-python<5\"\n"
                "(ou fournissez un modèle YuNet avec --yunet-model).")
        base = cv2.data.haarcascades
        self.front = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self.prof = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
        if self.front.empty():
            raise SystemExit("Cascade Haar introuvable dans l'installation d'OpenCV.")

    @staticmethod
    def _largest(boxes):
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        return float(x), float(y), float(w), float(h)

    def detect(self, frame_bgr):
        h0, w0 = frame_bgr.shape[:2]
        img = cv2.resize(frame_bgr, (self.width, int(round(h0 * self.width / w0))),
                         interpolation=cv2.INTER_AREA)
        if self.yunet is not None:
            self.yunet.setInputSize((img.shape[1], img.shape[0]))
            _, faces = self.yunet.detect(img)
            if faces is not None and len(faces):
                return 1, self._largest([f[:4] for f in faces])
            return 0, None
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        ms = (max(24, img.shape[1] // 20),) * 2
        f = self.front.detectMultiScale(gray, 1.1, 5, minSize=ms)
        if len(f):
            return 1, self._largest(f)
        if not self.prof.empty():                       # profil regardant d'un côté...
            p = self.prof.detectMultiScale(gray, 1.1, 5, minSize=ms)
            if len(p):
                return 2, self._largest(p)
            pf = self.prof.detectMultiScale(cv2.flip(gray, 1), 1.1, 5, minSize=ms)
            if len(pf):                                 # ... ou de l'autre (image miroir)
                x, y, w, h = self._largest(pf)
                return 2, (img.shape[1] - x - w, y, w, h)
        return 0, None


def analyze_video(video_path, detector, step):
    """Détecte le visage toutes les `step` images.
    Retourne fps, nb d'images, taille, et les tableaux (un élément par image échantillonnée)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir la vidéo : {video_path}\n"
                         "Essayez de la convertir : ffmpeg -i video.wmv -c:v libx264 video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    kinds, boxes = [], []
    W = H = None
    i = 0
    while cap.grab():                       # grab = avancer sans décoder les images inutiles
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                kinds.append(0)
                boxes.append((np.nan,) * 4)
            else:
                if W is None:
                    H, W = frame.shape[:2]
                k, b = detector.detect(frame)
                kinds.append(k)
                boxes.append(b if b is not None else (np.nan,) * 4)
        i += 1
        if i % 1000 == 0:
            print(f"  {i}/{total}" if total else f"  {i} images", end="\r")
    cap.release()
    print(f"  {i} images lues.            ")
    if i == 0 or not kinds:
        raise SystemExit("Aucune image lue dans la vidéo.")
    b = np.array(boxes, dtype=float)
    return dict(fps=fps, n_frames=i, W=W, H=H, step=step,
                kind=np.array(kinds, dtype=np.int8),
                bx=b[:, 0], by=b[:, 1], bw=b[:, 2], bh=b[:, 3])


# --------------------------------------------------------------------------- #
# 3. Posture : référence, classification, affinage
# --------------------------------------------------------------------------- #
def smooth_track(kind, cx, cy, w, win):
    """Médiane glissante des positions du visage (limite le tremblement de la détection)."""
    out = []
    for arr in (cx, cy, w):
        s = pd.Series(np.where(kind == 1, arr, np.nan))
        sm = s.rolling(win, center=True, min_periods=1).median().to_numpy()
        out.append(np.where(kind == 1, sm, np.nan))
    return out


def classify_posture(cx, cy, w, base, thr):
    """Compare un visage à la posture de référence.
    Écarts exprimés en LARGEURS DE VISAGE de référence. Retourne (type, score) ;
    score >= 1 signifie que le seuil du comportement est atteint."""
    dx = (cx - base["cx"]) / base["w"]
    dy = (cy - base["cy"]) / base["w"]          # dy > 0 : visage plus bas dans l'image
    lr = math.log(w / base["w"])
    ls = math.log(thr["scale"])
    cand = {
        "TETE_BAISSEE": dy / thr["down"],
        "TETE_LEVEE": -dy / thr["up"],
        "DEPLACEMENT_LATERAL": abs(dx) / thr["lateral"],
        "RAPPROCHEMENT_CAMERA": lr / ls,
        "ELOIGNEMENT_CAMERA": -lr / ls,
    }
    lab = max(cand, key=cand.get)
    return (lab, cand[lab]) if cand[lab] >= 1.0 else (NORMAL, cand[lab])


def refine_absent_runs(label, kind, lscore, sample_dt, absent_min, ctx_s):
    """Traite les plages « aucun visage détecté » :
      - si le visage était déjà en train de s'écarter juste avant/après, on reprend ce
        comportement (ex. tête qui descend puis disparaît => TETE_BAISSEE) ;
      - si la plage est très courte et la posture voisine normale : simple raté du
        détecteur => NORMAL ;
      - sinon : VISAGE_ABSENT."""
    det = np.where(kind > 0)[0]
    ctx_n = max(1, int(round(ctx_s / sample_dt)))
    for a, b in runs_of_true(label == T_ABSENT):
        cands = []
        if len(det):
            k = np.searchsorted(det, a)
            if k > 0 and a - det[k - 1] <= ctx_n:
                cands.append(det[k - 1])
            if k < len(det) and det[k] - b <= ctx_n:
                cands.append(det[k])
        best = max(cands, key=lambda j: lscore[j]) if cands else None
        dur = (b - a + 1) * sample_dt
        if best is not None and label[best] != NORMAL:
            new = label[best]
        elif cands and dur < absent_min:
            new = NORMAL
        else:
            new = T_ABSENT
        label[a:b + 1] = new
    return label


def despeckle(label, min_n):
    """Supprime les états isolés plus courts que min_n échantillons quand les deux
    voisins sont identiques (bruit de détection)."""
    for _ in range(3):
        runs = label_runs(list(label))
        changed = False
        for k in range(1, len(runs) - 1):
            s, e, lab = runs[k]
            if (e - s + 1) < min_n and runs[k - 1][2] == runs[k + 1][2] != lab:
                label[s:e + 1] = runs[k - 1][2]
                changed = True
        if not changed:
            break
    return label


def posture_labels(v, ref_ok, args):
    """Étiquette de posture pour chaque image échantillonnée de la vidéo."""
    kind = v["kind"]
    cx, cy = v["bx"] + v["bw"] / 2, v["by"] + v["bh"] / 2
    w = v["bw"]
    scx, scy, sw = smooth_track(kind, cx, cy, w, args.smooth)

    # Posture de référence : visage vu ET eye-tracker qui mesurait correctement
    ref = (kind == 1) & ref_ok
    if ref.sum() < 30:
        print("   ATTENTION : très peu d'instants de référence (visage vu + regard mesuré) ;"
              " la référence est calculée sur tous les visages détectés.")
        ref = kind == 1
    if ref.sum() == 0:
        raise SystemExit("Aucun visage détecté dans la vidéo : analyse de posture impossible.\n"
                         "Vérifiez que la vidéo montre bien la personne, ou essayez --yunet-model.")
    base = dict(cx=float(np.median(scx[ref])), cy=float(np.median(scy[ref])),
                w=float(np.median(sw[ref])))
    print(f"   Posture de référence (échelle de détection) : centre=({base['cx']:.0f}, "
          f"{base['cy']:.0f}) largeur du visage={base['w']:.0f} px, calculée sur {int(ref.sum())} instants.")

    thr = dict(down=args.thr_down, up=args.thr_up, lateral=args.thr_lateral, scale=args.thr_scale)
    ns = len(kind)
    label = np.empty(ns, dtype=object)
    lscore = np.zeros(ns)
    for i in range(ns):
        if kind[i] == 1:
            label[i], lscore[i] = classify_posture(scx[i], scy[i], sw[i], base, thr)
        elif kind[i] == 2:
            label[i], lscore[i] = T_TURNED, 99.0
        else:
            label[i] = T_ABSENT

    sample_dt = v["step"] / v["fps"]
    label = refine_absent_runs(label, kind, lscore, sample_dt, args.absent_min, args.context)
    label = despeckle(label, max(1, int(round(args.min_posture / sample_dt))))
    return label


# --------------------------------------------------------------------------- #
# 4. Épisodes, résumé, graphique, vidéo annotée
# --------------------------------------------------------------------------- #
def build_episodes(type_abs, t_s, dt):
    """Plages consécutives (dans l'ordre du temps) portant le même type d'absence."""
    valid = np.where(np.isfinite(t_s))[0]
    order = valid[np.argsort(t_s[valid], kind="stable")]
    lab, ts = type_abs[order], t_s[order]
    rows, i, n = [], 0, len(lab)
    while i < n:
        if lab[i] == NA_LABEL:
            i += 1
            continue
        j = i
        while j + 1 < n and lab[j + 1] == lab[i]:
            j += 1
        rows.append({"type_absence": lab[i], "debut_s": round(float(ts[i]), 3),
                     "fin_s": round(float(ts[j] + dt), 3),
                     "duree_s": round(float(ts[j] - ts[i] + dt), 3), "n_lignes": j - i + 1})
        i = j + 1
    return pd.DataFrame(rows, columns=["type_absence", "debut_s", "fin_s", "duree_s", "n_lignes"])


def plot_timeline(episodes, duration, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 4))
    for k, name in enumerate(ALL_TYPES):
        sub = episodes[episodes.type_absence == name]
        if len(sub):
            ax.broken_barh(list(zip(sub.debut_s, sub.duree_s)), (k - 0.4, 0.8),
                           color=tuple(c / 255 for c in COLORS_RGB[name]))
    ax.set_yticks(range(len(ALL_TYPES)))
    ax.set_yticklabels(ALL_TYPES, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, duration)
    ax.set_xlabel("Temps (s)")
    ax.set_title("Mesures manquantes et cause probable")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def write_annotated_video(video_path, out_path, v, frame_label, frame_missing, t_range):
    """Vidéo de contrôle : cadre du visage détecté, bandeau rouge pendant les absences."""
    cap = cv2.VideoCapture(str(video_path))
    W, H, fps, step = v["W"], v["H"], v["fps"], v["step"]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    f0 = 0 if t_range is None else int(t_range[0] * fps)
    f1 = v["n_frames"] if t_range is None else min(v["n_frames"], int(t_range[1] * fps))
    sc = max(0.5, W / 1000)
    ns = len(v["kind"])
    i = 0
    while i < f1 and cap.grab():
        if i >= f0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            s = min(i // step, ns - 1)
            if v["kind"][s] > 0:
                k = W / v.get("det_width", 480)
                x, y, w, h = (int(v[n][s] * k) for n in ("bx", "by", "bw", "bh"))
                col = (0, 200, 0) if v["kind"][s] == 1 else (0, 140, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
            if frame_missing[i]:
                name = frame_label[i]
                cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 255), int(8 * sc))
                cv2.putText(frame, f"MESURE MANQUANTE : {name}", (int(15 * sc), int(40 * sc)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * sc, (0, 0, 255), max(2, int(2 * sc)))
            else:
                cv2.putText(frame, "mesure OK", (int(15 * sc), int(40 * sc)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * sc, (0, 200, 0), max(2, int(2 * sc)))
            cv2.putText(frame, f"t={i / fps:7.2f}s", (int(15 * sc), H - int(15 * sc)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * sc, (255, 255, 255), max(2, int(2 * sc)))
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
    ap.add_argument("video", help="Vidéo (caméra fixe filmant la personne)")
    ap.add_argument("gaze", help="Fichier texte/CSV exporté par l'eye-tracker")
    ap.add_argument("--out-dir", default=None, help="Dossier de sortie (défaut : celui du fichier de regard)")

    g = ap.add_argument_group("Colonnes du fichier de regard (défauts = votre fichier)")
    g.add_argument("--time-col", default="Capture Time")
    g.add_argument("--time-unit", choices=list(UNIT_FACTOR), default="us",
                   help="Unité de la colonne de temps (défaut : us = microsecondes)")
    g.add_argument("--gaze-t0", type=float, default=None,
                   help="Valeur du temps correspondant au début de la vidéo (défaut : plus petite valeur)")
    g.add_argument("--left-x", default="Lft X Pos")
    g.add_argument("--left-y", default="Lft Y Pos")
    g.add_argument("--right-x", default="Rt X Pos")
    g.add_argument("--right-y", default="Rt Y Pos")
    g.add_argument("--eyes", choices=["both", "left", "right"], default="both", help="Yeux utilisés")
    g.add_argument("--missing-rule", choices=["all", "any"], default="all",
                   help="all : manquant si TOUS les yeux utilisés sont manquants ; any : si au moins un")

    g = ap.add_argument_group("Détection du visage")
    g.add_argument("--step", type=int, default=2, help="Détection toutes les N images (défaut 2)")
    g.add_argument("--face-width", type=int, default=480, help="Largeur de travail de la détection (px)")
    g.add_argument("--yunet-model", default=None, help="Modèle YuNet .onnx (optionnel)")

    g = ap.add_argument_group("Seuils de posture (en largeurs de visage de référence)")
    g.add_argument("--thr-down", type=float, default=0.35, help="Tête baissée : descente du visage")
    g.add_argument("--thr-up", type=float, default=0.35, help="Tête levée : montée du visage")
    g.add_argument("--thr-lateral", type=float, default=0.5, help="Déplacement latéral")
    g.add_argument("--thr-scale", type=float, default=1.3, help="Facteur de taille du visage (rapprochement / recul)")
    g.add_argument("--smooth", type=int, default=5, help="Lissage (nb d'échantillons) de la position du visage")
    g.add_argument("--min-posture", type=float, default=0.4, help="Durée min. d'une posture (s)")
    g.add_argument("--absent-min", type=float, default=0.6,
                   help="Un visage non détecté moins longtemps est considéré comme un raté du détecteur (s)")
    g.add_argument("--context", type=float, default=1.0,
                   help="Fenêtre (s) avant/après une disparition du visage pour en déduire la cause")

    g = ap.add_argument_group("Contrôle visuel")
    g.add_argument("--annotate", action="store_true", help="Écrit une vidéo annotée")
    g.add_argument("--range", type=float, nargs=2, metavar=("DEBUT", "FIN"), default=None,
                   help="Limite la vidéo annotée à [DEBUT, FIN] secondes")
    return ap


def main():
    args = build_parser().parse_args()
    video, gaze_path = Path(args.video), Path(args.gaze)
    for p in (video, gaze_path):
        if not p.exists():
            raise SystemExit(f"Fichier introuvable : {p}")
    out_dir = Path(args.out_dir) if args.out_dir else gaze_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Fichier de regard
    print(f"[1/5] Lecture du fichier de regard : {gaze_path.name}")
    df, sep, enc, t_s, missing = read_gaze(args)
    n_rows = len(df)

    # ---- 2. Vidéo
    print(f"[2/5] Détection du visage dans la vidéo : {video.name}")
    detector = FaceDetector(args.face_width, args.yunet_model)
    v = analyze_video(video, detector, args.step)
    v["det_width"] = args.face_width
    fps, n_frames = v["fps"], v["n_frames"]
    duration = n_frames / fps
    kind = v["kind"]
    print(f"   Vidéo : {n_frames} images, {fps:.2f} i/s, {duration:.1f} s. Visage de face sur "
          f"{100 * (kind == 1).mean():.1f} % des instants, de profil {100 * (kind == 2).mean():.1f} %, "
          f"aucun {100 * (kind == 0).mean():.1f} %.")
    span = float(np.nanmax(t_s) - np.nanmin(t_s))
    print(f"   Durée du fichier de regard : {span:.1f} s ; durée de la vidéo : {duration:.1f} s.")
    if abs(span - duration) > 0.2 * duration:
        print("   ATTENTION : durées très différentes, vérifiez --time-unit / --gaze-t0 "
              "(le regard n'est probablement pas aligné sur la vidéo).")

    # ---- 3. Lien lignes du fichier <-> images de la vidéo
    print("[3/5] Posture de la personne pendant l'enregistrement")
    vf = np.where(np.isfinite(t_s), np.floor(np.nan_to_num(t_s) * fps), -1).astype(int)
    inside = (vf >= 0) & (vf < n_frames)
    ntot = np.bincount(vf[inside], minlength=n_frames)
    nmiss = np.bincount(vf[inside], weights=missing[inside].astype(float), minlength=n_frames)
    with np.errstate(invalid="ignore", divide="ignore"):
        frame_missing_ratio = np.where(ntot > 0, nmiss / ntot, 1.0)
    # images de référence : au moins 80 % des mesures valides
    ns = len(kind)
    sample_frames = np.minimum(np.arange(ns) * v["step"], n_frames - 1)
    ref_ok = (ntot[sample_frames] > 0) & (frame_missing_ratio[sample_frames] <= 0.2)

    sample_label = posture_labels(v, ref_ok, args)
    frame_idx = np.minimum(np.arange(n_frames) // v["step"], ns - 1)
    frame_label = sample_label[frame_idx].astype(object)
    frame_label = np.where(frame_label == NORMAL, T_EYE_ERROR, frame_label)  # posture normale => appareil

    # ---- 4. type_absence pour chaque ligne
    print("[4/5] Attribution du type d'absence à chaque ligne")
    type_abs = np.full(n_rows, NA_LABEL, dtype=object)
    idx = np.where(missing)[0]
    ok = inside[idx]
    type_abs[idx[ok]] = frame_label[vf[idx[ok]]]
    type_abs[idx[~ok]] = T_OUT
    df["type_absence"] = type_abs

    stem = gaze_path.stem
    f_out = out_dir / f"{stem}_type_absence{gaze_path.suffix or '.txt'}"
    f_epi = out_dir / f"{stem}_episodes.csv"
    f_png = out_dir / f"{stem}_timeline.png"
    df.to_csv(f_out, sep=sep, index=False, encoding=enc)

    valid_t = t_s[np.isfinite(t_s)]
    dt = float(np.median(np.diff(np.sort(valid_t)))) if len(valid_t) > 1 else 1 / 60
    episodes = build_episodes(type_abs, t_s, dt)
    episodes.to_csv(f_epi, index=False)
    plot_timeline(episodes, duration, f_png)

    # ---- 5. Résumé
    print("[5/5] Résumé")
    n_miss = int(missing.sum())
    print(f"\n  Lignes : {n_rows} | mesures manquantes : {n_miss} ({100 * n_miss / n_rows:.1f} %)")
    if n_miss:
        cnt = Counter(type_abs[missing])
        tab = (episodes.groupby("type_absence")
               .agg(n_episodes=("n_lignes", "size"), n_lignes=("n_lignes", "sum"),
                    duree_s=("duree_s", "sum")).reset_index())
        tab["pct_des_manquants"] = (100 * tab.n_lignes / n_miss).round(1)
        tab["duree_s"] = tab.duree_s.round(1)
        tab = tab.sort_values("n_lignes", ascending=False)
        print("\n" + tab.to_string(index=False))
        err = cnt.get(T_EYE_ERROR, 0)
        print(f"\n  => {100 * err / n_miss:.1f} % des mesures manquantes ne s'expliquent par aucun "
              f"comportement visible : erreur probable de l'eye-tracker.")
    print(f"\nFichiers :\n  {f_out}\n  {f_epi}\n  {f_png}")

    if args.annotate:
        f_vid = out_dir / f"{stem}_annotated.mp4"
        print(f"\nÉcriture de la vidéo annotée : {f_vid.name}")
        write_annotated_video(video, f_vid, v, frame_label, frame_missing_ratio >= 0.5,
                              args.range)
        print(f"  {f_vid}")


if __name__ == "__main__":
    main()
