#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_head_pitch.py
====================
Détecte, dans la vidéo de la caméra scène d'un eye-tracker (lunettes), les
périodes où le sujet BAISSE la tête ou regarde EN HAUT (plafond), c'est-à-dire
les moments où le regard sort probablement du champ de mesure.

Principe
--------
1. Flux optique (Lucas-Kanade + RANSAC) entre images successives : on mesure le
   déplacement vertical médian de la scène (en pixels).
   Scène qui descend dans l'image  => la caméra se lève (regard vers le haut).
   Scène qui monte dans l'image    => la caméra se baisse (tête baissée).
2. On intègre ce déplacement dans le temps => angle de tangage (pitch) estimé.
3. On retire la dérive / la posture "normale" (médiane glissante ou globale).
4. Seuils avec durée minimale => segments "BAS" et "HAUT".
5. (Optionnel) On croise avec le fichier de regard pour calculer le % de
   fixations manquantes dans ces segments.

Usage
-----
    python detect_head_pitch.py video.mp4
    python detect_head_pitch.py video.mp4 --vfov 60 --thr-down 20 --thr-up 20
    python detect_head_pitch.py video.mp4 --gaze gaze.csv --time-col timestamp \
           --x-col gaze_x --y-col gaze_y --time-unit s --annotate

Dépendances : pip install opencv-python numpy pandas matplotlib
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Estimation du mouvement vertical de la caméra
# --------------------------------------------------------------------------- #
def estimate_vertical_motion(video_path, width=320, min_inliers=15):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir la vidéo : {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    dys, dxs, inl_list, tex_list = [], [], [], []
    prev = None
    small_h = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        small_w = width
        small_h = int(round(h * small_w / w))
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        dx = dy = 0.0
        n_inl = 0

        if prev is not None:
            pts = cv2.goodFeaturesToTrack(prev, maxCorners=250, qualityLevel=0.01,
                                          minDistance=8)
            if pts is not None and len(pts) >= 10:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                    prev, gray, pts, None, winSize=(21, 21), maxLevel=3)
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
        dys.append(dy)
        dxs.append(dx)
        inl_list.append(n_inl)
        tex_list.append(texture)
        prev = gray
        i += 1
        if i % 500 == 0:
            msg = f"  {i}/{n_total}" if n_total else f"  {i} images"
            print(msg, end="\r")
    cap.release()
    print(f"  {i} images analysées.            ")
    return fps, small_h, np.array(dys), np.array(dxs), np.array(inl_list), np.array(tex_list)


# --------------------------------------------------------------------------- #
# 2. Angle de tangage et segmentation
# --------------------------------------------------------------------------- #
def compute_pitch(dy, small_h, vfov_deg, fps, baseline="rolling", window_s=45.0):
    deg_per_px = vfov_deg / float(small_h)
    pitch_raw = np.cumsum(dy) * deg_per_px  # + = regard vers le haut
    s = pd.Series(pitch_raw)
    if baseline == "global":
        base = np.full(len(s), s.median())
    elif baseline == "rolling":
        win = max(int(window_s * fps), 3)
        base = s.rolling(win, center=True, min_periods=1).median().values
    else:  # none
        base = np.zeros(len(s))
    return pitch_raw, pitch_raw - base


def runs_from_state(state):
    """Retourne [(start_idx, end_idx_inclus, valeur)] pour chaque plage constante."""
    out = []
    if len(state) == 0:
        return out
    start = 0
    for i in range(1, len(state) + 1):
        if i == len(state) or state[i] != state[start]:
            out.append((start, i - 1, int(state[start])))
            start = i
    return out


def segment(pitch_rel, fps, thr_down, thr_up, min_dur, merge_gap):
    state = np.zeros(len(pitch_rel), dtype=int)
    state[pitch_rel <= -abs(thr_down)] = -1  # bas
    state[pitch_rel >= abs(thr_up)] = 1      # haut

    # Combler les petits trous (même état de part et d'autre)
    gap_n = int(merge_gap * fps)
    runs = runs_from_state(state)
    for k in range(1, len(runs) - 1):
        s, e, v = runs[k]
        if v == 0 and (e - s + 1) <= gap_n and runs[k - 1][2] == runs[k + 1][2] != 0:
            state[s:e + 1] = runs[k - 1][2]

    # Supprimer les segments trop courts
    min_n = int(min_dur * fps)
    for s, e, v in runs_from_state(state):
        if v != 0 and (e - s + 1) < min_n:
            state[s:e + 1] = 0

    segs = []
    for s, e, v in runs_from_state(state):
        if v == 0:
            continue
        seg_pitch = pitch_rel[s:e + 1]
        peak = seg_pitch.min() if v == -1 else seg_pitch.max()
        segs.append({
            "type": "TETE_BAISSEE" if v == -1 else "REGARD_HAUT",
            "start_frame": s,
            "end_frame": e,
            "start_s": round(s / fps, 3),
            "end_s": round((e + 1) / fps, 3),
            "duration_s": round((e - s + 1) / fps, 3),
            "peak_angle_deg": round(float(peak), 1),
        })
    return state, segs


# --------------------------------------------------------------------------- #
# 3. Croisement avec les données de regard (optionnel)
# --------------------------------------------------------------------------- #
def load_gaze_missing(path, time_col, x_col, y_col, time_unit, offset):
    df = pd.read_csv(path)
    t = pd.to_numeric(df[time_col], errors="coerce").astype(float)
    t = t * {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[time_unit]
    t = t - (t.iloc[0] if offset is None else offset)
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    normalized = np.nanmax(np.r_[x.values, y.values]) <= 1.5
    missing = x.isna() | y.isna()
    if normalized:
        missing |= (x < 0) | (x > 1) | (y < 0) | (y > 1)
    return pd.DataFrame({"t": t, "missing": missing.astype(int)})


def add_gaze_stats(segs, gaze, total_dur):
    for sg in segs:
        m = gaze[(gaze.t >= sg["start_s"]) & (gaze.t < sg["end_s"])]
        sg["gaze_missing_pct"] = round(100 * m.missing.mean(), 1) if len(m) else np.nan
    return segs


# --------------------------------------------------------------------------- #
# 4. Sorties : graphique + vidéo annotée
# --------------------------------------------------------------------------- #
def plot_result(t, pitch_rel, segs, thr_down, thr_up, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, pitch_rel, lw=0.8, color="k", label="Angle vertical relatif (°)")
    ax.axhline(thr_up, color="tab:orange", ls="--", lw=0.8)
    ax.axhline(-thr_down, color="tab:blue", ls="--", lw=0.8)
    for sg in segs:
        c = "tab:blue" if sg["type"] == "TETE_BAISSEE" else "tab:orange"
        ax.axvspan(sg["start_s"], sg["end_s"], color=c, alpha=0.3)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Haut (+)  /  Bas (-)  [°]")
    ax.set_title("Détection tête baissée (bleu) / regard vers le haut (orange)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def write_annotated_video(video_path, out_path, state, pitch_rel, fps):
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(state):
            break
        v = state[i]
        if v != 0:
            label, color = (("TETE BAISSEE", (255, 120, 0)) if v == -1
                            else ("REGARD VERS LE HAUT", (0, 140, 255)))
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 12)
            cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        cv2.putText(frame, f"{pitch_rel[i]:+.1f} deg", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        vw.write(frame)
        i += 1
    cap.release()
    vw.release()


# --------------------------------------------------------------------------- #
# Programme principal
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="Vidéo de la caméra scène de l'eye-tracker")
    ap.add_argument("--out-dir", default=None, help="Dossier de sortie (défaut : à côté de la vidéo)")
    ap.add_argument("--vfov", type=float, default=60.0,
                    help="Champ de vision VERTICAL de la caméra en degrés (défaut 60). "
                         "À régler selon votre modèle de lunettes.")
    ap.add_argument("--thr-down", type=float, default=20.0, help="Seuil tête baissée (°)")
    ap.add_argument("--thr-up", type=float, default=20.0, help="Seuil regard en haut (°)")
    ap.add_argument("--min-dur", type=float, default=0.4, help="Durée min. d'un segment (s)")
    ap.add_argument("--merge-gap", type=float, default=0.4, help="Fusionne les segments séparés de moins de X s")
    ap.add_argument("--baseline", choices=["rolling", "global", "none"], default="rolling",
                    help="Référence de la posture 'normale' (défaut : médiane glissante)")
    ap.add_argument("--window", type=float, default=45.0,
                    help="Fenêtre (s) de la médiane glissante (défaut 45)")
    # Regard
    ap.add_argument("--gaze", default=None, help="CSV de regard/fixations (optionnel)")
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--x-col", default="x")
    ap.add_argument("--y-col", default="y")
    ap.add_argument("--time-unit", choices=["s", "ms", "us", "ns"], default="s")
    ap.add_argument("--gaze-t0", type=float, default=None,
                    help="Instant (dans l'unité --time-unit) correspondant au début de la vidéo "
                         "(défaut : 1er échantillon du CSV)")
    ap.add_argument("--annotate", action="store_true", help="Écrit une vidéo annotée")
    args = ap.parse_args()

    video = Path(args.video)
    out_dir = Path(args.out_dir) if args.out_dir else video.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem

    print(f"[1/4] Analyse du mouvement de la caméra : {video.name}")
    fps, small_h, dy, dx, inl, tex = estimate_vertical_motion(video)

    print("[2/4] Estimation de l'angle vertical")
    pitch_raw, pitch_rel = compute_pitch(dy, small_h, args.vfov, fps,
                                         args.baseline, args.window)
    t = np.arange(len(dy)) / fps

    print("[3/4] Détection des segments")
    state, segs = segment(pitch_rel, fps, args.thr_down, args.thr_up,
                          args.min_dur, args.merge_gap)

    if args.gaze:
        off = None if args.gaze_t0 is None else args.gaze_t0 * {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[args.time_unit]
        gaze = load_gaze_missing(args.gaze, args.time_col, args.x_col, args.y_col,
                                 args.time_unit, off)
        segs = add_gaze_stats(segs, gaze, t[-1])
        in_seg = np.zeros(len(gaze), dtype=bool)
        for sg in segs:
            in_seg |= ((gaze.t >= sg["start_s"]) & (gaze.t < sg["end_s"])).values
        print(f"   Mesures manquantes DANS les segments : "
              f"{100 * gaze.missing[in_seg].mean():.1f} %"
              if in_seg.any() else "   Aucun segment détecté.")
        if (~in_seg).any():
            print(f"   Mesures manquantes HORS segments     : "
                  f"{100 * gaze.missing[~in_seg].mean():.1f} %")

    # Sauvegardes
    frames_csv = out_dir / f"{stem}_frames.csv"
    segs_csv = out_dir / f"{stem}_segments.csv"
    plot_png = out_dir / f"{stem}_pitch.png"

    pd.DataFrame({
        "frame": np.arange(len(t)), "time_s": t,
        "pitch_raw_deg": pitch_raw, "pitch_rel_deg": pitch_rel,
        "state": state,  # -1 bas, 0 normal, +1 haut
        "tracking_inliers": inl, "texture": tex,
    }).to_csv(frames_csv, index=False)
    seg_df = pd.DataFrame(segs)
    seg_df.to_csv(segs_csv, index=False)
    plot_result(t, pitch_rel, segs, args.thr_down, args.thr_up, plot_png)

    print("[4/4] Résultats")
    if segs:
        print(seg_df.to_string(index=False))
        tot_down = seg_df[seg_df.type == "TETE_BAISSEE"].duration_s.sum()
        tot_up = seg_df[seg_df.type == "REGARD_HAUT"].duration_s.sum()
        print(f"\n   Durée totale tête baissée : {tot_down:.1f} s "
              f"({100 * tot_down / t[-1]:.1f} % de la vidéo)")
        print(f"   Durée totale regard haut  : {tot_up:.1f} s "
              f"({100 * tot_up / t[-1]:.1f} % de la vidéo)")
    else:
        print("   Aucun segment détecté (essayez de baisser --thr-down / --thr-up).")

    if args.annotate:
        ann = out_dir / f"{stem}_annotated.mp4"
        print(f"   Écriture de la vidéo annotée : {ann.name}")
        write_annotated_video(video, ann, state, pitch_rel, fps)

    print(f"\nFichiers : {segs_csv.name}, {frames_csv.name}, {plot_png.name}")


if __name__ == "__main__":
    main()
