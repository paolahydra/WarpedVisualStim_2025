#!/usr/bin/env python3
import os, sys, json, pickle
import numpy as np
import pandas as pd

def _alias_numpy_core_for_pickle_compat():
    """
    Some logs were pickled with NumPy 2.x which uses module path 'numpy._core'.
    Alias it to 'numpy.core' if running under NumPy 1.x to allow unpickling.
    """
    try:
        import numpy.core as np_core
        import sys as _sys
        _sys.modules.setdefault('numpy._core', np_core)
    except Exception:
        pass

def load_log(pkl_path):
    _alias_numpy_core_for_pickle_compat()
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def _get_refresh_rate(mon):
    # Common keys; be defensive to differing field names
    for k in ("refresh_rate","refreshRate","RefreshRate","rr","fps"):
        if k in mon and mon[k]:
            return float(mon[k])
    raise KeyError("monitor.refresh_rate not found in log['monitor']")

def _ensure_index_and_unique_frames(log):
    """
    Returns (idx, frames_unique) for ON/OFF detection.

    If 'index_to_display' not present (frame-by-frame pipeline),
    reconstruct unique frames and the index array by de-duplicating tuples.
    """
    st = log["stimulation"]
    idx = st.get("index_to_display", None)
    frames_unique = st.get("frames_unique", None)
    if frames_unique is None and "frames_unique" in log:
        frames_unique = log["frames_unique"]

    if idx is None:
        # Try frame-by-frame fallback
        frames = log.get("frames", None) or st.get("frames", None)
        if frames is None:
            raise KeyError("Neither 'index_to_display' nor per-frame 'frames' present.")
        # Each frame is a tuple: (is_display, indicator_value, display_color)
        arr = np.asarray(frames, dtype=object)
        unique, inv = np.unique(arr, axis=0, return_inverse=True)
        frames_unique = [tuple(row) for row in unique.tolist()]
        idx = inv.tolist()

    return np.asarray(idx, dtype=int), np.array(frames_unique, dtype=object)

def _detect_on_off(idx, frames_unique):
    """
    Identify ON runs and their onset/offset frames.
    Returns onset_frames, offset_frames, color_indices, color_map.
    """
    # frames_unique columns: (is_display, indicator_value, display_color)
    is_display = frames_unique[:,0].astype(int)
    gap_indices   = np.where(is_display == 0)[0]
    color_indices = np.where(is_display == 1)[0]
    if len(gap_indices) == 0 or len(color_indices) < 1:
        raise ValueError("Could not identify GAP/COLOR states in frames_unique.")
    # Map unique index -> color value
    color_map = {int(i): float(frames_unique[i,2]) for i in color_indices}

    is_on = np.isin(idx, list(color_indices))
    # Onset = first ON of a run
    onset_mask  = is_on & np.r_[True, idx[1:] != idx[:-1]]
    # Offset = last ON of a run
    offset_mask = is_on & np.r_[idx[:-1] != idx[1:], True]

    onset_frames  = np.flatnonzero(onset_mask)
    offset_frames = np.flatnonzero(offset_mask)
    return onset_frames, offset_frames, color_indices, color_map

def build_schedule(log):
    st  = log["stimulation"]
    mon = log["monitor"]
    rr  = _get_refresh_rate(mon)

    idx, frames_unique = _ensure_index_and_unique_frames(log)
    onset_frames, offset_frames, color_indices, color_map = _detect_on_off(idx, frames_unique)

    n = onset_frames.size
    onset_idx_vals = idx[onset_frames]
    onset_colors   = [color_map[int(iv)] for iv in onset_idx_vals]

    # Pull optional fields
    color_seq_logged  = st.get("color_sequence_per_flash", None)
    flash_frame_num   = st.get("flash_frame_num", None)
    midgap_frame_num  = st.get("midgap_frame_num", None)
    pregap_frame_num  = st.get("pregap_frame_num", st.get("pre_gap_frame_num", None))
    postgap_frame_num = st.get("postgap_frame_num", st.get("post_gap_frame_num", None))
    n_reps            = st.get("n_reps", None)
    rng_seed          = st.get("rng_seed", None)
    colors_tuple      = st.get("colors", None)

    df = pd.DataFrame({
        "trial": np.arange(1, n+1, dtype=int),
        "onset_frame": onset_frames,
        "offset_frame": offset_frames,
        "onset_time_s": onset_frames / rr,
        "offset_time_s": offset_frames / rr,
        "unique_index_at_onset": onset_idx_vals,
        "color_at_onset": onset_colors,
    })

    # measured duration (inclusive)
    df["measured_flash_frames"] = (df["offset_frame"] - df["onset_frame"] + 1).astype(int)
    df["measured_flash_dur_s"]  = df["measured_flash_frames"] / rr

    # planned duration (if stored)
    if flash_frame_num is not None:
        df["planned_flash_frames"] = int(flash_frame_num)
        df["planned_flash_dur_s"]  = float(flash_frame_num) / rr
        df["flash_frames_match"]   = df["measured_flash_frames"] == df["planned_flash_frames"]

    # logged randomized order (if stored)
    if color_seq_logged is not None and len(color_seq_logged) == n:
        df["logged_color_sequence"]   = color_seq_logged
        df["logged_vs_detected_match"] = np.isclose(df["logged_color_sequence"], df["color_at_onset"])

    # label as A/B if colors tuple available
    if isinstance(colors_tuple, (list, tuple)) and len(colors_tuple) == 2:
        cA, cB = float(colors_tuple[0]), float(colors_tuple[1])
        def ab(c): 
            return "A" if np.isclose(c, cA) else ("B" if np.isclose(c, cB) else "unknown")
        df["color_label"] = [ab(c) for c in df["color_at_onset"]]

    meta = {
        "refresh_rate_hz": rr,
        "n_flashes_detected": int(n),
        "n_reps_param": int(n_reps) if n_reps is not None else None,
        "rng_seed": rng_seed,
        "colors": tuple(colors_tuple) if isinstance(colors_tuple, (list, tuple)) else None,
        "flash_frame_num": int(flash_frame_num) if flash_frame_num is not None else None,
        "midgap_frame_num": int(midgap_frame_num) if midgap_frame_num is not None else None,
        "pregap_frame_num": int(pregap_frame_num) if pregap_frame_num is not None else None,
        "postgap_frame_num": int(postgap_frame_num) if postgap_frame_num is not None else None,
    }
    return df, meta

def main(pkl_path):
    log = load_log(pkl_path)
    df, meta = build_schedule(log)

    base, _ = os.path.splitext(pkl_path)
    csv_out  = base + "_schedule.csv"
    json_out = base + "_schedule_meta.json"
    df.to_csv(csv_out, index=False)
    with open(json_out, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved schedule to: {csv_out}")
    print(f"Saved metadata to: {json_out}")
    print(df.head(min(10, len(df))))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_randomized_uniform_flashes.py /path/to/log.pkl")
        sys.exit(1)
    main(sys.argv[1])
