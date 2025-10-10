# extract_stim_schedules_backbone.py
import sys, pickle, pathlib
from collections.abc import Mapping
import pandas as pd
from itertools import product

# ---------- Compatibility for older NumPy pickles ----------
try:
    import numpy as np
    import sys as _sys
    _sys.modules.setdefault('numpy._core', np.core)
except Exception:
    pass

# ---------- Loading ----------
def load_pickle(pkl_path: str):
    # 1) plain pickle
    try:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e1:
        err1 = e1
    # 2) numpy alias fallback
    try:
        import numpy as _np
        _sys.modules.setdefault('numpy._core', _np.core)
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e2:
        err2 = e2
    # 3) pandas
    try:
        return pd.read_pickle(pkl_path)
    except Exception as e3:
        err3 = e3
    # 4) dill (optional)
    try:
        import dill
        with open(pkl_path, 'rb') as f:
            return dill.load(f)
    except Exception as e4:
        err4 = e4
    raise RuntimeError(
        "Could not unpickle file.\n"
        f"1) pickle.load -> {type(err1).__name__}: {err1}\n"
        f"2) pickle.load (+numpy alias) -> {type(err2).__name__}: {err2}\n"
        f"3) pandas.read_pickle -> {type(err3).__name__}: {err3}\n"
        f"4) dill.load -> {type(err4).__name__}: {err4}"
    )

# ---------- Locate per-stimulus logs ----------
def get_logs_container(data):
    """Returns {stim_key: stim_log} (e.g., '002_DriftingGratingCircle': {...})."""
    if isinstance(data, dict):
        stim = data.get('stimulation', {})
        if isinstance(stim, Mapping):
            ilogs = stim.get('individual_logs', None)
            if isinstance(ilogs, Mapping) and ilogs:
                return ilogs
    # fallback(s)
    if isinstance(data, Mapping):
        for k in ('individual_logs','logs','log'):
            v = data.get(k)
            if isinstance(v, Mapping) and v:
                return v
    return None

# ---------- Minimal utilities ----------
def class_of_stim_key(stim_key: str) -> str:
    """'002_DriftingGratingCircle' -> 'DriftingGratingCircle'"""
    return stim_key.split('_', 1)[-1] if '_' in stim_key else stim_key

def make_csv_friendly(df: pd.DataFrame) -> pd.DataFrame:
    """Convert lists/dicts in cells to CSV-friendly forms."""
    if df is None or df.empty:
        return df
    for c in df.columns:
        df[c] = df[c].apply(lambda x: tuple(x) if isinstance(x, list)
                            else (repr(x) if isinstance(x, dict) else x))
    return df

def ensure_identity_cols(df: pd.DataFrame, stim_key: str, stim_class: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # Put identity columns up front
    df.insert(0, "stim_key", stim_key)
    df.insert(1, "stim_class", stim_class)
    return df

# ---------- YOUR extractors: plug your working code here ----------
def extract_DriftingGratingMultipleCircle(stim_key, stim_log, ctx):
    """
    Trial-level extractor for DriftingGrating(Multiple)Circle.

    Returns one row per trial with real onset/offset frame numbers.
    Identity columns ('stim_key','stim_class') are intentionally NOT added
    to avoid colliding with ensure_identity_cols() upstream.

    Output columns (order):
      ['trial','iteration','onset_frame','offset_frame','sf','tf','direction','contrast',
       'radius','center','block_dur','midgap_dur','pregap_dur','postgap_dur',
       'is_random_start_phase','coordinate','background','smooth_width_ratio','is_smooth_edge']
    """
    import numpy as np
    import pandas as pd

    if not isinstance(stim_log, dict):
        raise ValueError("stim_log must be a dict")
    if "frames_unique" not in stim_log or "index_to_display" not in stim_log:
        raise ValueError("stim_log must contain 'frames_unique' and 'index_to_display'")

    fu = np.array(stim_log["frames_unique"], dtype=object)
    idx = np.asarray(stim_log["index_to_display"], dtype=int)

    # ---- helpers ----
    def _is_seq(x):
        return isinstance(x, (list, tuple, np.ndarray))

    def _is_on_entry(entry):
        if isinstance(entry, dict):
            for k in ("is_display", "display", "is_on", "on"):
                if k in entry:
                    return bool(entry[k])
            for k in ("sf", "tf", "dire", "direction", "con", "contrast"):
                if k in entry:
                    return True
            return False
        if _is_seq(entry) and len(entry) > 0:
            try:
                return float(entry[0]) == 1.0
            except Exception:
                return False
        return False

    def _as_center_tuple(c):
        if isinstance(c, (list, tuple, np.ndarray)) and len(c) == 2:
            try:
                return (float(c[0]), float(c[1]))
            except Exception:
                return tuple(c.tolist()) if isinstance(c, np.ndarray) else tuple(c)
        return c

    def _params_from_entry(entry):
        sf = tf = direction = contrast = radius = center = np.nan
        if isinstance(entry, dict):
            sf = entry.get("sf", sf)
            tf = entry.get("tf", tf)
            direction = entry.get("direction", entry.get("dire", direction))
            contrast  = entry.get("contrast",  entry.get("con",  contrast))
            radius = entry.get("radius", radius)
            center = entry.get("center", entry.get("centre", center))
        elif _is_seq(entry):
            e = entry.tolist() if isinstance(entry, np.ndarray) else entry
            # DG layout in your logs: (1, 1, sf, tf, dir, con, radius, center, phase, ...)
            if len(e) >= 8 and float(e[0]) == 1.0:
                sf, tf, direction, contrast, radius, center = e[2], e[3], e[4], e[5], e[6], e[7]
            elif len(e) >= 7:
                # fallback: [flag, sf, tf, dir, con, radius, center, ...]
                sf, tf, direction, contrast, radius, center = e[1], e[2], e[3], e[4], e[5], e[6]
        center = _as_center_tuple(center)
        return sf, tf, direction, contrast, radius, center

    is_on = np.array([_is_on_entry(x) for x in fu], dtype=bool)

    # iteration (global default; per-entry may override)
    iteration_global = stim_log.get("iteration", 1)
    try:
        iteration_global = int(iteration_global)
    except Exception:
        pass

    # ---- collect ON frames with parameters ----
    per_frame = []
    for fnum, uix in enumerate(idx):
        if not (0 <= uix < fu.shape[0]):
            continue
        if not is_on[uix]:
            continue
        entry = fu[uix]
        sf, tf, direction, contrast, radius, center = _params_from_entry(entry)
        it = iteration_global
        if isinstance(entry, dict) and "iteration" in entry:
            try:
                it = int(entry["iteration"])
            except Exception:
                it = entry["iteration"]
        per_frame.append({
            "frame": int(fnum),
            "iteration": it,
            "sf": sf, "tf": tf, "direction": direction, "contrast": contrast,
            "radius": radius, "center": center,
        })

    if not per_frame:
        raise ValueError("No ON frames detected in index_to_display mapping.")

    # ---- collapse consecutive frames with identical key into trials ----
    def _trial_key(d):
        return (d["iteration"], d["sf"], d["tf"], d["direction"], d["contrast"], d["center"])

    rows, start, trial = [], 0, 0
    while start < len(per_frame):
        trial += 1
        k = _trial_key(per_frame[start])
        onset = per_frame[start]["frame"]
        end = start
        while (end + 1 < len(per_frame)
               and _trial_key(per_frame[end + 1]) == k
               and per_frame[end + 1]["frame"] == per_frame[end]["frame"] + 1):
            end += 1
        offset = per_frame[end]["frame"]
        d0 = per_frame[start]
        rows.append({
            "trial": trial,
            "iteration": d0["iteration"],
            "onset_frame": onset,
            "offset_frame": offset,
            "sf": d0["sf"],
            "tf": d0["tf"],
            "direction": d0["direction"],
            "contrast": d0["contrast"],
            "radius": d0["radius"],
            "center": d0["center"],
        })
        start = end + 1

    import pandas as pd
    df = pd.DataFrame(rows)

    # ---- attach globals (replicated per row) ----
    globals_map = {
        "block_dur":             stim_log.get("block_dur", np.nan),
        "midgap_dur":            stim_log.get("midgap_dur", np.nan),
        "pregap_dur":            stim_log.get("pregap_dur", np.nan),
        "postgap_dur":           stim_log.get("postgap_dur", np.nan),
        "is_random_start_phase": stim_log.get("is_random_start_phase", False),
        "coordinate":            stim_log.get("coordinate", None),
        "background":            stim_log.get("background", np.nan),
        "smooth_width_ratio":    stim_log.get("smooth_width_ratio", np.nan),
        "is_smooth_edge":        stim_log.get("is_smooth_edge", False),
    }
    for k, v in globals_map.items():
        if isinstance(v, list):
            v = tuple(v)
        df[k] = v

    # ---- order/ensure columns (no stim_key/stim_class here) ----
    requested = [
        "trial","iteration","onset_frame","offset_frame",
        "sf","tf","direction","contrast","radius","center",
        "block_dur","midgap_dur","pregap_dur","postgap_dur",
        "is_random_start_phase","coordinate","background","smooth_width_ratio","is_smooth_edge",
    ]
    for col in requested:
        if col not in df.columns:
            df[col] = np.nan
    return df[requested]







def extract_DriftingGratingCircle(stim_key, stim_log, ctx):
    """
    Frame-level extractor for DriftingGratingCircle with real onset/offset frames.
    """
    import numpy as np
    import pandas as pd

    # ---- refresh rate ----
    rr = None
    mon = ctx.get("monitor") if isinstance(ctx, dict) else None
    if isinstance(mon, dict):
        for k in ("refresh_rate","refreshRate","RefreshRate","rr","fps"):
            if k in mon and mon[k]:
                rr = float(mon[k]); break
    if rr is None or rr <= 0:
        rr = 60.0
    dt = 1.0 / rr

    # ---- inputs ----
    frames_unique = stim_log.get("frames_unique")
    if not isinstance(frames_unique, (list, tuple)) or len(frames_unique) < 3:
        raise ValueError("frames_unique missing or malformed")
    idx = np.asarray(stim_log["index_to_display"], dtype=int)

    fu = np.array(frames_unique, dtype=object)

    # Robust is_display extraction (treat None as 0)
    def _is_disp(fr):
        try:
            x = fr[0]
            return int(x) if x is not None else 0
        except Exception:
            return 0

    is_display = np.array([_is_disp(fr) for fr in fu], dtype=int)
    on_indices = np.where(is_display == 1)[0]
    if on_indices.size == 0:
        raise ValueError("No ON states in frames_unique (is_display==1)")

    # ON/OFF runs
    is_on = np.isin(idx, on_indices)
    onset_mask  = is_on & np.r_[True, idx[1:] != idx[:-1]]
    offset_mask = is_on & np.r_[idx[:-1] != idx[1:], True]
    onset_frames  = np.flatnonzero(onset_mask)
    offset_frames = np.flatnonzero(offset_mask)
    if onset_frames.size == 0:
        raise ValueError("No ON transitions detected")
    onset_idx_vals = idx[onset_frames]

    # Condition id from ON index: ON=2*c+1 → c=(i-1)//2
    n_cond = (len(frames_unique) - 1) // 2

    def _params_from_on_frame(fr):
        """
        Accepts the ON-frame entry and extracts (sf, tf, dire, con, radius).
        Works for list/tuple or dict payloads.
        """
        if isinstance(fr, dict):
            sf   = fr.get("sf")
            tf   = fr.get("tf")
            dire = fr.get("dire", fr.get("direction"))
            con  = fr.get("con",  fr.get("contrast"))
            rad  = fr.get("radius")
            return sf, tf, dire, con, rad
        if isinstance(fr, (list, tuple)):
            # Common layout: [flag, sf, tf, dire, con, radius, ...]
            if len(fr) >= 6:
                return fr[1], fr[2], fr[3], fr[4], fr[5]
        # Fallback: fill NaNs
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    cond_ids = ((onset_idx_vals - 1) // 2).astype(int)
    rows = []
    for i, (f_on, f_off, cid) in enumerate(zip(onset_frames, offset_frames, cond_ids), start=1):
        if 0 <= cid < n_cond:
            on_fr = frames_unique[2 * cid + 1]
        else:
            on_fr = None
        sf, tf, dire, con, rad = _params_from_on_frame(on_fr) if on_fr is not None else (np.nan,)*5
        rows.append({
            "trial": i,
            "onset_frame": int(f_on),
            "offset_frame": int(f_off),
            "onset_time_s": float(f_on * dt),
            "offset_time_s": float((f_off + 1) * dt),
            "sf": sf, "tf": tf, "dire": dire, "con": con, "radius": rad,
        })

    df = pd.DataFrame(rows)

    # Attach useful globals per row
    for k in ("block_dur","midgap_dur","iteration","is_blank_block",
              "coordinate","background","center","is_random_start_phase",
              "is_smooth_edge","smooth_width_ratio"):
        if k in stim_log and k not in df.columns:
            v = stim_log[k]
            if isinstance(v, list):
                v = tuple(v)
            df[k] = [v] * len(df)

    return df




def extract_StaticGratingCircle(stim_key, stim_log, ctx):
    """
    StaticGratingCircle with true onset/offset:
    - Detect ON/OFF runs from index_to_display (or frame_config) and frames_unique(_compact).
    - Return one row per ON presentation with onset/offset frame/time and (sf, ph, ori, con, radius).
    """
    import numpy as np
    import pandas as pd

    # ---- resolve frames_unique_compact / _frames_unique_compact ----
    frames_unique = None
    for key in ("frames_unique_compact", "_frames_unique_compact"):
        if key in stim_log and isinstance(stim_log[key], (list, tuple)):
            frames_unique = stim_log[key]
            break
    if frames_unique is None:
        return pd.DataFrame([{"_extract_error": "SGC: missing frames_unique_compact/_frames_unique_compact"}])

    # ---- resolve index_to_display / frame_config (ints or dicts with frame_idx) ----
    if "index_to_display" in stim_log:
        idx = np.asarray(stim_log["index_to_display"], dtype=int)
    elif "frame_config" in stim_log:
        fc = stim_log["frame_config"]
        if len(fc) > 0 and isinstance(fc[0], dict) and "frame_idx" in fc[0]:
            idx = np.asarray([int(d["frame_idx"]) for d in fc], dtype=int)
        else:
            idx = np.asarray(fc, dtype=int)
    else:
        return pd.DataFrame([{"_extract_error": "SGC: missing index_to_display/frame_config"}])

    # ---- monitor refresh rate ----
    rr = None
    mon = ctx.get("monitor") if isinstance(ctx, dict) else None
    if isinstance(mon, dict):
        for k in ("refresh_rate","refreshRate","RefreshRate","rr","fps"):
            if k in mon and mon[k]:
                rr = float(mon[k]); break
    if not rr or rr <= 0:
        return pd.DataFrame([{"_extract_error": "SGC: missing/invalid monitor.refresh_rate"}])
    dt = 1.0 / rr

    # ---- ON/OFF detection ----
    # Convention: frames_unique[0] = GAP; for condition c: ON = 2*c+1, OFF = 2*c+2
    n_cond = (len(frames_unique) - 1) // 2

    def cond_params(cid):
        # ON frame payload has (flag, sf, ph, ori, con, radius, ...)
        fr_tuple = frames_unique[2 * cid + 1]
        return tuple(fr_tuple[1:6])  # (sf, ph, ori, con, radius)

    is_gap = (idx == 0)
    is_on  = (~is_gap) & (idx % 2 == 1)

    # onsets: first ON of a run; offsets: last ON of a run
    onset_mask  = is_on & np.r_[True, idx[1:] != idx[:-1]]
    offset_mask = is_on & np.r_[idx[:-1] != idx[1:], True]

    onset_frames  = np.flatnonzero(onset_mask)
    offset_frames = np.flatnonzero(offset_mask)
    if onset_frames.size == 0:
        return pd.DataFrame([{"_extract_error": "SGC: no ON onsets detected"}])

    # map ON frame index -> condition id
    cond_ids = ((idx[onset_frames] - 1) // 2).astype(int)
    cond_ids = np.clip(cond_ids, 0, max(n_cond - 1, 0))

    # ---- assemble rows ----
    rows = []
    for i, (f_on, f_off, cid) in enumerate(zip(onset_frames, offset_frames, cond_ids), start=1):
        try:
            sf, ph, ori, con, rad = cond_params(int(cid))
        except Exception:
            rows.append({"_extract_error": f"SGC: bad cond id {int(cid)}",
                         "trial": i, "onset_frame": int(f_on), "offset_frame": int(f_off)})
            continue
        rows.append({
            "trial": i,
            "onset_frame": int(f_on),
            "offset_frame": int(f_off),
            "onset_time_s": float(f_on * dt),
            # offset_time_s is the boundary after the last included ON frame
            "offset_time_s": float((f_off + 1) * dt),
            "sf": sf, "ph": ph, "ori": ori, "con": con, "radius": rad,
            "measured_frames": int(f_off - f_on + 1),
            "measured_dur_s": float((f_off - f_on + 1) * dt),
        })

    df = pd.DataFrame(rows)

    # ---- optional: attach useful globals per row ----
    globals_like = ("display_dur", "midgap_dur", "iteration", "is_blank_block",
                    "coordinate", "background", "center")
    for k in globals_like:
        if k in stim_log and k not in df.columns:
            val = stim_log[k]
            if isinstance(val, list):
                val = tuple(val)
            df[k] = [val] * len(df)

    return df



def extract_RandomizedUniformFlashes(stim_key, stim_log, ctx):
    """
    Per-flash schedule for RandomizedUniformFlashes, one row per ON onset.
    Robust to:
      - frames: 'frames_unique_compact' | '_frames_unique_compact' | 'frames_unique'
      - indices: 'index_to_display' | 'frame_config' (ints or dicts with 'frame_idx')
      - refresh rate keys: refresh_rate / refreshRate / RefreshRate / rr / fps

    Output columns (best-effort):
      trial, onset_frame, offset_frame, onset_time_s, offset_time_s,
      unique_index_at_onset, color_at_onset,
      measured_flash_frames, measured_flash_dur_s,
      (optional) planned_flash_frames, planned_flash_dur_s, flash_frames_match,
      (optional) logged_color_sequence, logged_vs_detected_match,
      (optional) color_label (A/B) if st['colors'] present,
      plus a few useful globals repeated per row.
    """
    import numpy as np
    import pandas as pd

    # -------- helpers --------
    def _get_refresh_rate(ctx):
        cand = None
        mon = ctx.get("monitor") if isinstance(ctx, dict) else None
        if isinstance(mon, dict):
            for k in ("refresh_rate","refreshRate","RefreshRate","rr","fps"):
                if k in mon and mon[k]:
                    cand = float(mon[k]); break
        if not cand or cand <= 0:
            # final fallback: try at top level
            top = ctx.get("top") if isinstance(ctx, dict) else None
            if isinstance(top, dict) and "monitor" in top and isinstance(top["monitor"], dict):
                for k in ("refresh_rate","refreshRate","RefreshRate","rr","fps"):
                    v = top["monitor"].get(k)
                    if v:
                        cand = float(v); break
        if not cand or cand <= 0:
            raise KeyError("monitor.refresh_rate not found")
        return cand

    def _resolve_frames_unique_and_idx(stim_log, ctx):
        # frames_unique* from stim_log first
        fu = None
        for k in ("frames_unique_compact","_frames_unique_compact","frames_unique"):
            if k in stim_log and isinstance(stim_log[k], (list, tuple)):
                fu = stim_log[k]; break
        # idx: index_to_display preferred, else frame_config
        idx = None
        if "index_to_display" in stim_log:
            idx = np.asarray(stim_log["index_to_display"], dtype=int)
        elif "frame_config" in stim_log:
            fc = stim_log["frame_config"]
            if len(fc) > 0 and isinstance(fc[0], dict) and "frame_idx" in fc[0]:
                idx = np.asarray([int(d["frame_idx"]) for d in fc], dtype=int)
            else:
                idx = np.asarray(fc, dtype=int)

        # If still missing, check ctx['top'] (rare)
        if (fu is None or idx is None) and isinstance(ctx.get("top"), dict):
            top = ctx["top"]
            # sometimes stored at top['stimulation'] as a parallel view
            stim_top = top.get("stimulation", {})
            if fu is None:
                for k in ("frames_unique_compact","_frames_unique_compact","frames_unique"):
                    v = stim_top.get(k)
                    if isinstance(v, (list, tuple)):
                        fu = v; break
            if idx is None:
                if "index_to_display" in stim_top:
                    idx = np.asarray(stim_top["index_to_display"], dtype=int)
                elif "frame_config" in stim_top:
                    fc = stim_top["frame_config"]
                    if len(fc) > 0 and isinstance(fc[0], dict) and "frame_idx" in fc[0]:
                        idx = np.asarray([int(d["frame_idx"]) for d in fc], dtype=int)
                    else:
                        idx = np.asarray(fc, dtype=int)

        if fu is None or idx is None:
            raise KeyError("RUF: missing frames_unique*/index_to_display/frame_config")

        fu = np.array(fu, dtype=object)
        return idx, fu

    def _detect_onsets_offsets(idx, fu):
        """
        frames_unique rows expected like: (is_display, indicator_value, display_color)
        Identify ON runs among indices that map to rows with is_display==1.
        """
        # If fu doesn't have at least 3 fields per row, fail loudly.
        try:
            is_display = fu[:, 0].astype(int)
            display_color = fu[:, 2]  # may be float/int
        except Exception as e:
            raise ValueError(f"RUF: frames_unique shape unexpected: {fu.shape}") from e

        color_indices = np.where(is_display == 1)[0]
        if color_indices.size == 0:
            raise ValueError("RUF: no ON states in frames_unique (is_display==1)")

        color_map = {int(i): float(display_color[i]) for i in color_indices}
        is_on = np.isin(idx, color_indices)

        onset_mask  = is_on & np.r_[True, idx[1:] != idx[:-1]]
        offset_mask = is_on & np.r_[idx[:-1] != idx[1:], True]

        onset_frames  = np.flatnonzero(onset_mask)
        offset_frames = np.flatnonzero(offset_mask)

        return onset_frames, offset_frames, color_map

    # -------- main extraction --------
    rr = _get_refresh_rate(ctx)
    idx, fu = _resolve_frames_unique_and_idx(stim_log, ctx)
    onset_frames, offset_frames, color_map = _detect_onsets_offsets(idx, fu)

    n = onset_frames.size
    onset_idx_vals = idx[onset_frames]
    color_at_onset = [color_map.get(int(iv), np.nan) for iv in onset_idx_vals]

    # Optional fields stored in stim_log
    st = stim_log  # alias
    color_seq_logged  = st.get("color_sequence_per_flash", None)
    flash_frame_num   = st.get("flash_frame_num", st.get("flash_frames"))
    midgap_frame_num  = st.get("midgap_frame_num", st.get("midgap_frames"))
    pregap_frame_num  = st.get("pregap_frame_num", st.get("pre_gap_frame_num"))
    postgap_frame_num = st.get("postgap_frame_num", st.get("post_gap_frame_num"))
    n_reps            = st.get("n_reps", None)
    rng_seed          = st.get("rng_seed", None)
    colors_tuple      = st.get("colors", None)

    df = pd.DataFrame({
        "trial":            np.arange(1, n+1, dtype=int),
        "onset_frame":      onset_frames,
        "offset_frame":     offset_frames,
        "onset_time_s":     onset_frames / rr,
        "offset_time_s":    offset_frames / rr,
        "unique_index_at_onset": onset_idx_vals,
        "color_at_onset":   color_at_onset,
    })

    # measured duration (inclusive)
    df["measured_flash_frames"] = (df["offset_frame"] - df["onset_frame"] + 1).astype(int)
    df["measured_flash_dur_s"]  = df["measured_flash_frames"] / rr

    # planned duration (if stored)
    if flash_frame_num is not None:
        try:
            pf = int(flash_frame_num)
            df["planned_flash_frames"] = pf
            df["planned_flash_dur_s"]  = pf / rr
            df["flash_frames_match"]   = df["measured_flash_frames"] == df["planned_flash_frames"]
        except Exception:
            pass

    # logged randomized order (if stored)
    if isinstance(color_seq_logged, (list, tuple)) and len(color_seq_logged) == n:
        try:
            logged = np.asarray(color_seq_logged, dtype=float)
            df["logged_color_sequence"]    = logged
            df["logged_vs_detected_match"] = np.isclose(logged, df["color_at_onset"].astype(float))
        except Exception:
            df["logged_color_sequence"]    = color_seq_logged
            df["logged_vs_detected_match"] = None

    # A/B labeling if exactly two colors were used
    if isinstance(colors_tuple, (list, tuple)) and len(colors_tuple) == 2:
        cA, cB = float(colors_tuple[0]), float(colors_tuple[1])
        def _ab(c):
            try:
                return "A" if np.isclose(float(c), cA) else ("B" if np.isclose(float(c), cB) else "unknown")
            except Exception:
                return "unknown"
        df["color_label"] = [ _ab(c) for c in df["color_at_onset"] ]

    # repeat a few useful globals per row
    globals_like = ("flash_frame_num","midgap_frame_num","pregap_frame_num","postgap_frame_num",
                    "n_reps","rng_seed","colors","coordinate","background")
    gl_vals = {
        "flash_frame_num":   flash_frame_num,
        "midgap_frame_num":  midgap_frame_num,
        "pregap_frame_num":  pregap_frame_num,
        "postgap_frame_num": postgap_frame_num,
        "n_reps":            n_reps,
        "rng_seed":          rng_seed,
        "colors":            tuple(colors_tuple) if isinstance(colors_tuple, (list, tuple)) else colors_tuple,
        "coordinate":        st.get("coordinate"),
        "background":        st.get("background"),
    }
    for k, v in gl_vals.items():
        if v is not None and k not in df.columns:
            if isinstance(v, list):
                v = tuple(v)
            df[k] = [v] * len(df)

    return df



def extract_StaticImages(stim_key, stim_log, ctx):
    """
    Extract full randomized StaticImages display schedule from a stim_log dict.

    Robust to slightly different key layouts (frames_unique/_compact, index_to_display/index).
    Returns a DataFrame (one row per displayed image) with:
        presentation_idx, iteration, within_iteration_idx, unique_index, image_index,
        start_frame, end_frame, onset_s, offset_s, duration_s, n_frames,
        plus useful global metadata repeated per row.

    No file I/O — suitable for use in backbone pipeline.
    """
    import numpy as np
    import pandas as pd

    # ---------- helper functions ----------
    def _get_refresh_rate(ctx_or_stim):
        """Get refresh rate from ctx['monitor'] or stim_log['monitor']."""
        mon = None
        if isinstance(ctx_or_stim, dict):
            if "monitor" in ctx_or_stim and isinstance(ctx_or_stim["monitor"], dict):
                mon = ctx_or_stim["monitor"]
            elif "refresh_rate" in ctx_or_stim:  # direct call on stim_log
                mon = ctx_or_stim
        if mon is None and isinstance(ctx, dict) and "monitor" in ctx:
            mon = ctx["monitor"]
        if not isinstance(mon, dict):
            raise KeyError("monitor info not found in ctx or stim_log")

        for k in ("refresh_rate","refreshRate","RefreshRate","rr","fps"):
            if k in mon and mon[k]:
                return float(mon[k])
        raise KeyError("monitor.refresh_rate not found")

    def _normalize_frames_unique(frames_unique):
        """Ensure each row is (is_display:int, image_index:int|None, indicator_val:float)."""
        fu = []
        for fr in frames_unique:
            if isinstance(fr, (list, tuple)) and len(fr) >= 3:
                is_disp = int(fr[0])
                img_idx = None if fr[1] is None else int(fr[1])
                ind_val = float(fr[2])
            else:
                fr = list(fr)
                is_disp = int(fr[0])
                img_idx = None if fr[1] is None else int(fr[1])
                ind_val = float(fr[2])
            fu.append((is_disp, img_idx, ind_val))
        return fu

    # ---------- get data ----------
    rr = _get_refresh_rate(ctx)
    dt = 1.0 / rr

    frames_unique = (
        stim_log.get("frames_unique")
        or stim_log.get("_frames_unique_compact")
        or stim_log.get("frames")
    )
    index_to_display = (
        stim_log.get("index_to_display")
        or stim_log.get("frame_config")
        or stim_log.get("index")
    )
    if frames_unique is None or index_to_display is None:
        raise KeyError(
            f"Missing frames_unique/index_to_display in stim_log. Keys: {list(stim_log.keys())}"
        )

    fu = _normalize_frames_unique(frames_unique)
    idx = np.asarray(index_to_display, dtype=int)
    n_total = idx.size

    # ---------- compress consecutive identical frame indices ----------
    runs = []
    if n_total > 0:
        start = 0
        cur = idx[0]
        for i in range(1, n_total):
            if idx[i] != cur:
                runs.append((int(cur), start, i - 1))
                start = i
                cur = idx[i]
        runs.append((int(cur), start, n_total - 1))

    rows = []
    for run_id, (uix, s, e) in enumerate(runs, start=1):
        if not (0 <= uix < len(fu)):
            raise IndexError(f"index_to_display refers to unknown frames_unique index {uix}")
        is_display, img_index, ind_val = fu[uix]
        nfrm = (e - s + 1)
        kind = "gap"
        if is_display == 1 and (img_index is None or img_index < 0):
            kind = "blank"
        elif is_display == 1 and (img_index is not None and img_index >= 0):
            kind = "display"
        rows.append({
            "run_idx": run_id,
            "kind": kind,
            "unique_index": uix,
            "image_index": ("" if img_index is None else img_index),
            "start_frame": s,
            "end_frame": e,
            "n_frames": nfrm,
            "onset_s": s * dt,
            "offset_s": (e + 1) * dt,   # inclusive end frame -> add 1
            "duration_s": nfrm * dt,
        })

    df_all = pd.DataFrame(rows)

    # ---------- select only randomized display runs ----------
    df_disp = df_all[df_all["kind"] == "display"].copy().reset_index(drop=True)
    df_disp.insert(0, "presentation_idx", np.arange(1, len(df_disp) + 1))

    # ---------- infer iteration structure if possible ----------
    iteration = stim_log.get("iteration", None)
    if iteration is not None:
        try:
            iteration = int(iteration)
            per_iter = len(df_disp) // max(iteration, 1)
            if per_iter > 0:
                iter_labels = np.repeat(np.arange(1, iteration + 1), per_iter)
                df_disp["iteration"] = iter_labels[: len(df_disp)]
                df_disp["within_iteration_idx"] = (
                    df_disp.groupby("iteration").cumcount() + 1
                )
        except Exception:
            pass

    # ---------- attach useful global fields ----------
    globals_like = (
        "display_dur","midgap_dur","pregap_dur","postgap_dur",
        "iteration","is_blank_block","coordinate","background"
    )
    for k in globals_like:
        if k in stim_log and k not in df_disp.columns:
            val = stim_log[k]
            if isinstance(val, list):
                val = tuple(val)
            df_disp[k] = [val] * len(df_disp)

    return df_disp



def extract_StimulusSeparator(stim_key, stim_log, ctx) -> pd.DataFrame:
    return pd.DataFrame([{"note": "IMPLEMENT extract_StimulusSeparator"}])

# Map class name -> extractor function
EXTRACTOR_REGISTRY = {
    "DriftingGratingMultipleCircle": extract_DriftingGratingMultipleCircle,
    "RandomizedUniformFlashes":      extract_RandomizedUniformFlashes,
    "DriftingGratingCircle":         extract_DriftingGratingCircle,
    "StaticGratingCircle":           extract_StaticGratingCircle,
    "StaticImages":                  extract_StaticImages,
    "StimulusSeparator":             extract_StimulusSeparator,    #not implemented
}

def default_extractor(stim_key, stim_log, ctx) -> pd.DataFrame:
    """Fallback: returns shallow metadata so you can see what was available."""
    if isinstance(stim_log, Mapping):
        meta = {k: v for k, v in stim_log.items()
                if isinstance(v, (str,int,float,bool)) or v is None}
        return pd.DataFrame([meta]) if meta else pd.DataFrame()
    return pd.DataFrame()

# ---------- Main ----------
def main(pkl_path: str, out_dir: str = "stim_schedules_csv", merge_csv: bool = False):
    data = load_pickle(pkl_path)
    logs = get_logs_container(data)
    if not logs:
        raise RuntimeError("Could not find per-stimulus logs (expected data['stimulation']['individual_logs']).")

    ctx = {
        "top": data,
        "monitor": data.get("monitor") if isinstance(data, dict) else None,
        "indicator": data.get("indicator") if isinstance(data, dict) else None,
        "presentation": data.get("presentation") if isinstance(data, dict) else None,
    }

    outdir = pathlib.Path(out_dir)
    outdir.mkdir(exist_ok=True)
    written = []
    all_rows = []

    # Iterate only over what exists
    for stim_key, stim_log in sorted(logs.items()):
        stim_class = class_of_stim_key(stim_key)
        extractor = EXTRACTOR_REGISTRY.get(stim_class, default_extractor)

        try:
            df = extractor(stim_key, stim_log, ctx)  # <-- YOUR logic
        except Exception as e:
            # Fail soft: capture error info
            df = pd.DataFrame([{"_extract_error": f"{type(e).__name__}: {e}"}])

        # Identity + CSV-friendly
        if df is None or df.empty:
            # Leave an empty file to signal "no rows extracted"?
            # (You can choose to skip instead.)
            df = pd.DataFrame([{"_note": "extractor returned empty"}])

        df = ensure_identity_cols(df, stim_key, stim_class)
        df = make_csv_friendly(df)

        out_path = outdir / f"{stim_key}_schedule.csv"
        df.to_csv(out_path, index=False)
        written.append(out_path)
        if merge_csv:
            all_rows.append(df)

    if merge_csv and all_rows:
        merged = pd.concat(all_rows, ignore_index=True)
        merged_path = outdir / "ALL_stim_schedules_merged.csv"
        merged.to_csv(merged_path, index=False)
        written.append(merged_path)

    print("Wrote:")
    for p in written:
        print(" -", p)

if __name__ == "__main__":
    # Usage:
    #   python extract_stim_schedules_backbone.py path/to/log.pkl [out_dir] [--merge]
    if len(sys.argv) < 2:
        print("Usage: python extract_stim_schedules_backbone.py /path/to/log.pkl [out_dir] [--merge]")
        sys.exit(1)
    pkl = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 and not sys.argv[2].startswith('-') else "stim_schedules_csv"
    merge = ("--merge" in sys.argv)
    main(pkl, out_dir, merge_csv=merge)
