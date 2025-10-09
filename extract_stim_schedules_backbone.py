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
    Return a per-trial schedule for DriftingGratingMultipleCircle.

    Priorities:
      1) 'condition_orders_params'  -> list of lists of params (per iteration)
      2) 'condition_orders_keys' + 'condi_key_to_params' -> map keys to params
      3) 'all_conditions_shuffled'  -> flat randomized list used in this run
      4) fallback cartesian grid from parameter lists (not randomized order)

    Output columns:
      ['trial','iteration','position','sf','tf','direction','contrast','radius','center', ...globals]
    """

    # --- helpers ---
    def _row_from_params(p):
        """Accepts a dict or a tuple/list (sf, tf, dire, con, radius, center)."""
        if isinstance(p, dict):
            # be tolerant to naming minor variants
            sf   = p.get('sf')
            tf   = p.get('tf')
            dire = p.get('dire', p.get('direction'))
            con  = p.get('con',  p.get('contrast'))
            rad  = p.get('radius', p.get('rad'))
            cen  = p.get('center', p.get('centre'))
        else:
            # assume ordered tuple/list
            sf, tf, dire, con, rad, cen = p
        return {
            'sf': sf, 'tf': tf, 'direction': dire, 'contrast': con,
            'radius': rad, 'center': tuple(cen) if isinstance(cen, (list, tuple)) else cen
        }

    def _attach_globals(df):
        """Repeat useful globals per row (safe broadcast)."""
        if df is None or df.empty:
            return df
        globals_like = ['block_dur','pregap_dur','postgap_dur','is_random_start_phase',
                        'coordinate','background','smooth_width_ratio','is_smooth_edge']
        for k in globals_like:
            if k in stim_log and k not in df.columns:
                val = stim_log[k]
                df[k] = [tuple(val) if isinstance(val, list) else val] * len(df)
        return df

    # --- 1) condition_orders_params (best) ---
    if isinstance(stim_log.get('condition_orders_params'), (list, tuple)) and stim_log['condition_orders_params']:
        rows = []
        trial = 1
        for it, conds in enumerate(stim_log['condition_orders_params']):
            # conds can be list of tuples or dicts
            for pos, p in enumerate(conds):
                r = {'trial': trial, 'iteration': it, 'position': pos}
                r.update(_row_from_params(p))
                rows.append(r)
                trial += 1
        df = pd.DataFrame(rows)
        return _attach_globals(df)

    # --- 2) condition_orders_keys + condi_key_to_params ---
    cok = stim_log.get('condition_orders_keys')
    ck2p = stim_log.get('condi_key_to_params')
    if isinstance(cok, (list, tuple)) and ck2p and isinstance(ck2p, dict):
        rows = []
        trial = 1
        # keys may be list of lists (per iteration) or a flat list
        if cok and isinstance(cok[0], (list, tuple)):
            for it, keys in enumerate(cok):
                for pos, key in enumerate(keys):
                    p = ck2p.get(key, {})
                    r = {'trial': trial, 'iteration': it, 'position': pos, 'cond_key': key}
                    r.update(_row_from_params(p))
                    rows.append(r)
                    trial += 1
        else:
            for pos, key in enumerate(cok):
                p = ck2p.get(key, {})
                r = {'trial': trial, 'iteration': 0, 'position': pos, 'cond_key': key}
                r.update(_row_from_params(p))
                rows.append(r)
                trial += 1
        df = pd.DataFrame(rows)
        return _attach_globals(df)

    # --- 3) all_conditions_shuffled (flat per-trial randomized list) ---
    if isinstance(stim_log.get('all_conditions_shuffled'), (list, tuple)) and stim_log['all_conditions_shuffled']:
        rows = []
        for pos, p in enumerate(stim_log['all_conditions_shuffled']):
            r = {'trial': pos+1, 'iteration': 0, 'position': pos}
            r.update(_row_from_params(p))
            rows.append(r)
        df = pd.DataFrame(rows)
        return _attach_globals(df)

    # --- 4) fallback: rebuild cartesian grid (NOT the realized order) ---
    # This at least gives you the full set of unique parameter tuples.
    sf_list     = list(stim_log.get('sf_list', []))
    tf_list     = list(stim_log.get('tf_list', []))
    dire_list   = list(stim_log.get('dire_list', []))
    con_list    = list(stim_log.get('con_list', []))
    radius_list = list(stim_log.get('radius_list', []))
    center_list = list(stim_log.get('center_list', []))
    if all(len(lst) > 0 for lst in (sf_list, tf_list, dire_list, con_list, radius_list, center_list)):
        rows = []
        for i, (sf, tf, dire, con, rad, cen) in enumerate(
            product(sf_list, tf_list, dire_list, con_list, radius_list, center_list), start=1
        ):
            rows.append({
                'trial': i, 'iteration': 0, 'position': i-1,
                'sf': sf, 'tf': tf, 'direction': dire, 'contrast': con,
                'radius': rad, 'center': tuple(cen) if isinstance(cen, (list, tuple)) else cen,
                '_note': 'cartesian_grid_fallback_not_realized_order'
            })
        df = pd.DataFrame(rows)
        return _attach_globals(df)

    # No recognized fields:
    return pd.DataFrame([{'_extract_error': 'DGMC: no recognizable schedule fields present'}])


def extract_DriftingGratingCircle(stim_key, stim_log, ctx):
    """
    Per-trial schedule for DriftingGratingCircle.

    Priority:
      1) 'condition_orders_params'            -> list[list[params]] (per iteration)
      2) 'condition_orders_keys' + 'condi_key_to_params'
      3) 'all_conditions_shuffled'            -> flat randomized list (this run)
      4) cartesian grid from lists (fallback; NOT realized order)

    Output columns:
      ['trial','iteration','position','sf','tf','direction','contrast','radius', ...globals]
    """

    # --- helpers ---
    def _row_from_params(p):
        """Accept a dict or tuple/list (sf, tf, dire, con, radius)."""
        if isinstance(p, dict):
            sf   = p.get('sf')
            tf   = p.get('tf')
            dire = p.get('dire', p.get('direction'))
            con  = p.get('con',  p.get('contrast'))
            rad  = p.get('radius', p.get('rad'))
        else:
            sf, tf, dire, con, rad = p
        return {'sf': sf, 'tf': tf, 'direction': dire, 'contrast': con, 'radius': rad}

    def _attach_globals(df):
        if df is None or df.empty:
            return df
        # Repeat useful globals per row
        globals_like = [
            'block_dur','pregap_dur','postgap_dur','is_random_start_phase',
            'coordinate','background','smooth_width_ratio','is_smooth_edge','center'
        ]
        for k in globals_like:
            if k in stim_log and k not in df.columns:
                val = stim_log[k]
                if isinstance(val, list):
                    val = tuple(val)
                df[k] = [val] * len(df)
        return df

    # --- 1) condition_orders_params (best) ---
    cop = stim_log.get('condition_orders_params')
    if isinstance(cop, (list, tuple)) and cop:
        rows, trial = [], 1
        for it, conds in enumerate(cop):
            for pos, p in enumerate(conds):
                r = {'trial': trial, 'iteration': it, 'position': pos}
                r.update(_row_from_params(p))
                rows.append(r)
                trial += 1
        df = pd.DataFrame(rows)
        return _attach_globals(df)

    # --- 2) condition_orders_keys + mapping ---
    cok  = stim_log.get('condition_orders_keys')
    ck2p = stim_log.get('condi_key_to_params')
    if isinstance(cok, (list, tuple)) and ck2p and isinstance(ck2p, dict):
        rows, trial = [], 1
        if cok and isinstance(cok[0], (list, tuple)):
            for it, keys in enumerate(cok):
                for pos, key in enumerate(keys):
                    p = ck2p.get(key, {})
                    r = {'trial': trial, 'iteration': it, 'position': pos, 'cond_key': key}
                    r.update(_row_from_params(p))
                    rows.append(r); trial += 1
        else:
            for pos, key in enumerate(cok):
                p = ck2p.get(key, {})
                r = {'trial': trial, 'iteration': 0, 'position': pos, 'cond_key': key}
                r.update(_row_from_params(p))
                rows.append(r); trial += 1
        df = pd.DataFrame(rows)
        return _attach_globals(df)

    # --- 3) all_conditions_shuffled (flat randomized list) ---
    acs = stim_log.get('all_conditions_shuffled')
    if isinstance(acs, (list, tuple)) and acs:
        rows = []
        for pos, p in enumerate(acs):
            r = {'trial': pos+1, 'iteration': 0, 'position': pos}
            r.update(_row_from_params(p))
            rows.append(r)
        df = pd.DataFrame(rows)
        return _attach_globals(df)

    # --- 4) cartesian grid fallback (NOT realized order) ---
    sf_list     = list(stim_log.get('sf_list', []))
    tf_list     = list(stim_log.get('tf_list', []))
    dire_list   = list(stim_log.get('dire_list', []))
    con_list    = list(stim_log.get('con_list', []))
    radius_list = list(stim_log.get('radius_list', []))
    if all(len(lst) > 0 for lst in (sf_list, tf_list, dire_list, con_list, radius_list)):
        rows = []
        for i, (sf, tf, dire, con, rad) in enumerate(
            product(sf_list, tf_list, dire_list, con_list, radius_list), start=1
        ):
            rows.append({
                'trial': i, 'iteration': 0, 'position': i-1,
                'sf': sf, 'tf': tf, 'direction': dire, 'contrast': con, 'radius': rad,
                '_note': 'cartesian_grid_fallback_not_realized_order'
            })
        df = pd.DataFrame(rows)
        return _attach_globals(df)

    # No recognizable schedule fields
    return pd.DataFrame([{'_extract_error': 'DGC: no recognizable schedule fields present'}])


def extract_StaticGratingCircle(stim_key, stim_log, ctx):
    """
    Build a per-presentation table for StaticGratingCircle by detecting ON-frame onsets.
    Robust to:
      - frames list name: 'frames_unique_compact' or '_frames_unique_compact'
      - index sequence:    'index_to_display' or 'frame_config'
    Requires monitor refresh rate from ctx['monitor']['refresh_rate'].
    Output columns: onset_frame, onset_time_s, sf, ph, ori, con, radius
    """

    # ---- resolve frames_unique_compact ----
    frames_unique = None
    for key in ("frames_unique_compact", "_frames_unique_compact"):
        if key in stim_log and isinstance(stim_log[key], (list, tuple)):
            frames_unique = stim_log[key]
            break
    if frames_unique is None:
        return pd.DataFrame([{"_extract_error": "SGC: missing frames_unique_compact/_frames_unique_compact"}])

    # ---- resolve index_to_display (frame indices over time) ----
    if "index_to_display" in stim_log:
        idx = np.asarray(stim_log["index_to_display"], dtype=int)
    elif "frame_config" in stim_log:
        # frame_config can be a list of ints or list of dicts with 'frame_idx'
        fc = stim_log["frame_config"]
        if len(fc) > 0 and isinstance(fc[0], dict) and "frame_idx" in fc[0]:
            idx = np.asarray([int(d["frame_idx"]) for d in fc], dtype=int)
        else:
            idx = np.asarray(fc, dtype=int)
    else:
        return pd.DataFrame([{"_extract_error": "SGC: missing index_to_display/frame_config"}])

    # ---- monitor refresh rate ----
    fr = None
    try:
        mon = ctx.get("monitor") if isinstance(ctx, dict) else None
        if mon and "refresh_rate" in mon:
            fr = float(mon["refresh_rate"])
    except Exception:
        pass
    if not fr or fr <= 0:
        return pd.DataFrame([{"_extract_error": "SGC: missing/invalid monitor.refresh_rate"}])

    # ---- helper: map condition id -> (sf, ph, ori, con, radius) ----
    # frames_unique is typically: index 0 = GAP; then alternating ON/OFF pairs:
    # ON index = 2*c + 1, OFF index = 2*c + 2  (so cond_id = (idx-1)//2)
    n_cond = (len(frames_unique) - 1) // 2
    def cond_params(cid):
        # frames_unique[2*cid + 1] is the ON frame; element [1:6] holds (sf, ph, ori, con, radius)
        fr_tuple = frames_unique[2 * cid + 1]
        # tolerate either tuple/list with leading flag; keep [1:6]
        return tuple(fr_tuple[1:6])

    # ---- detect ON onsets ----
    is_gap = (idx == 0)
    is_on  = (~is_gap) & (idx % 2 == 1)
    onset_mask = is_on & np.r_[True, idx[1:] != idx[:-1]]  # first ON frame or change of ON index
    onset_frames = np.flatnonzero(onset_mask)
    if onset_frames.size == 0:
        return pd.DataFrame([{"_extract_error": "SGC: no ON onsets detected"}])

    # condition id for each onset
    cond_ids = ((idx[onset_frames] - 1) // 2).astype(int)
    # safety clamp
    cond_ids = np.clip(cond_ids, 0, max(n_cond - 1, 0))

    # ---- assemble rows ----
    rows = []
    for f, cid in zip(onset_frames, cond_ids):
        try:
            sf, ph, ori, con, rad = cond_params(int(cid))
        except Exception:
            rows.append({"_extract_error": f"SGC: bad cond id {int(cid)}", "onset_frame": int(f)})
            continue
        rows.append({
            "onset_frame": int(f),
            "onset_time_s": float(f / fr),
            "sf": sf, "ph": ph, "ori": ori, "con": con, "radius": rad
        })

    df = pd.DataFrame(rows)

    # ---- optional: attach useful globals repeated per row ----
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
    Per-presentation table for StaticImages.

    Priority:
      1) presentations
      2) condition_orders_params
      3) condition_orders_keys + condi_key_to_params
      4) all_conditions_shuffled
      5) frames_unique_compact/_frames_unique_compact + index_to_display/frame_config (ON onsets)

    Output columns (best-effort): trial, iteration, position, image_id, image_path, label, category,
                                  onset_frame, onset_time_s, plus selected globals.
    """
    import numpy as np
    import pandas as pd

    # ---- helpers ----
    def _take(d, *keys):
        """Return first present key from dict d."""
        for k in keys:
            if isinstance(d, dict) and k in d:
                return d[k]
        return None

    def _norm_image_row(p):
        """
        Accept dict or tuple/list; extract common image fields.
        Returns a dict with any of: image_id, image_path, label, category.
        """
        row = {}
        if isinstance(p, dict):
            for k in ("image_id", "img_id", "id"):
                v = p.get(k)
                if v is not None:
                    row["image_id"] = v
                    break
            for k in ("image_path", "path", "file", "filename", "filepath"):
                v = p.get(k)
                if v is not None:
                    row["image_path"] = v
                    break
            for k in ("label", "name", "title"):
                v = p.get(k)
                if v is not None:
                    row["label"] = v
                    break
            for k in ("category", "class"):
                v = p.get(k)
                if v is not None:
                    row["category"] = v
                    break
            # If nothing explicit, try a generic 'image'/'stim' field
            if not row.get("image_path") and not row.get("image_id"):
                v = _take(p, "image", "stim", "item")
                if isinstance(v, str):
                    row["image_path"] = v
        else:
            # tuple/list fallback: assume first string-looking element is path/id
            if isinstance(p, (list, tuple)):
                for el in p:
                    if isinstance(el, str):
                        # crude heuristic: path if contains a separator or dot, else id
                        if ("/" in el or "\\" in el or "." in el) and "image_path" not in row:
                            row["image_path"] = el
                        elif "image_id" not in row:
                            row["image_id"] = el
        return row

    def _attach_globals(df):
        if df is None or df.empty:
            return df
        globals_like = ("display_dur", "midgap_dur", "iteration", "is_blank_block",
                        "coordinate", "background", "center")
        for k in globals_like:
            if k in stim_log and k not in df.columns:
                val = stim_log[k]
                if isinstance(val, list):
                    val = tuple(val)
                df[k] = [val] * len(df)
        return df

    # ---- 1) presentations ----
    pres = stim_log.get("presentations")
    if isinstance(pres, (list, tuple)) and pres:
        rows = []
        for i, p in enumerate(pres, start=1):
            r = {"trial": i, "iteration": 0, "position": i-1}
            if isinstance(p, dict):
                r.update(_norm_image_row(p))
                # carry timing if present
                for tk in ("t_start", "t_stop", "t_on", "t_off", "start_time", "stop_time", "onset_frame", "onset_time_s"):
                    if tk in p:
                        r[tk] = p[tk]
            else:
                r.update(_norm_image_row(p))
            rows.append(r)
        df = pd.DataFrame(rows)
        return _attach_globals(df)

    # ---- 2) condition_orders_params ----
    cop = stim_log.get("condition_orders_params")
    if isinstance(cop, (list, tuple)) and cop:
        rows = []
        trial = 1
        for it, conds in enumerate(cop):
            for pos, p in enumerate(conds):
                r = {"trial": trial, "iteration": it, "position": pos}
                r.update(_norm_image_row(p))
                rows.append(r); trial += 1
        df = pd.DataFrame(rows)
        return _attach_globals(df)

    # ---- 3) condition_orders_keys + mapping ----
    cok  = stim_log.get("condition_orders_keys")
    ck2p = stim_log.get("condi_key_to_params")
    if isinstance(cok, (list, tuple)) and ck2p and isinstance(ck2p, dict):
        rows = []
        trial = 1
        if cok and isinstance(cok[0], (list, tuple)):
            for it, keys in enumerate(cok):
                for pos, key in enumerate(keys):
                    p = ck2p.get(key, {})
                    r = {"trial": trial, "iteration": it, "position": pos, "cond_key": key}
                    r.update(_norm_image_row(p))
                    rows.append(r); trial += 1
        else:
            for pos, key in enumerate(cok):
                p = ck2p.get(key, {})
                r = {"trial": trial, "iteration": 0, "position": pos, "cond_key": key}
                r.update(_norm_image_row(p))
                rows.append(r); trial += 1
        df = pd.DataFrame(rows)
        return _attach_globals(df)

    # ---- 4) all_conditions_shuffled ----
    acs = stim_log.get("all_conditions_shuffled")
    if isinstance(acs, (list, tuple)) and acs:
        rows = []
        for pos, p in enumerate(acs):
            r = {"trial": pos+1, "iteration": 0, "position": pos}
            r.update(_norm_image_row(p))
            rows.append(r)
        df = pd.DataFrame(rows)
        return _attach_globals(df)

    # ---- 5) frames_unique_compact route (detect ON onsets) ----
    frames_unique = None
    for key in ("frames_unique_compact", "_frames_unique_compact"):
        if key in stim_log and isinstance(stim_log[key], (list, tuple)):
            frames_unique = stim_log[key]; break

    # index/time base
    idx = None
    if "index_to_display" in stim_log:
        idx = np.asarray(stim_log["index_to_display"], dtype=int)
    elif "frame_config" in stim_log:
        fc = stim_log["frame_config"]
        if len(fc) > 0 and isinstance(fc[0], dict) and "frame_idx" in fc[0]:
            idx = np.asarray([int(d["frame_idx"]) for d in fc], dtype=int)
        else:
            idx = np.asarray(fc, dtype=int)

    # monitor refresh
    fr = None
    try:
        mon = ctx.get("monitor") if isinstance(ctx, dict) else None
        if mon and "refresh_rate" in mon:
            fr = float(mon["refresh_rate"])
    except Exception:
        pass

    if frames_unique is not None and idx is not None and fr and fr > 0:
        # 0 = GAP; for c in [0..n_cond-1]: ON = 2*c+1, OFF = 2*c+2
        is_gap = (idx == 0)
        is_on  = (~is_gap) & (idx % 2 == 1)
        onset_mask = is_on & np.r_[True, idx[1:] != idx[:-1]]
        onset_frames = np.flatnonzero(onset_mask)

        def _cond_params_from_frames_unique(fr_u):
            n_cond = (len(fr_u) - 1) // 2
            # For images, the ON frame payload (fr_u[2*c+1]) may contain various things; try common fields/slots.
            def _img_from_on_frame(onf):
                # onf could be a tuple/list like (flag, image_id/path/obj, ...)
                # Heuristic: if dict at [1], use it; else take first string as image_path/id.
                if isinstance(onf, (list, tuple)) and len(onf) >= 2:
                    meta = onf[1]
                    if isinstance(meta, dict):
                        return _norm_image_row(meta)
                    if isinstance(meta, str):
                        if ("/" in meta or "\\" in meta or "." in meta):
                            return {"image_path": meta}
                        return {"image_id": meta}
                # if dict
                if isinstance(onf, dict):
                    return _norm_image_row(onf)
                return {}
            return {c: _img_from_on_frame(fr_u[2*c + 1]) for c in range(max(((len(fr_u)-1)//2), 0))}

        cond_map = _cond_params_from_frames_unique(frames_unique)
        cond_ids = ((idx[onset_frames] - 1) // 2).astype(int)

        rows = []
        for t, (f, cid) in enumerate(zip(onset_frames, cond_ids), start=1):
            im = cond_map.get(int(cid), {})
            r = {
                "trial": t, "iteration": 0, "position": t-1,
                "onset_frame": int(f), "onset_time_s": float(f / fr)
            }
            r.update(im)
            rows.append(r)
        df = pd.DataFrame(rows)
        return _attach_globals(df)

    # Nothing recognized
    return pd.DataFrame([{"_extract_error": "StaticImages: no recognizable schedule fields present"}])


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
