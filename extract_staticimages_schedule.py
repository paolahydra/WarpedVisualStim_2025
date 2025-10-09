# extract_staticimages_schedule.py
import os, sys, pickle
import numpy as np
import pandas as pd

def alias_numpy_core_for_pickle_compat():
    """Allow unpickling logs written with NumPy 2.x."""
    try:
        import numpy.core as np_core
        sys.modules.setdefault('numpy._core', np_core)
    except Exception:
        pass

def load_pickle(path):
    alias_numpy_core_for_pickle_compat()
    with open(path, "rb") as f:
        return pickle.load(f)

def get_refresh_rate(mon):
    for k in ("refresh_rate","refreshRate","RefreshRate","rr","fps"):
        if isinstance(mon, dict) and k in mon and mon[k]:
            return float(mon[k])
    raise KeyError("monitor.refresh_rate not found in log['monitor']")

def find_dict(d, key):
    if not isinstance(d, dict):
        return None
    if key in d:
        return d[key]
    aliases = {
        "stimulation": ["stimulation","stim","stimulus","stim_dict"],
        "monitor": ["monitor","mon","screen"],
        "indicator": ["indicator","sync","photodiode"]
    }
    for k in aliases.get(key,[key]):
        if k in d:
            return d[k]
    return None

def extract_schedule_from_log(log: dict):
    stim = find_dict(log, "stimulation")
    mon  = find_dict(log, "monitor")
    if stim is None or mon is None:
        raise KeyError(f"Log missing 'stimulation' or 'monitor'. Top-level keys: {list(log.keys())}")

    rr = get_refresh_rate(mon)
    dt = 1.0 / rr

    frames_unique = stim.get("frames_unique", stim.get("frames", None))
    index_to_display = stim.get("index_to_display", stim.get("index", None))
    if frames_unique is None or index_to_display is None:
        raise KeyError(f"Missing frames_unique/index_to_display in stimulation. Keys: {list(stim.keys())}")

    # Normalize frames_unique to list of (is_display:int, image_index:int|None, indicator_val:float)
    fu = []
    for fr in frames_unique:
        if isinstance(fr, (list, tuple)) and len(fr) >= 3:
            is_disp = int(fr[0])
            img_idx = None if fr[1] is None else int(fr[1])
            ind_val = float(fr[2])
            fu.append((is_disp, img_idx, ind_val))
        else:
            # e.g., numpy object array row
            fr = list(fr)
            is_disp = int(fr[0])
            img_idx = None if fr[1] is None else int(fr[1])
            ind_val = float(fr[2])
            fu.append((is_disp, img_idx, ind_val))

    idx = np.asarray(index_to_display, dtype=int)
    n_total = idx.size

    # Compress consecutive frames with same unique index
    runs = []
    if n_total > 0:
        start = 0
        cur = idx[0]
        for i in range(1, n_total):
            if idx[i] != cur:
                runs.append((int(cur), start, i-1))
                start = i
                cur = idx[i]
        runs.append((int(cur), start, n_total-1))

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

    # Only real presentations: this is the randomized order
    df_disp = df_all[df_all["kind"] == "display"].copy().reset_index(drop=True)
    df_disp.insert(0, "presentation_idx", np.arange(1, len(df_disp)+1))

    # Try to infer iteration structure (optional)
    iteration = stim.get("iteration", None)
    if iteration is not None:
        try:
            iteration = int(iteration)
            per_iter = len(df_disp) // max(iteration, 1)
            if per_iter > 0:
                iter_labels = np.repeat(np.arange(1, iteration+1), per_iter)
                df_disp["iteration"] = iter_labels[:len(df_disp)]
                df_disp["within_iteration_idx"] = df_disp.groupby("iteration").cumcount() + 1
        except Exception:
            pass

    meta = {
        "refresh_rate": rr,
        "display_dur": stim.get("display_dur", None),
        "midgap_dur":  stim.get("midgap_dur", None),
        "pregap_dur":  stim.get("pregap_dur", None),
        "postgap_dur": stim.get("postgap_dur", None),
        "iteration":   stim.get("iteration", None),
        "is_blank_block": stim.get("is_blank_block", None),
    }
    return df_all, df_disp, meta

def main(pkl_path):
    obj = load_pickle(pkl_path)

    # The file may contain just the log dict, or a tuple/list (mov, log)
    log = None
    if isinstance(obj, dict):
        log = obj
    elif isinstance(obj, (list, tuple)):
        # try to find the dict that has 'stimulation'
        for x in obj:
            if isinstance(x, dict) and find_dict(x, "stimulation") is not None:
                log = x
                break
        if log is None and len(obj) >= 2 and isinstance(obj[1], dict):
            log = obj[1]
    else:
        raise TypeError(f"Unsupported pickle top-level type: {type(obj)}")

    df_all, df_disp, meta = extract_schedule_from_log(log)

    base = os.path.splitext(os.path.basename(pkl_path))[0]
    out_dir = os.path.dirname(pkl_path)
    csv_all  = os.path.join(out_dir, f"{base}_schedule_all_runs.csv")
    csv_disp = os.path.join(out_dir, f"{base}_schedule_displays.csv")
    df_all.to_csv(csv_all, index=False)
    df_disp.to_csv(csv_disp, index=False)

    # Small console summary
    print("Wrote:")
    print("  ", csv_all)
    print("  ", csv_disp)
    print("Meta:", meta)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_staticimages_schedule.py <path_to_log.pkl>")
        sys.exit(1)
    main(sys.argv[1])
