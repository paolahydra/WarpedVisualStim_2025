# extract_stim_schedules.py
import sys, pickle, pathlib
from collections import OrderedDict
import pandas as pd

# --- Small compatibility shim for older pickles that reference numpy._core ---
try:
    import numpy as np
    sys.modules.setdefault('numpy._core', np.core)
except Exception:
    pass

PRESENTATION_KEYS = ("presentations","trials","trial_table")
CONDITION_KEYS     = ("all_conditions","conditions","param_grid")
SEQUENCE_KEYS      = ("sequence","schedule","order","cond_indices","trial_indices")
TIMING_GUESS       = ("t_start","t_stop","t_on","t_off","start_time","stop_time","timestamp","frame_index")

# Field normalization per class (non-fatal: missing keys are ignored)
RENAMES = {
    "DriftingGratingMultipleCircle": {
        "ori":"ori_deg", "dire":"ori_motion", "sf":"sf_cpd", "tf":"tf_hz",
        "con":"contrast", "radius":"radius_deg", "phase":"phase_deg"
    },
    "DriftingGratingCircle": {
        "ori":"ori_deg", "dire":"ori_motion", "sf":"sf_cpd", "tf":"tf_hz",
        "con":"contrast", "radius":"radius_deg", "phase":"phase_deg"
    },
    "StaticGratingCircle": {
        "ori":"ori_deg", "sf":"sf_cpd", "con":"contrast", "radius":"radius_deg",
        "phase":"phase_deg"
    },
    "RandomizedUniformFlashes": {
        # names vary; leave mostly as-is but map obvious ones
        "flash_dur":"flash_dur_s", "midgap_dur":"midgap_dur_s", "block_dur":"block_dur_s",
        "intensity":"intensity_rel", "luminance":"luminance_rel"
    },
    "StaticImages": {
        "image_id":"image_id", "image_path":"image_path", "duration":"duration_s"
    },
    "StimulusSeparator": {
        "duration":"duration_s"
    }
}

def load_pickle(pkl_path):
    # 1) Plain pickle
    try:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e1:
        err1 = e1

    # 2) Help NumPy alias mismatches sometimes seen in older pickles
    try:
        import numpy as np
        sys.modules.setdefault('numpy._core', np.core)
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e2:
        err2 = e2

    # 3) pandas can be more forgiving (handles some compat cases)
    try:
        return pd.read_pickle(pkl_path)
    except Exception as e3:
        err3 = e3

    # 4) dill fallback (pip install dill)
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

def get_logs_container(data):
    """
    Return the mapping of {stim_name: stim_log} for this file layout.
    For your file, logs are under data['stimulation']['individual_logs'].
    Falls back to prior heuristics if that path isn't present.
    """
    from collections.abc import Mapping
    # Preferred (observed) path:
    if isinstance(data, dict):
        stim = data.get('stimulation', {})
        if isinstance(stim, Mapping):
            ilogs = stim.get('individual_logs', None)
            if isinstance(ilogs, Mapping) and ilogs:
                return ilogs

    # Fallbacks (keep your earlier heuristics if you want robustness across files)
    if isinstance(data, Mapping):
        for k in ("individual_logs","logs","log"):
            v = data.get(k)
            if isinstance(v, Mapping) and v:
                return v
    return None

def find_sequence(d):
    for k in SEQUENCE_KEYS:
        if k in d and isinstance(d[k], (list,tuple)):
            return d[k]
    # some logs store blocks/timeline with per-entry 'cond_idx'
    for k in ("blocks","timeline"):
        if k in d and isinstance(d[k], (list,tuple)) and d[k] and isinstance(d[k][0], dict) and "cond_idx" in d[k][0]:
            return [b["cond_idx"] for b in d[k]]
    return None

def normalize_row(row):
    # make lists hashable/CSV-friendly; keep scalars as-is
    out = {}
    for k,v in row.items():
        if isinstance(v, list):
            out[k] = tuple(v)
        elif isinstance(v, dict):
            # shallow stringify dicts to avoid wide columns
            out[k] = repr(v)
        else:
            out[k] = v
    return out

def rename_fields(df, class_name):
    mapping = RENAMES.get(class_name, {})
    keep = {c: mapping.get(c, c) for c in df.columns}
    return df.rename(columns=keep)

def df_from_presentations(pres):
    # pres: list[dict] or list of simple objects
    if not pres:
        return pd.DataFrame()
    if isinstance(pres[0], dict):
        rows = [normalize_row(p) for p in pres]
        return pd.DataFrame(rows)
    # fallback: wrap scalar list
    return pd.DataFrame({"value": pres})

def df_from_conditions(cond_list, seq):
    rows = []
    for i, idx in enumerate(seq, start=1):
        c = cond_list[int(idx)]
        row = {"trial": i, "cond_index": int(idx)}
        if isinstance(c, dict):
            row.update(c)
        else:
            row["condition"] = c
        rows.append(normalize_row(row))
    return pd.DataFrame(rows)

def enrich_with_known_fields(df, stim_log):
    # if df is empty, try to attach canonical lists for traceability
    added = False
    for k in ("sf_list","tf_list","ori_list","dire_list","con_list","radius_list","phase_list",
              "center","block_dur","midgap_dur","iteration","is_blank_block","is_random_start_phase"):
        if isinstance(stim_log, dict) and k in stim_log and k not in df.columns:
            df[k] = [stim_log[k]]
            added = True
    return df

def extract_one(stim_name, stim_log):
    # 1) direct presentations
    for key in PRESENTATION_KEYS:
        if isinstance(stim_log, dict) and key in stim_log and isinstance(stim_log[key], (list,tuple)):
            df = df_from_presentations(list(stim_log[key]))
            return rename_fields(df, stim_name.split("_",1)[-1] if "_" in stim_name else stim_name)
    # 2) conditions + sequence
    conds = None
    for key in CONDITION_KEYS:
        if isinstance(stim_log, dict) and key in stim_log and isinstance(stim_log[key], (list,tuple)):
            conds = list(stim_log[key]); break
    if conds is not None:
        seq = find_sequence(stim_log)
        if seq is not None:
            df = df_from_conditions(conds, seq)
            return rename_fields(df, stim_name.split("_",1)[-1] if "_" in stim_name else stim_name)
        # fallback: just the grid
        df = pd.DataFrame([normalize_row(c) if isinstance(c,dict) else {"condition":c} for c in conds])
        return rename_fields(df, stim_name.split("_",1)[-1] if "_" in stim_name else stim_name)
    # 3) last resort: shallow flatten
    if isinstance(stim_log, dict):
        flat = {k:v for k,v in stim_log.items()
                if isinstance(v,(str,int,float,bool)) or v is None}
        if flat:
            return pd.DataFrame([normalize_row(flat)])
    return pd.DataFrame()

import pandas as pd

def df_from_list_of_dicts(lst):
    if not lst:
        return pd.DataFrame()
    if isinstance(lst[0], dict):
        return pd.DataFrame(lst)
    return pd.DataFrame({"value": lst})

def extract_grating_trials(log_dict):
    """
    Handles DriftingGratingMultipleCircle / DriftingGratingCircle.
    Prefers fully expanded per-trial lists, then key->param mapping, then shuffled conditions.
    """
    # 1) Already expanded per-trial params
    for k in ("presentations","trials","trial_table","condition_orders_params"):
        if k in log_dict and isinstance(log_dict[k], (list,tuple)) and log_dict[k]:
            df = df_from_list_of_dicts(list(log_dict[k]))
            df.insert(0, "trial", range(1, len(df)+1))
            return df

    # 2) Keys -> params mapping
    if ("condition_orders_keys" in log_dict and
        "condi_key_to_params" in log_dict and
        isinstance(log_dict["condition_orders_keys"], (list,tuple)) and
        isinstance(log_dict["condi_key_to_params"], dict)):
        rows = []
        for i, key in enumerate(log_dict["condition_orders_keys"], start=1):
            params = log_dict["condi_key_to_params"].get(key, {})
            row = {"trial": i, "cond_key": key}
            if isinstance(params, dict):
                row.update(params)
            rows.append(row)
        return pd.DataFrame(rows)

    # 3) Shuffled full conditions (already randomized)
    if "all_conditions_shuffled" in log_dict and isinstance(log_dict["all_conditions_shuffled"], (list,tuple)):
        rows = []
        for i, params in enumerate(log_dict["all_conditions_shuffled"], start=1):
            row = {"trial": i}
            if isinstance(params, dict):
                row.update(params)
            else:
                row["condition"] = params
            rows.append(row)
        return pd.DataFrame(rows)

    # 4) At least return the unique condition grid
    if "all_conditions" in log_dict and isinstance(log_dict["all_conditions"], (list,tuple)):
        df = df_from_list_of_dicts(list(log_dict["all_conditions"]))
        df.insert(0, "grid_index", range(len(df)))
        return df

    return pd.DataFrame()

def extract_static_grating_trials(log_dict):
    """
    Handles StaticGratingCircle.
    Typical layout shows:
      - iteration, display_dur, midgap_dur, is_blank_block
      - _all_conditions: list of dicts describing parameter combos
      - _frames_unique_compact: unique frames (parameter tuples)
      - frame_config: sequence of indices (or dicts) referring to unique frames
    Strategy:
      1) If 'presentations' exists, use it.
      2) If 'frame_config' + '_frames_unique_compact' exist, map indices -> params, build one row per frame.
         If 'iteration' > 1, frame_config likely already encodes the repeated order; we just expand as given.
      3) Else, fall back to '_all_conditions'.
    """
    # 1) Already expanded
    for k in ("presentations","trials","trial_table"):
        if k in log_dict and isinstance(log_dict[k], (list,tuple)) and log_dict[k]:
            df = df_from_list_of_dicts(list(log_dict[k]))
            df.insert(0, "row", range(1, len(df)+1))
            return df

    # 2) frame_config + _frames_unique_compact
    fc = log_dict.get("frame_config", None)
    uniq = log_dict.get("_frames_unique_compact", None)
    if isinstance(fc, (list,tuple)) and isinstance(uniq, (list,tuple)) and len(uniq) > 0:
        rows = []
        for i, idx in enumerate(fc, start=1):
            # idx may be integer or dict; handle both
            if isinstance(idx, dict) and "frame_idx" in idx:
                uix = idx["frame_idx"]
            else:
                uix = idx
            params = {}
            if isinstance(uix, int) and 0 <= uix < len(uniq):
                u = uniq[uix]
                if isinstance(u, dict):
                    params.update(u)
                else:
                    params["frame_value"] = u
            row = {"frame_row": i, "unique_frame_index": uix}
            row.update(params)
            rows.append(row)
        df = pd.DataFrame(rows)
        # Attach useful global fields if present
        for k in ("display_dur","midgap_dur","iteration","is_blank_block","coordinate","background","center","sf_list","phase_list","ori_list","con_list","radius_list"):
            if k in log_dict and k not in df.columns:
                df[k] = log_dict[k]
        return df

    # 3) _all_conditions as fallback
    if "_all_conditions" in log_dict and isinstance(log_dict["_all_conditions"], (list,tuple)):
        df = df_from_list_of_dicts(list(log_dict["_all_conditions"]))
        df.insert(0, "grid_index", range(len(df)))
        for k in ("display_dur","midgap_dur","iteration","is_blank_block"):
            if k in log_dict and k not in df.columns:
                df[k] = log_dict[k]
        return df

    return pd.DataFrame()


def main(pkl_path):
    data = load_pickle(pkl_path)
    logs = get_logs_container(data)
    if not logs:
        raise RuntimeError("Could not find per-stimulus logs in the pickle.")
    out_dir = pathlib.Path("stim_schedules_csv")
    out_dir.mkdir(exist_ok=True)
    written = []
    # logs may be keyed with numeric prefixes; preserve order if available
    for stim_key, stim_log in sorted(logs.items()):
        # Normalize class label (strip numeric prefix if present)
        cls = stim_key.split("_", 1)[-1] if "_" in stim_key else stim_key

        if cls in ("DriftingGratingMultipleCircle","DriftingGratingCircle"):
            df = extract_grating_trials(stim_log)
        elif cls == "StaticGratingCircle":
            df = extract_static_grating_trials(stim_log)
        elif cls in ("RandomizedUniformFlashes","StaticImages","StimulusSeparator"):
            # Reuse generic path: try 'presentations', then grid+sequence if present
            df = extract_grating_trials(stim_log)  # covers presentations / all_conditions_shuffled / keys->params
            if df.empty and "_all_conditions" in stim_log:
                df = df_from_list_of_dicts(list(stim_log["_all_conditions"]))
                df.insert(0, "grid_index", range(len(df)))
        else:
            # Generic fallback
            df = extract_grating_trials(stim_log)

        # If still empty, add trace fields to help debugging
        if df.empty and isinstance(stim_log, dict):
            meta = {k: v for k, v in stim_log.items() if isinstance(v, (str, int, float, bool)) or v is None}
            if meta:
                df = pd.DataFrame([meta])

        # Make lists CSV-friendly
        for c in df.columns:
            df[c] = df[c].apply(lambda x: tuple(x) if isinstance(x, list) else x)

        out_path = out_dir / f"{stim_key}_schedule.csv"
        df.to_csv(out_path, index=False)
        written.append(out_path)

    print("Wrote CSVs to:", out_dir)
    for p in written:
        print(" -", p)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_stim_schedules.py /path/to/log.pkl")
        sys.exit(1)
    main(sys.argv[1])
