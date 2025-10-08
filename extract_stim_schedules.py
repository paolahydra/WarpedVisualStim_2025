# extract_stim_schedules.py
import sys, pickle, pathlib, sys
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
    class SafeUnpickler(pickle.Unpickler):
        ALLOWED = {
            'builtins': {
                'dict','list','tuple','set','frozenset','str','int','float','bool','bytes','bytearray',
                'complex','range','slice'
            },
            'collections': {'OrderedDict'}
        }
        def find_class(self, module, name):
            if module in self.ALLOWED and name in self.ALLOWED[module]:
                return getattr(__import__(module, fromlist=[name]), name)
            # fallback stub; we want dict/list payloads, not classes
            class Stub:
                def __init__(self, *a, **k): pass
                def __repr__(self): return f"<{module}.{name} (stub)>"
            return Stub
    with open(pkl_path, 'rb') as f:
        return SafeUnpickler(f).load()

def get_logs_container(data):
    if isinstance(data, dict):
        for k in ("individual_logs","logs","log"):
            if k in data and isinstance(data[k], dict):
                return data[k]
        # sometimes stimuli are at top-level dict already
        return {k:v for k,v in data.items() if isinstance(k,str) and any(
            s in k for s in ("Drifting","Static","Randomized","Images","Separator"))}
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

def main(pkl_path):
    data = load_pickle(pkl_path)
    logs = get_logs_container(data)
    if not logs:
        raise RuntimeError("Could not find per-stimulus logs in the pickle.")
    out_dir = pathlib.Path("stim_schedules_csv")
    out_dir.mkdir(exist_ok=True)
    written = []
    # logs may be keyed with numeric prefixes; preserve order if available
    for stim_key in sorted(logs.keys()):
        stim_log = logs[stim_key]
        df = extract_one(stim_key, stim_log)
        if df.empty:
            df = enrich_with_known_fields(df, stim_log)
        # carry best-effort timing columns to front if present
        cols = list(df.columns)
        time_cols = [c for c in TIMING_GUESS if c in cols]
        other_cols = [c for c in cols if c not in time_cols]
        df = df[time_cols + other_cols] if time_cols else df
        # write
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
