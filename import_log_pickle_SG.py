import pickle
import numpy as np

def _cond_params_from_frames_unique(frames_unique):
    n_cond = (len(frames_unique) - 1) // 2
    return {c: tuple(frames_unique[2*c + 1][1:6]) for c in range(n_cond)}

def build_presentation_table(log):
    st = log['stimulation']; fr = log['monitor']['refresh_rate']
    frames_unique = st['frames_unique_compact']
    idx = np.asarray(st['index_to_display'], dtype=int)

    is_gap = (idx == 0)
    is_on  = (~is_gap) & (idx % 2 == 1)              # ON frames are odd indices
    # Onset if: frame is ON and previous index is different (or this is the first frame)
    onset_mask = is_on & np.r_[True, idx[1:] != idx[:-1]]
    onset_frames = np.flatnonzero(onset_mask)

    cond_ids = ((idx[onset_frames] - 1) // 2).astype(int)
    cond_params = _cond_params_from_frames_unique(frames_unique)

    table = [
        dict(
            onset_frame=int(f),
            onset_time_s=float(f / fr),
            params=cond_params[int(c)]  # (sf, ph, ori, con, radius)
        )
        for f, c in zip(onset_frames, cond_ids)
    ]
    return table



######### ---------------------------------


with open(r'D:\data\displaySequence\visual_display_log\251006191040-StaticGratingCircle-MMOUSE-USER-TEST-notTriggered-incomplete.pkl', "rb") as f:
    #mov, log = pickle.load(f)   # mov: np.ndarray, log: dict
    log = pickle.load(f)

# quick sanity check
assert isinstance(log, dict) and 'stimulation' in log and 'monitor' in log

presentations = build_presentation_table(log)


import pandas as pd

# convert to DataFrame
df = pd.DataFrame([
    dict(
        onset_frame=p["onset_frame"],
        onset_time_s=p["onset_time_s"],
        sf=p["params"][0],
        ph=p["params"][1],
        ori=p["params"][2],
        con=p["params"][3],
        radius=p["params"][4],
    )
    for p in presentations
])

csv_path = r'D:\data\displaySequence\visual_display_log\251006191040-StaticGratingCircle-MMOUSE-USER-TEST-notTriggered-incomplete.csv'
df.to_csv(csv_path, index=False, float_format="%.6f")

print(f"Saved {len(df)} presentations to {csv_path}")


