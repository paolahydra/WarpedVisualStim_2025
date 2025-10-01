import pickle
from itertools import product

def load_log(pkl_path):
    """Load a log saved with pickle or joblib; return the top-level object."""
    try:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        # some projects save logs via joblib.dump; fall back
        try:
            from joblib import load as joblib_load
            return joblib_load(pkl_path)
        except Exception as e:
            raise RuntimeError(f"Could not load {pkl_path}: {e}")

def get_stimulation(obj):
    """
    Accepts various shapes:
    - dict with 'stimulation'
    - (mov, log) tuple/list where log is a dict with 'stimulation'
    Returns the stimulation dict.
    """
    if isinstance(obj, dict) and 'stimulation' in obj:
        return obj['stimulation']
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        maybe_log = obj[1]
        if isinstance(maybe_log, dict) and 'stimulation' in maybe_log:
            return maybe_log['stimulation']
    raise KeyError("No 'stimulation' found. Inspect the loaded object structure.")

def rebuild_all_conditions_if_missing(stim):
    """
    If your log predates the patch and lacks 'all_conditions',
    rebuild the cartesian set from the parameter lists in the log.
    (Order is NOT the realized presentation order.)
    """
    if 'all_conditions' in stim and stim['all_conditions']:
        return list(map(tuple, stim['all_conditions']))  # already present
    # fall back: build from parameter lists stored in the stim dict
    sf_list   = list(stim['sf_list'])
    tf_list   = list(stim['tf_list'])
    dire_list = list(stim['dire_list'])
    con_list  = list(stim['con_list'])
    radius_list = list(stim['radius_list'])
    return list(product(sf_list, tf_list, dire_list, con_list, radius_list))  # (sf, tf, dire, con, radius)

# --- usage ---
obj  = load_log(r'D:\data\displaySequence\visual_display_log\251001070359-DriftingGratingCircle-MTest-Name-000-notTriggered-complete.pkl')
stim = get_stimulation(obj)

# stim is your full metadata dict (center, lists, timing, flags, etc.)
print("Stim keys:", sorted(stim.keys()))

# 1) the set of conditions (unordered):
all_conditions = rebuild_all_conditions_if_missing(stim)

# 2) if you applied the patch I suggested earlier, you may also have:
orders_params = stim.get('condition_orders_params', None)  # per-iteration realized order (list of lists)
orders_keys   = stim.get('condition_orders_keys', None)    # same, but as condi_XXXX keys


import pandas as pd
df = pd.DataFrame(all_conditions, columns=['sf','tf','direction','contrast','radius'])
df.to_csv("/path/to/all_conditions.csv", index=False)

# If realized orders are present:
if orders_params:
    rows = []
    for it, conds in enumerate(orders_params):
        for pos, (sf, tf, dire, con, rad) in enumerate(conds):
            rows.append({'iteration': it, 'position': pos,
                         'sf': sf, 'tf': tf, 'direction': dire,
                         'contrast': con, 'radius': rad})
    pd.DataFrame(rows).to_csv("/path/to/condition_order.csv", index=False)




import pandas as pd
df = pd.DataFrame(all_conditions, columns=['sf','tf','direction','contrast','radius'])
df.to_csv('D:\data\displaySequence\visual_display_log\251001070359-DriftingGratingCircle-MTest-Name-000-notTriggered_all_conditions.csv', index=False)

# If realized orders are present:
if orders_params:
    rows = []
    for it, conds in enumerate(orders_params):
        for pos, (sf, tf, dire, con, rad) in enumerate(conds):
            rows.append({'iteration': it, 'position': pos,
                         'sf': sf, 'tf': tf, 'direction': dire,
                         'contrast': con, 'radius': rad})
    pd.DataFrame(rows).to_csv("/path/to/condition_order.csv", index=False)



