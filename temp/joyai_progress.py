import os

import pandas as pd


pids = [48212, 48213, 48214, 48215, 48216, 48217, 48218, 48219]
frame = pd.read_parquet("/data/kaichen/data/MatchTime/manifests/valid.parquet")
paths = {path: index for index, path in enumerate(frame.video_path)}

for rank, pid in enumerate(pids):
    current = None
    for fd in os.listdir(f"/proc/{pid}/fd"):
        try:
            target = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if target in paths:
            current = paths[target]
            break
    assigned = list(range(rank, len(frame), len(pids)))
    completed = assigned.index(current) if current in assigned else None
    print(rank, current, completed, len(assigned))
