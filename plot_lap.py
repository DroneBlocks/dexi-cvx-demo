"""Plot a lap from oval_fly.py's --log file: the path, the flown track and the flight area."""
import json, sys
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from oval_fly import PATH, AREA_LO, AREA_HI, TOL

L = json.load(open(sys.argv[1])); P = np.array([s["pos"] for s in L["samples"]])
FT = 0.3048
fig, ax = plt.subplots(figsize=(4, 7))
lo, hi = AREA_LO + TOL, AREA_HI - TOL
ax.add_patch(plt.Rectangle((lo[1] / FT, lo[0] / FT), (hi - lo)[1] / FT, (hi - lo)[0] / FT, fill=False, ls="--", color="#888", label="15 x 5 ft area"))
ax.plot(PATH[:, 1] / FT, PATH[:, 0] / FT, color="#bbb", lw=6, label="path")
ax.plot(P[:, 1] / FT, P[:, 0] / FT, color="#1f6fd1", lw=1.5, label=sys.argv[3] if len(sys.argv) > 3 else "flown")
ax.plot(0, 0, "o", color="#2a9d55", label="takeoff / landing")
ax.set_aspect("equal"); ax.set_xlabel("ft (right of start)"); ax.set_ylabel("ft (forward)")
ax.set_xlim(-4, 4); ax.set_ylim(-1, 17); ax.legend(loc="center", fontsize=8, framealpha=0.9)
out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].rsplit(".", 1)[0] + ".png"
fig.tight_layout(); fig.savefig(out, dpi=150); print(out)
