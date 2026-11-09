"""Fly the racetrack with no drone: each solve's first step becomes the next state. Prints solve times.
Run this on the companion computer to check it is fast enough before flying."""
import time, numpy as np, cvxpy as cp, oval_fly as o
pos=np.array([0,0,-1.0]); vel=np.zeros(3); s=0.0; ms=[]
for i in range(400):
    s,ref=o.reference(pos[:2],s,-1.0); o.P0.value,o.V0.value,o.REF.value=pos,vel,ref
    t=time.perf_counter(); o.PROB.solve(solver=cp.CLARABEL); ms.append(1000*(time.perf_counter()-t))
    pos,vel=o.p.value[1],o.v.value[1]
    if s>o.ARC[-1]-0.05 and np.linalg.norm(pos[:2])<0.05: break
ms=np.array(ms); print(f"racetrack: {len(ms)} solves: median {np.median(ms):.1f} ms, p95 {np.percentile(ms,95):.1f} ms, max {ms.max():.1f} ms")
