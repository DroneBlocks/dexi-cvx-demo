"""Benchmark CVXPY SDP solves (Clarabel, SCS) on this computer: max-cut relaxations, Lyapunov LMIs,
and a parameterized LMI re-solved in a loop. --big adds a 100x100 max-cut, which needs ~1.4 GB of RAM
with Clarabel: on a 2 GB DEXI stop DEXI-OS first or it will run out of memory."""
import sys
import time, statistics as st, numpy as np, cvxpy as cp, platform
rng = np.random.default_rng(0)
SOLVERS = ["CLARABEL", "SCS"]
def med(xs): return 1000*st.median(xs)

def maxcut(n):
    W = rng.random((n, n)); W = (W + W.T) / 2
    L = np.diag(W.sum(1)) - W
    X = cp.Variable((n, n), symmetric=True)
    return cp.Problem(cp.Maximize(cp.trace(L @ X) / 4), [X >> 0, cp.diag(X) == 1])

def lyap(n):
    A = rng.standard_normal((n, n)); A -= (np.max(np.linalg.eigvals(A).real) + 0.5) * np.eye(n)
    P = cp.Variable((n, n), symmetric=True)
    return cp.Problem(cp.Minimize(cp.trace(P)), [P >> np.eye(n), A.T @ P + P @ A << -np.eye(n)])

print(platform.node(), platform.machine())
print(f"{'problem':<16}{'solver':<10}{'total ms':>10}{'solver ms':>11}{'iters':>7}  status")
for name, fn, sizes in [("maxcut", maxcut, [10, 20, 50]), ("lyapunov LMI", lyap, [4, 8, 12, 24])]:
    for n in sizes:
        for s in SOLVERS:
            tot, sol = [], []
            for _ in range(5):
                p = fn(n); t = time.perf_counter(); p.solve(solver=s)
                tot.append(time.perf_counter() - t); sol.append(p.solver_stats.solve_time or 0)
            print(f"{name+' n='+str(n):<16}{s:<10}{med(tot):>10.1f}{med(sol):>11.1f}{p.solver_stats.num_iters or 0:>7}  {p.status}")

# repeated solve: parameterized Lyapunov LMI (DPP), new A each step
n = 12
Ap = cp.Parameter((n, n)); P = cp.Variable((n, n), symmetric=True)
prob = cp.Problem(cp.Minimize(cp.trace(P)), [P >> np.eye(n), Ap.T @ P + P @ Ap << -np.eye(n)])
for s in SOLVERS:
    ts = []
    for k in range(51):
        A = rng.standard_normal((n, n)); A -= (np.max(np.linalg.eigvals(A).real) + 0.5) * np.eye(n)
        Ap.value = A; t = time.perf_counter(); prob.solve(solver=s); ts.append(time.perf_counter() - t)
    print(f"param loop n={n} {s:<9} first {1000*ts[0]:.1f} ms, then median {med(ts[1:]):.1f} ms (~{1/st.median(ts[1:]):.0f} Hz)")

if "--big" not in sys.argv: sys.exit()
import resource
for s_ in SOLVERS:
    p = maxcut(100); t = time.perf_counter(); p.solve(solver=s_)
    print(f"maxcut n=100    {s_:<10}{1000*(time.perf_counter()-t):>10.1f}{1000*(p.solver_stats.solve_time or 0):>11.1f}  {p.status}  peakRSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024} MB")
