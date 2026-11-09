"""Racetrack demo: follow an out-and-back path with a semicircle at the far end, inside a hard
15 ft x 5 ft flight area, re-planned every 0.1 s with CVXPY + Clarabel.

The course is laid out from wherever the PILOT ARMS the drone, in the direction the drone faces:
  start -> right lane (+2.5 ft) -> 15 ft forward -> 2.5 ft-radius semicircle -> left lane -> back -> start.

Sequence:
  1. Pilot arms in Position mode. This script never arms or disarms.
  2. Script streams setpoints, switches PX4 to OFFBOARD, climbs to --alt (1.0 m).
  3. Every 0.1 s: find progress along the path, take the next 3 s of path as the reference, solve for
     the plan closest to it within speed/accel limits and inside the flight area, stream it to PX4.
  4. Back at the start: land (PX4 Land mode).
  The pilot can take over at any time by switching mode; the script then stops sending and exits.

--sim-pilot: in the sim there is no pilot, so the script does the pilot's part (Position mode + arm).
"""
import argparse, json, time
import numpy as np, cvxpy as cp
from px4link import Px4, N, DT

FT = 0.3048
LENGTH, HALF_W = 15 * FT, 2.5 * FT          # 4.57 m long, lanes 0.76 m either side of center
SPEED, VMAX, AMAX = 0.5, 1.0, 1.5           # cruise along the path, hard limits
TOL = 0.15                                  # flight area = 15 x 5 ft + 15 cm
AREA_LO, AREA_HI = np.array([0.0, -HALF_W]) - TOL, np.array([LENGTH, HALF_W]) + TOL   # course frame
OFFBOARD, POSCTL = 6 << 16, 3 << 16         # PX4 custom_mode values in HEARTBEAT


def build_path(step=0.02):
    """Racetrack in the course frame (x forward from the arming spot, y right), dense, with arc length."""
    r, far = HALF_W, LENGTH - HALF_W
    pts = [np.zeros(2), np.array([0.0, r]), np.array([far, r])]
    pts += [np.array([far + r * np.sin(t), r * np.cos(t)]) for t in np.linspace(0, np.pi, 60)[1:]]
    pts += [np.array([0.0, -r]), np.zeros(2)]
    dense = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        k = max(1, int(np.linalg.norm(b - a) / step))
        dense += [a + (b - a) * t for t in np.linspace(0, 1, k + 1)[1:]]
    dense = np.array(dense)
    return dense, np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])


PATH, ARC = build_path()

# ---- optimization problem, in the course frame: track the reference, never leave the area ----
p = cp.Variable((N + 1, 3)); v = cp.Variable((N + 1, 3)); a = cp.Variable((N, 3))
P0, V0, REF = cp.Parameter(3), cp.Parameter(3), cp.Parameter((N, 3))
cons = [p[0] == P0, v[0] == V0,
        p[1:, :2] >= np.tile(AREA_LO, (N, 1)), p[1:, :2] <= np.tile(AREA_HI, (N, 1))]
for k in range(N):
    cons += [p[k + 1] == p[k] + DT * v[k] + 0.5 * DT**2 * a[k], v[k + 1] == v[k] + DT * a[k],
             cp.norm(v[k + 1]) <= VMAX, cp.norm(a[k]) <= AMAX]
PROB = cp.Problem(cp.Minimize(cp.sum_squares(p[1:] - REF) + 0.3 * cp.sum_squares(a)), cons)


def reference(pos_xy, s_prev, z):
    """Progress = closest path point searching forward only (the two lanes run side by side),
    then the next N points at cruise speed."""
    lo, hi = np.searchsorted(ARC, s_prev), np.searchsorted(ARC, s_prev + 1.0)
    i = lo + int(np.argmin(np.linalg.norm(PATH[lo:hi + 1] - pos_xy, axis=1)))
    s_now = ARC[min(i, len(ARC) - 1)]
    s_ref = np.minimum(s_now + SPEED * DT * np.arange(1, N + 1), ARC[-1])
    xy = np.column_stack([np.interp(s_ref, ARC, PATH[:, 0]), np.interp(s_ref, ARC, PATH[:, 1])])
    return s_now, np.column_stack([xy, np.full(N, z)])


class Frame:
    """Course frame <-> PX4 local NED: origin at the arming spot, x along the heading at arming."""
    def __init__(self, origin, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        self.o, self.R = np.asarray(origin, float), np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    def to_course(self, x, vec=False): return (x if vec else x - self.o) @ self.R          # R^T x
    def to_ned(self, x, vec=False): return x @ self.R.T + (0 if vec else self.o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="udpout:127.0.0.1:14540")
    ap.add_argument("--alt", type=float, default=1.0)
    ap.add_argument("--speed", type=float, default=0.5, help="cruise along the path, m/s (max 0.8)")
    ap.add_argument("--sim-pilot", action="store_true", help="sim only: do the pilot's Position-mode arm")
    ap.add_argument("--log", default="oval_log.json")
    args = ap.parse_args()
    global SPEED
    SPEED = min(args.speed, 0.8)
    px4 = Px4(args.url)
    px4.sim_sticks = args.sim_pilot
    print(f"position updates: {px4.pos_hz:.0f} Hz")
    if px4.pos_hz < 8: print("position updates too slow for a 10 Hz planner: not flying"); return
    P0.value, V0.value = np.zeros(3), np.zeros(3); REF.value = reference(np.zeros(2), 0.0, -args.alt)[1]
    t = time.perf_counter(); PROB.solve(solver=cp.CLARABEL)       # warm-up: the first solve sets up the problem
    print(f"solver ready (first solve {1000 * (time.perf_counter() - t):.0f} ms)")

    def wait(cond, timeout, step=0.1):
        t_end = time.time() + timeout
        while time.time() < t_end:
            if cond(): return True
            time.sleep(step)
        return False

    def pilot_arm():          # what the pilot does on the transmitter, done over MAVLink in the sim
        px4.cmd(176, 1, 3)    # Position mode
        assert wait(lambda: px4.mode == POSCTL, 5), f"sim pilot: Position mode refused (mode={px4.mode})"
        px4.arm(); assert wait(lambda: px4.armed, 5), "sim pilot: arming refused"

    def land():
        print("landing"); px4.land(); wait(lambda: not px4.armed, 30, 0.5); time.sleep(2)

    def offboard_climb(z_air):
        """Armed on the ground: hold setpoint at climb height, switch to OFFBOARD, wait for the climb."""
        px4.hold = np.array([px4.pos[0], px4.pos[1], z_air]); time.sleep(1.0)   # PX4 needs setpoints first
        px4.offboard()
        assert wait(lambda: px4.mode == OFFBOARD, 5), f"OFFBOARD refused (mode={px4.mode})"
        print(f"OFFBOARD, climbing to {args.alt} m")
        wait(lambda: abs(px4.pos[2] - z_air) < 0.1 or px4.mode != OFFBOARD, 20); time.sleep(2)

    if args.sim_pilot:
        pilot_arm()
    else:
        print("waiting for the pilot to arm in Position mode (Ctrl-C to quit)")
        wait(lambda: px4.armed, 600, 0.2)
    assert px4.armed, "not armed"
    f = Frame([px4.pos[0], px4.pos[1], px4.pos[2]], px4.yaw)   # course runs along the drone's heading
    z_air_c = -args.alt                                       # course frame: origin at the ground, NED z
    corners = [f.to_ned(np.array([x, y, 0]))[:2] for x, y in
               [(0, -HALF_W), (LENGTH, -HALF_W), (LENGTH, HALF_W), (0, HALF_W)]]
    print(f"armed at NED {np.round(f.o, 2)}, course heading {np.degrees(np.arctan2(f.R[1,0], f.R[0,0])):.0f} deg; area corners "
          + " ".join(f"({c[0]:.2f},{c[1]:.2f})" for c in corners))
    yaw_arm = px4.yaw
    offboard_climb(f.to_ned(np.array([0, 0, z_air_c]))[2])
    # Heading on the ground (no flow below ~8 cm) can be wrong. Re-take it in the air, and refuse to fly
    # the course if it moved more than 15 deg during the climb.
    dyaw = np.degrees(np.angle(np.exp(1j * (px4.yaw - yaw_arm))))
    print(f"heading at arm {np.degrees(yaw_arm):.0f} deg, after climb {np.degrees(px4.yaw):.0f} deg (change {dyaw:+.0f})")
    if abs(dyaw) > 15:
        print("heading moved more than 15 deg during the climb: not flying the course"); land(); px4.alive = False; return
    f = Frame(f.o, px4.yaw)

    log = {"origin": f.o.tolist(), "yaw": px4.yaw, "path": PATH.tolist(), "samples": []}
    s_prev, t0, ms_all, took_over = 0.0, time.time(), [], False
    print("racetrack: out, semicircle, back")
    while True:
        tick = time.time()
        if px4.mode != OFFBOARD or not px4.armed:
            took_over = True; print("pilot took over (left OFFBOARD): stopping"); break
        pos_c, vel_c = f.to_course(px4.pos), f.to_course(px4.vel, vec=True)
        s_prev, ref = reference(pos_c[:2], s_prev, z_air_c)
        P0.value, V0.value, REF.value = pos_c, vel_c, ref
        t = time.perf_counter(); PROB.solve(solver=cp.CLARABEL); ms = 1000 * (time.perf_counter() - t)
        if PROB.status in ("optimal", "optimal_inaccurate"):
            plan = (tick, f.to_ned(p.value), f.to_ned(v.value, vec=True), f.to_ned(a.value, vec=True))
            with px4.lock: px4.plan = plan
        log["samples"].append({"t": tick - t0, "pos": pos_c.tolist(), "s": s_prev, "solve_ms": ms, "status": PROB.status})
        ms_all.append(ms)
        if s_prev > ARC[-1] - 0.05 and np.linalg.norm(pos_c[:2]) < 0.15 and np.linalg.norm(vel_c) < 0.15: break
        if tick - t0 > 60: print("timeout"); break
        time.sleep(max(0, DT - (time.time() - tick)))
    with px4.lock: px4.plan = None; px4.hold = None if took_over else f.to_ned(np.array([0, 0, z_air_c]))
    if not took_over:
        print(f"lap done in {time.time() - t0:.1f} s"); time.sleep(2); land()

    P = np.array([s["pos"] for s in log["samples"]])[:, :2]
    dev = [np.min(np.linalg.norm(PATH - q, axis=1)) for q in P]
    outside = np.maximum(np.maximum(AREA_LO + TOL - P, P - (AREA_HI - TOL)), 0).max()
    json.dump(log, open(args.log, "w")); px4.alive = False
    print(f"path deviation median {np.median(dev)*100:.0f} cm, max {np.max(dev)*100:.0f} cm; "
          f"furthest outside the 15x5 ft area {outside*100:.0f} cm; "
          f"solves {len(ms_all)}, median {np.median(ms_all):.1f} ms, max {np.max(ms_all):.1f} ms")


if __name__ == "__main__":
    main()
