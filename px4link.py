"""PX4 offboard link over MAVLink (pymavlink): streams a planner's latest plan to PX4 as
position + velocity + acceleration setpoints at 20 Hz, and tracks position, heading, mode and arming."""
import threading, time
import numpy as np
from pymavlink import mavutil

N, DT = 30, 0.1                    # planner horizon: 30 steps x 0.1 s = 3 s


class Px4:
    MASK = 0b0000_1100_0000_0000   # use pos+vel+acc, ignore yaw + yaw rate (bit 9 = force, must stay 0)

    def __init__(self, url):
        self.m = mavutil.mavlink_connection(url, source_system=250)
        self.lock, self.plan, self.hold = threading.Lock(), None, None
        self.pos = self.vel = None; self.mode = None; self.armed = False; self.yaw = 0.0; self.n_pos = 0
        self.alive = True
        threading.Thread(target=self._rx, daemon=True).start()
        threading.Thread(target=self._tx, daemon=True).start()
        while self.pos is None: time.sleep(0.05)
        # PX4's companion link may send position as slowly as 0.5 Hz (it did on a DEXI-3): ask for 50 Hz
        # position and attitude, then check what actually arrives. A planner fed stale state anchors the drone.
        self.cmd(511, 32, 20000)    # LOCAL_POSITION_NED at 50 Hz
        self.cmd(511, 30, 200000)   # ATTITUDE at 5 Hz (heading only; keeps link bandwidth for position)
        time.sleep(1.0); n0 = self.n_pos; time.sleep(3.0); self.pos_hz = (self.n_pos - n0) / 3.0

    def _rx(self):
        while self.alive:
            msg = self.m.recv_match(blocking=True, timeout=0.5)
            if not msg: continue
            t = msg.get_type()
            if t == "LOCAL_POSITION_NED":
                self.pos = np.array([msg.x, msg.y, msg.z]); self.vel = np.array([msg.vx, msg.vy, msg.vz])
                self.n_pos += 1
            elif t == "ATTITUDE":
                self.yaw = msg.yaw
            elif t == "STATUSTEXT":
                print("  PX4:", msg.text.strip())
            elif t == "HEARTBEAT" and msg.get_srcSystem() == 1 and msg.get_srcComponent() == 1:
                self.mode, self.armed = msg.custom_mode, bool(msg.base_mode & 128)

    def _tx(self):   # 20 Hz setpoints + 1 Hz GCS heartbeat
        i = 0
        while self.alive:
            if i % 20 == 0: self.m.mav.heartbeat_send(6, 8, 0, 0, 0)
            if getattr(self, "sim_sticks", False): self.m.mav.manual_control_send(1, 0, 0, 0, 0, 0)  # sim: no RC, centered sticks, throttle low
            with self.lock: plan, hold = self.plan, self.hold
            if plan is not None:      # interpolate the latest plan at "now"
                t0, P, V, A = plan
                f = min((time.time() - t0) / DT, N - 1e-6); k = int(f); w = f - k
                sp_p = (1 - w) * P[k] + w * P[k + 1]; sp_v = (1 - w) * V[k] + w * V[k + 1]; sp_a = A[min(k, N - 1)]
            elif hold is not None:
                sp_p, sp_v, sp_a = hold, np.zeros(3), np.zeros(3)
            else:
                sp_p = None
            if sp_p is not None:
                self.m.mav.set_position_target_local_ned_send(0, 1, 1, 1, self.MASK, *sp_p, *sp_v, *sp_a, 0, 0)
                self.last_sp = sp_p
            i += 1; time.sleep(0.05)

    def cmd(self, c, *prm):
        self.m.mav.command_long_send(1, 1, c, 0, *(list(prm) + [0] * (7 - len(prm))))

    def offboard(self): self.cmd(176, 1, 6)                 # MAV_CMD_DO_SET_MODE, PX4 main mode OFFBOARD
    def arm(self, on=True): self.cmd(400, 1 if on else 0)
    def land(self): self.cmd(21)                            # MAV_CMD_NAV_LAND

