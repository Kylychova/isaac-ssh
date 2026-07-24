"""
=====================================================================
  AK70-10 KV100 — dynamic motor simulation in Isaac Sim 5.1
=====================================================================
Motor only: housing fixed, output shaft free. A dyno test.

Self-contained: writes its own URDF, no extra files needed.

Written against the official Isaac Sim 5.1 API docs:
  docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_simulation/articulation_controller.html

Lessons baked in:
  * robot.initialize() AFTER world.reset()      - mandatory per docs
  * set_max_efforts() sets the torque ceiling
  * "For effort control, stiffness and damping must be set to zero"
  * GPU check up front - Blackwell (RTX 5090) silently breaks PhysX

Run:  python isaac_motor.py
=====================================================================
"""
import os, math, csv, sys

# ---------------------------------------------------------------- URDF
URDF_DIR = "/root/ak70"
URDF = f"{URDF_DIR}/motor.urdf"
os.makedirs(URDF_DIR, exist_ok=True)
with open(URDF, "w") as f:
    f.write('''<?xml version="1.0"?>
<robot name="ak70_10_motor">
  <link name="base_link">
    <inertial><mass value="0.45"/>
      <inertia ixx="3.17468e-04" ixy="0" ixz="0" iyy="3.17468e-04" iyz="0" izz="4.455562e-04"/>
    </inertial>
    <visual><geometry><cylinder radius="0.0445" length="0.05025"/></geometry></visual>
    <collision><geometry><cylinder radius="0.0445" length="0.05025"/></geometry></collision>
  </link>
  <link name="output_link">
    <inertial><mass value="0.071"/>
      <inertia ixx="1.6827e-05" ixy="0" ixz="0" iyy="1.6827e-05" iyz="0" izz="3.195e-05"/>
    </inertial>
    <visual><origin xyz="0 0 0.006"/><geometry><cylinder radius="0.030" length="0.012"/></geometry></visual>
    <collision><origin xyz="0 0 0.006"/><geometry><cylinder radius="0.030" length="0.012"/></geometry></collision>
  </link>
  <joint name="joint_output" type="continuous">
    <parent link="base_link"/><child link="output_link"/>
    <origin xyz="0 0 0.025125"/><axis xyz="0 0 1"/>
    <limit effort="24.8" velocity="50.27"/>
    <dynamics damping="0.0" friction="0.0"/>
  </joint>
</robot>
''')
print(f"URDF written: {URDF}")

# ------------------------------------------------------------ boot Isaac
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np
import omni.kit.commands
from pxr import PhysxSchema
from isaacsim.core.api import World
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

# ------------------------------------------- AK70-10 KV100 (datasheet)
GEAR         = 10.0
ARMATURE     = 7.534788e-05 * GEAR**2   # 7.5348e-3 kg*m^2 at the output
PEAK_TORQUE  = 24.8                     # Nm
RATED_TORQUE = 8.3                      # Nm
NOLOAD_SPEED = 480/60*2*math.pi         # 50.27 rad/s
FRICTION     = 0.48                     # Nm back-drive
KT_OUT       = 0.123*GEAR               # 1.23 Nm/A
HZ           = 1000

def rpm(w): return w*60/(2*math.pi)
def available(w):
    """BLDC torque-speed curve: stall torque -> 0 at no-load speed."""
    return PEAK_TORQUE*max(0.0, min(1.0, 1.0 - abs(w)/NOLOAD_SPEED))

print("\n" + "="*68)
print("  AK70-10 KV100 — motor characterization, Isaac Sim 5.1")
print("="*68)

world = World(physics_dt=1/HZ, rendering_dt=1/60, stage_units_in_meters=1.0)

st, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
cfg.merge_fixed_joints    = False
cfg.fix_base              = True
cfg.import_inertia_tensor = True
cfg.make_default_prim     = True
cfg.self_collision        = False
st, root = omni.kit.commands.execute(
    "URDFParseAndImportFile", urdf_path=URDF, import_config=cfg)
print(f"  root prim : {root}")

stage = get_current_stage()
jpath = [p.GetPath().pathString for p in stage.Traverse()
         if p.GetName() == "joint_output"][0]
print(f"  joint     : {jpath}")
pj = PhysxSchema.PhysxJointAPI.Apply(stage.GetPrimAtPath(jpath))
pj.CreateArmatureAttr(ARMATURE)
pj.CreateJointFrictionAttr(FRICTION)
pj.CreateMaxJointVelocityAttr(NOLOAD_SPEED)

robot = SingleArticulation(prim_path=root, name="ak70_motor")
world.scene.add(robot)
world.reset()
robot.initialize()                       # <- required by the docs

ctrl = robot.get_articulation_controller()
ctrl.set_gains(kps=np.array([0.0]), kds=np.array([0.0]))   # torque mode
ctrl.set_max_efforts(np.array([PEAK_TORQUE]))
try:
    ctrl.switch_control_mode("effort")
except Exception as e:
    print(f"  (switch_control_mode: {type(e).__name__})")

# -------------------------------------------------------- DIAGNOSTICS
print("\n" + "-"*68)
print("  DIAGNOSTICS")
print("-"*68)
print(f"  dof names   : {list(robot.dof_names)}")
try:
    kps, kds = ctrl.get_gains()
    print(f"  gains Kp/Kd : {kps} / {kds}   <- must be 0")
except Exception as e:
    print(f"  gains       : ERR {e}")
try:
    print(f"  max efforts : {ctrl.get_max_efforts()}   <- must be 24.8")
except Exception as e:
    print(f"  max efforts : ERR {e}")
print(f"  armature    : {pj.GetArmatureAttr().Get()}")
print(f"  friction    : {pj.GetJointFrictionAttr().Get()}")
print("-"*68)


def run(seconds, cmd_fn, w0=0.0):
    robot.set_joint_velocities(np.array([w0]))
    n, log = int(seconds*HZ), []
    for i in range(n):
        t = i/HZ
        w = float(robot.get_joint_velocities()[0])
        lim = available(w)
        tau = max(-lim, min(lim, cmd_fn(t, w)))
        robot.apply_action(ArticulationAction(joint_efforts=np.array([tau])))
        world.step(render=False)
        log.append((t, w, tau))
    return log


# ---- 1  spin-up at peak torque
T1 = run(0.5, lambda t, w: PEAK_TORQUE)
w_term = T1[-1][1]
t95 = next((r[0] for r in T1 if abs(r[1]) >= 0.95*abs(w_term)), None) \
      if abs(w_term) > 1e-6 else None
print(f"\n[1]  SPIN-UP AT PEAK TORQUE (24.8 Nm)")
print(f"     terminal speed : {w_term:.2f} rad/s = {rpm(w_term):.0f} rpm")
if t95: print(f"     time to 95%    : {t95*1000:.0f} ms")
print(f"     theory accel   : {(PEAK_TORQUE-FRICTION)/ARMATURE:.0f} rad/s2")

# ---- 2  spin-up at rated torque
T2 = run(0.5, lambda t, w: RATED_TORQUE)
print(f"\n[2]  SPIN-UP AT RATED TORQUE (8.3 Nm)")
print(f"     terminal speed : {rpm(T2[-1][1]):.0f} rpm")
print(f"     current        : {RATED_TORQUE/KT_OUT:.1f} A  (datasheet 7.2 A)")

# ---- 3  torque steps
def steps(t, w):
    if t < 0.04: return 5.0
    if t < 0.08: return 15.0
    if t < 0.12: return 0.0
    return -8.0
T3 = run(0.20, steps)
print(f"\n[3]  TORQUE STEPS (5 -> 15 -> 0 -> -8 Nm, 40 ms each)")
print(f"       t[ms]  speed[rpm]  tau[Nm]")
for mk in (0.039, 0.079, 0.119, 0.199):
    k = int(mk*HZ)
    print(f"      {mk*1000:5.0f}  {rpm(T3[k][1]):9.0f}  {T3[k][2]:+7.2f}")

# ---- 4  coast-down
T4 = run(1.0, lambda t, w: 0.0, w0=w_term)
t_stop = next((r[0] for r in T4 if abs(r[1]) < 0.05), None)
print(f"\n[4]  COAST-DOWN (torque cut)")
if t_stop:
    print(f"     from {rpm(w_term):.0f} rpm -> stop in {t_stop*1000:.0f} ms")
else:
    print(f"     still at {rpm(T4[-1][1]):.0f} rpm after 1 s")
print(f"     theory decel   : {FRICTION/ARMATURE:.0f} rad/s2")

# ---- 5  torque-speed curve
ws = np.linspace(0, NOLOAD_SPEED, 100)
tq = [available(w) for w in ws]
pw = [t*w for t, w in zip(tq, ws)]
i  = int(np.argmax(pw))
print(f"\n[5]  TORQUE-SPEED CURVE")
print(f"     stall      : {tq[0]:.1f} Nm @ 0 rpm")
print(f"     no-load    : {rpm(ws[-1]):.0f} rpm @ 0 Nm")
print(f"     peak power : {pw[i]:.0f} W @ {rpm(ws[i]):.0f} rpm")

print("\n" + "="*68)
print(f"  RESULT: unloaded shaft settles at {rpm(w_term):.0f} rpm")
print(f"  (independent calculation predicts ~452 rpm)")
print("="*68)

if abs(rpm(w_term)) < 50:
    print("\n  !! SHAFT NOT SPINNING. Check the log for:")
    print("     'no suitable CUDA GPU'  /  'Failed to create any GPU devices'")
    print("     -> that means the GPU is incompatible (Blackwell?), not the code.")
elif abs(rpm(w_term) - 452) < 60:
    print("\n  ** Isaac Sim agrees with the independent calculation. **")

with open("/root/motor_isaac.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["test","t_s","omega_rad_s","rpm","tau_Nm","current_A"])
    for nm, lg in (("peak",T1),("rated",T2),("steps",T3),("coast",T4)):
        for r in lg[::5]:
            wr.writerow([nm, f"{r[0]:.4f}", f"{r[1]:.4f}", f"{rpm(r[1]):.1f}",
                         f"{r[2]:.4f}", f"{r[2]/KT_OUT:.2f}"])
print("\n  CSV -> /root/motor_isaac.csv")

os._exit(0)
