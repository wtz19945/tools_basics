import time
import math 

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('scene-fixed.xml')
d = mujoco.MjData(m)

default_configuration = {
    "left-hip-roll": 0.31,
    "left-hip-yaw": 0.0,
    "left-hip-pitch": 0.2,
    "left-knee": 0.19,
    "left-shin": 0.0,
    "left-tarsus": 0.0,
    "left-toe-pitch": -0.126,
    "left-toe-roll": 0.0,
    "left-shoulder-roll": -0.15,
    "left-shoulder-pitch": 1.1,
    "left-shoulder-yaw": 0.0,
    "left-elbow": -0.145,
    "right-hip-roll": -0.31,
    "right-hip-yaw": 0.0,
    "right-hip-pitch": -0.2,
    "right-knee": -0.19,
    "right-shin": 0.0,
    "right-tarsus": 0.0,
    "right-toe-pitch": 0.126,
    "right-toe-roll": 0.0,
    "right-shoulder-roll": 0.15,
    "right-shoulder-pitch": -1.1,
    "right-shoulder-yaw": 0.0,
    "right-elbow": 0.145
}

desired_configuration = {
    "left-hip-roll": 0.31,
    "left-hip-yaw": 0.0,
    "left-hip-pitch": 0.4,
    "left-knee": 0.19,
    "left-shin": 0.0,
    "left-tarsus": 0.0,
    "left-toe-pitch": -0.126,
    "left-toe-roll": 0.0,
    "left-shoulder-roll": -0.15,
    "left-shoulder-pitch": 1.1,
    "left-shoulder-yaw": 0.0,
    "left-elbow": -0.145,
    "right-hip-roll": -0.31,
    "right-hip-yaw": 0.0,
    "right-hip-pitch": -0.4,
    "right-knee": -0.19,
    "right-shin": 0.0,
    "right-tarsus": 0.0,
    "right-toe-pitch": 0.126,
    "right-toe-roll": 0.0,
    "right-shoulder-roll": 0.15,
    "right-shoulder-pitch": -1.1,
    "right-shoulder-yaw": 0.0,
    "right-elbow": 0.145
}

motor_names = ["left-hip-roll", "left-hip-yaw", "left-hip-pitch", 
    "left-knee", "left-toe-pitch", "left-toe-roll",  "right-hip-roll", "right-hip-yaw", 
    "right-hip-pitch", "right-knee", "right-toe-pitch", "right-toe-roll", "left-shoulder-roll", "left-shoulder-pitch", 
    "left-shoulder-yaw", "left-elbow", "right-shoulder-roll", 
    "right-shoulder-pitch", "right-shoulder-yaw", "right-elbow"]

motor_names2 = ["left-hip-roll", "left-hip-yaw", "left-hip-pitch", 
    "left-knee", "left-toe-A", "left-toe-B",  "right-hip-roll", "right-hip-yaw", 
    "right-hip-pitch", "right-knee", "right-toe-A", "right-toe-B", "left-shoulder-roll", "left-shoulder-pitch", 
    "left-shoulder-yaw", "left-elbow", "right-shoulder-roll", 
    "right-shoulder-pitch", "right-shoulder-yaw", "right-elbow"]

for joint, value in default_configuration.items():
  d.joint(joint).qpos[0] = value

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  count = 1
  while 1:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    count += 1
    if count < 500:
      d.joint('base').qvel[0] = 1
    else:
      d.joint('base').qvel[0] = -1
    if count == 1000:
      count = 0

    for i in range(0, len(motor_names)):
      q_des = desired_configuration[motor_names[i]] 
      if i == 2:
        q_des += .5 * math.sin(4*time.time())
      if i == 8:
        q_des += .5 * math.sin(4*time.time())
      if i == 12:
        q_des += .5 * math.sin(4*time.time())
      if i == 16:
        q_des += .5 * math.sin(4*time.time())
      d.actuator(motor_names2[i]).ctrl[0] = 150 * (q_des - d.joint(motor_names[i]).qpos[0]) + 5 * (- d.joint(motor_names[i]).qvel[0])

    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)