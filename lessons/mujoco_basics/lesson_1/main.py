import os

import numpy as np
import mujoco
import mujoco.viewer

import time

"""
    In this file we will learn how to locate a mujoco xml file
    and get its file path using the os module from the standard library.

    We will also learn how to load the xml file and simulate the system.
"""


def main(argv=None):
    filepath = os.path.join(os.path.dirname(__file__), 'cart_pole.xml')

    # Load the xml file
    model = mujoco.MjModel.from_xml_path(filepath)
    print('Model loaded successfully!')

    # Update timestep:
    model.opt.timestep = 0.002

    # Create a data object#
    data = mujoco.MjData(model)
    print('Data loaded successfully!')

    # Update the initial state
    data.qpos = np.array([0.0, -np.pi / 2])
    data.qvel = np.array([0.0, 0.0])
    data.ctrl = np.array([0.0])
    mujoco.mj_forward(model, data)

    # Visualize and Simulate the system
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Move Cartpole to 1.0:
            desired_position = 1.0
            kp = 5.0
            kd = 2.0

            # Get the positions and velocity:
            qpos_cart = data.qpos[0]
            qvel_cart = data.qvel[0]

            # PD Controller:
            u = kp * (desired_position - qpos_cart) - kd * qvel_cart
            data.ctrl = np.array([u])

            # Simulate the system for a few steps:
            mujoco.mj_step(model, data)

            # Update the viewer
            viewer.sync()

            # Sleep for a bit to visualize the simulation:
            time.sleep(0.002)


if __name__ == '__main__':
    main()
