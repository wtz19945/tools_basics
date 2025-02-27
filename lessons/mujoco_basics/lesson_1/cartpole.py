import os
import time

import numpy as np

import mujoco
import mujoco.viewer


"""
    MacOS Must run this script via mjpython
"""


def main(argv=None):
    # Locate the xml file
    xml_path = os.path.join(os.path.dirname(__file__), 'cart_pole.xml')

    # Load the xml file
    model = mujoco.MjModel.from_xml_path(xml_path)

    # Update timestep:
    model.opt.timestep = 0.002
    visualization_rate = 0.02
    num_steps = int(visualization_rate / model.opt.timestep)

    data = mujoco.MjData(model)

    # Update the initial state
    data.qpos = np.array([0.0, -np.pi / 2])
    data.qvel = np.array([0.0, 0.0])
    data.ctrl = np.array([0.0])
    mujoco.mj_forward(model, data)

    # Visualize and Simulate the system
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Simulate the system for a few steps:
            for _ in range(num_steps):
                mujoco.mj_step(model, data)

            # Update the viewer
            viewer.sync()

            # Sleep for a bit to visualize the simulation:
            time.sleep(visualization_rate)


if __name__ == '__main__':
    main()
