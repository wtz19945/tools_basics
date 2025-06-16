import os

import numpy as np
import mujoco
import mujoco.viewer

import time

import scipy.sparse as sparse
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as Rlib

"""
    In this file we will learn how to locate a mujoco xml file
    and get its file path using the os module from the standard library.

    We will also learn how to load the xml file and simulate the system.
"""

def main(argv=None):
    # xml_path = os.path.expanduser('~/Tianze/mujoco_work/mujoco_menagerie/unitree_g1/scene.xml')
    # Load the XML file
    filepath = os.path.join(os.path.dirname(__file__), 'scene.xml')
    model = mujoco.MjModel.from_xml_path(filepath)
    data = mujoco.MjData(model)

    # Set timestep
    model.opt.timestep = 0.002

    # Reset to keyframe
    # keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    # mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)

    # Get control limits


    # Start visualization
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            # time.sleep(15)
            # Step Simulation
            mujoco.mj_step(model, data)

            ncon = data.ncon
            if ncon > 0:
                contact = data.contact[0]

            GRAVITY_SENSOR = "hrr_force"
            sensor_id = model.sensor(GRAVITY_SENSOR).id
            sensor_adr = model.sensor_adr[sensor_id]
            sensor_dim = model.sensor_dim[sensor_id]
            global_vector_adr = np.array(list(range(sensor_adr, sensor_adr + sensor_dim)))
            print(data.sensordata[global_vector_adr].ravel())
            print(" ")

            # Get constant torque input
            u = np.array([0.0, 16.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0])
            data.ctrl[:] = u
            # Sync Viewer
            viewer.sync()
            time.sleep(0.002)  # Sleep to match timestep


if __name__ == '__main__':
    main()
