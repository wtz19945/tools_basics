import os

import numpy as np
import mujoco
import mujoco.viewer

import time

import osqp
import scipy.sparse as sparse
from scipy.linalg import block_diag

# Code for OSC using Quadratic Programming
# Do pip install osqp to run the code


def main(argv=None):
    filepath = os.path.join(os.path.dirname(__file__), 'digit/test_spring.xml')

    # Load the xml file
    model = mujoco.MjModel.from_xml_path(filepath)
    print('Model loaded successfully!')

    # Update timestep:
    model.opt.timestep = 0.002

    # Create a data object#
    data = mujoco.MjData(model)
    print('Data loaded successfully!')

    # Update the initial state
    model.opt.timestep = 0.002
    # Visualize and Simulate the system
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            p1 = data.site_xpos[model.site('attach1').id]
            p2 = data.site_xpos[model.site('attach2').id]
            
            # spring force computation
            k = 100000  # spring stiffness
            rest_length = 0

            disp = p2 - p1
            length = np.linalg.norm(disp)

            direction = disp / (length + 1e-6)

            force = -k * (length - rest_length) * direction

            site_id1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, 'attach1')
            body_id1 = model.site_bodyid[site_id1]
            
            site_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, 'attach2')
            body_id2 = model.site_bodyid[site_id2]  
            
            # apply equal and opposite force to both bodies
            mujoco.mj_applyFT(model, data,  force, np.zeros(3), p1, body_id1, data.qfrc_applied)
            mujoco.mj_applyFT(model, data, -force, np.zeros(3), p2, body_id2, data.qfrc_applied)
        
            # Simulate the system for a few steps:
            mujoco.mj_step(model, data)

            # Update the viewer
            viewer.sync()

            # Sleep for a bit to visualize the simulation:
            time.sleep(0.002)


if __name__ == '__main__':
    main()
