import mujoco
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
# Define the model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Enable joint visualization option
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

# Duration and framerate
duration = 3.8  # seconds
framerate = 60  # Hz

# Create a figure for displaying the frames
fig, ax = plt.subplots()
ax.axis('off')

# Initialize the renderer
with mujoco.Renderer(model) as renderer:
    frames = []
    mujoco.mj_resetData(model, data)

    # Function to update and display the frame
    def update_frame(i):
        mujoco.mj_step(model, data)
        renderer.update_scene(data, scene_option=scene_option)
        pixels = renderer.render()

        # Update the image with the new frame
        ax.imshow(pixels)
        ax.set_title(f"Time: {data.time:.2f} seconds")

    # Create an animation that updates the frame
    ani = animation.FuncAnimation(fig, update_frame, frames=int(duration * framerate), interval=1000 / framerate, repeat=False)

    # Show the animation
    plt.show()