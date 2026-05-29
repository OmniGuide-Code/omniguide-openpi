import pybullet as p
import pybullet_data

# Start the simulation
p.connect(p.GUI) # or p.DIRECT for non-graphical mode

# Add the default search path for common PyBullet data (like plane.urdf)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load a URDF file
# Example 1: Loading a plane from pybullet_data
plane_id = p.loadURDF("panda_gripper.urdf")

# Example 2: Loading a custom robot URDF (replace "path/to/your/robot.urdf" with the actual path)
# robot_id = p.loadURDF("path/to/your/robot.urdf")

# You can also specify the base position and orientation when loading
# robot_id = p.loadURDF("path/to/your/robot.urdf", basePosition=[0, 0, 0.5], baseOrientation=[0, 0, 0, 1])

# Run the simulation (optional, for visualization)
while True:
    p.stepSimulation()