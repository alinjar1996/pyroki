import time
import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
import pyroki_snippets as pks
from scipy.spatial.transform import Rotation as R, Slerp


def slerp_quaternion(q1, q2, t):
    """
    Perform spherical linear interpolation between quaternions q1 and q2 at fraction t [0..1].
    Quaternions q1 and q2 should be in (x,y,z,w) format.
    """
    key_rots = R.from_quat([q1, q2])          # create Rotation objects for key frames
    key_times = [0, 1]                        # define times for those key frames
    slerp = Slerp(key_times, key_rots)       # initialize Slerp
    interp_rot = slerp([t])                   # interpolate at time t (list input)
    return interp_rot.as_quat()[0]            # return interpolated quaternion


def lerp_position(p1, p2, t):
    """Linear interpolation of positions p1 to p2 by fraction t (0 to 1)."""
    return (1 - t) * np.array(p1) + t * np.array(p2)


def main():
    # urdf = load_robot_description("panda_description")
    # target_link_name = "panda_hand"

    urdf = load_robot_description("ur5_description")
    target_link_name = "ee_link"

    robot = pk.Robot.from_urdf(urdf)

    # print("Available link names:", robot.links.names)  # Debug print

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Remove interactive control; define start/end targets programmatically
    start_pos = np.array([0.61, 0.0, 0.56])
    start_quat = np.array([0, 0, 1, 0])       # (x,y,z,w)

    end_pos = np.array([0.3, 0.0, 0.2])
    end_quat = np.array([0, 0, 0, 1])         # (x,y,z,w)

    steps = 1000  # number of interpolation steps
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    for i in range(steps + 1):
        t = i / steps

        # Interpolate position and orientation target
        interp_pos = lerp_position(start_pos, end_pos, t)
        interp_quat = slerp_quaternion(start_quat, end_quat, t)

        # Solve IK for the interpolated target pose
        start_time = time.time()
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=interp_pos,
            target_wxyz=interp_quat,
        )
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        urdf_vis.update_cfg(solution)

        # Delay to visualize smoothly
        time.sleep(0.02)  # 20 ms per step

    # Keep visualization window open after motion completes
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
