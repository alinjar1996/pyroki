import time
import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
import pyroki_snippets as pks
from scipy.spatial.transform import Rotation as R, Slerp


def slerp_quaternion(q1, q2, t):
    key_rots = R.from_quat([q1, q2])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    interp_rot = slerp([t])
    return interp_rot.as_quat()[0]


def lerp_position(p1, p2, t):
    return (1 - t) * np.array(p1) + t * np.array(p2)


def main():
    urdf1 = load_robot_description("ur5_description")
    urdf2 = load_robot_description("ur5_description")

    robot1 = pk.Robot.from_urdf(urdf1)
    robot2 = pk.Robot.from_urdf(urdf2)

    target_link_name1 = "ee_link"
    target_link_name2 = "ee_link"

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=3, height=3)

    # Create transform controls (interactive or programmatic) to offset roots
    base_offset1 = server.scene.add_transform_controls(
        "/robot1_base", position=(0, 0, 0), wxyz=(0, 0, 0, 1), scale=0.1
    )
    base_offset2 = server.scene.add_transform_controls(
        "/robot2_base", position=(0.7, 0, 0), wxyz=(0, 0, 0, 1), scale=0.1
    )

    # Attach robot visuals to these transform controls by specifying root_node_name
    urdf_vis1 = ViserUrdf(server, urdf1, root_node_name="/robot1_base")
    urdf_vis2 = ViserUrdf(server, urdf2, root_node_name="/robot2_base")

    start_pos1 = [0.5, 0.0, 0.3]
    end_pos1 = [0.4, 0.2, 0.5]
    start_quat1 = [0, 0, 0, 1]
    end_quat1 = [0, 0, 0, 1]

    start_pos2 = [0.5, 0.0, 0.3]
    end_pos2 = [0.4, -0.2, 0.5]
    start_quat2 = [0, 0, 0, 1]
    end_quat2 = [0, 0, 0, 1]

    steps = 500
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    for i in range(steps + 1):
        t = i / steps

        interp_pos1 = lerp_position(start_pos1, end_pos1, t)
        interp_quat1 = slerp_quaternion(start_quat1, end_quat1, t)

        interp_pos2 = lerp_position(start_pos2, end_pos2, t)
        interp_quat2 = slerp_quaternion(start_quat2, end_quat2, t)

        start_time = time.time()
        sol1 = pks.solve_ik(
            robot=robot1,
            target_link_name=target_link_name1,
            target_position=interp_pos1,
            target_wxyz=interp_quat1,
        )
        sol2 = pks.solve_ik(
            robot=robot2,
            target_link_name=target_link_name2,
            target_position=interp_pos2,
            target_wxyz=interp_quat2,
        )
        elapsed = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed * 1000)

        # Update robot configurations
        urdf_vis1.update_cfg(sol1)
        urdf_vis2.update_cfg(sol2)

        # Optionally update base_offset positions dynamically here
        # base_offset1.position = (x1, y1, z1)
        # base_offset2.position = (x2, y2, z2)

        time.sleep(0.02)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
