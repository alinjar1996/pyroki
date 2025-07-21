import time
import numpy as np
import pyroki as pk
import viser
from pyroki.collision import HalfSpace, RobotCollision, Sphere
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
import pyroki_snippets as pks
from scipy.spatial.transform import Rotation as R, Slerp


def slerp_quaternion(q1, q2, t):
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    slerp = Slerp([0, 1], R.concatenate([r1, r2]))
    interp_rot = slerp([t])
    return interp_rot.as_quat()[0]


def lerp_position(p1, p2, t):
    return (1 - t) * np.array(p1) + t * np.array(p2)


def generate_trajectory(start_pos, end_pos, start_quat, end_quat, steps):
    positions = np.linspace(start_pos, end_pos, steps)
    quats = []
    for t in np.linspace(0, 1, steps):
        quats.append(slerp_quaternion(start_quat, end_quat, t))
    return positions, np.array(quats)


def main():
    # Load UR5 URDF twice for two robots
    urdf1 = load_robot_description("ur5_description")
    urdf2 = load_robot_description("ur5_description")

    robot1 = pk.Robot.from_urdf(urdf1)
    robot2 = pk.Robot.from_urdf(urdf2)

    robot_coll1 = RobotCollision.from_urdf(urdf1)
    robot_coll2 = RobotCollision.from_urdf(urdf2)

    target_link1 = "ee_link"
    target_link2 = "ee_link"

    # Setup Viser server and scene
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=3, height=3, cell_size=0.1)

    # Create transform controls as root frames to separate robots spatially
    base1 = server.scene.add_transform_controls(
        "/robot1_base", scale=0.1, position=(0, -0.2, 0), wxyz=(0, 0, 0, 1)
    )
    base2 = server.scene.add_transform_controls(
        "/robot2_base", scale=0.1, position=(0.7, 0.2, 0), wxyz=(0, 0, 0, 1)
    )

    # Visualize robots attached to their respective root frames
    urdf_vis1 = ViserUrdf(server, urdf1, root_node_name="/robot1_base")
    urdf_vis2 = ViserUrdf(server, urdf2, root_node_name="/robot2_base")

    # Setup collision objects: ground plane and sphere obstacle shared for both robots
    plane_coll = HalfSpace.from_point_and_normal(np.array([0, 0, 0]), np.array([0, 0, 1]))
    sphere_center = np.array([0.2, 0.2, 0.2])
    sphere_radius = 0.2
    sphere_coll = Sphere.from_center_and_radius(sphere_center, sphere_radius)

    # Add interactive transform control for obstacle sphere
    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.2, position=tuple(sphere_center), wxyz=(0, 0, 0, 1)
    )
    server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=sphere_coll.to_trimesh())

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    steps = 900

    # Define trajectories for each robot's end-effector
    start_pos1, end_pos1 = np.array([0.5, 0.0, 0.3]), np.array([0.3, 0.3, 0.5])
    start_quat1, end_quat1 = np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0])

    start_pos2, end_pos2 = np.array([0.5, 0.0, 0.3]), np.array([0.3, -0.3, 0.5])
    start_quat2, end_quat2 = np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0])

    positions1, quats1 = generate_trajectory(start_pos1, end_pos1, start_quat1, end_quat1, steps)
    positions2, quats2 = generate_trajectory(start_pos2, end_pos2, start_quat2, end_quat2, steps)

    for i in range(steps):
        # Update sphere obstacle collision pose dynamically from user interaction
        sphere_coll_curr = sphere_coll.transform_from_wxyz_position(
            wxyz=np.array(sphere_handle.wxyz),
            position=np.array(sphere_handle.position),
        )
        world_coll_list = [plane_coll, sphere_coll_curr]

        start_time = time.time()

        # Solve IK with collision for robot 1
        sol1 = pks.solve_ik_with_collision(
            robot=robot1,
            coll=robot_coll1,
            world_coll_list=world_coll_list,
            target_link_name=target_link1,
            target_position=positions1[i],
            target_wxyz=quats1[i],
        )

        # Solve IK with collision for robot 2
        sol2 = pks.solve_ik_with_collision(
            robot=robot2,
            coll=robot_coll2,
            world_coll_list=world_coll_list,
            target_link_name=target_link2,
            target_position=positions2[i],
            target_wxyz=quats2[i],
        )

        elapsed = (time.time() - start_time) * 1000
        timing_handle.value = timing_handle.value * 0.99 + 0.01 * elapsed

        # Update visualizations
        urdf_vis1.update_cfg(sol1)
        urdf_vis2.update_cfg(sol2)

        time.sleep(0.02)  # smooth animation (~50 Hz)

    # Keep program alive so visualizer window stays open
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
