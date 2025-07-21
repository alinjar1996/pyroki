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
    """Spherical linear interpolation of two quaternions q1 to q2."""
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    slerp = Slerp([0, 1], R.concatenate([r1, r2]))
    interp_rot = slerp([t])
    return interp_rot.as_quat()[0]


def generate_trajectory(start_pos, end_pos, start_quat, end_quat, steps):
    """Generate linear interpolated positions and slerped orientations."""
    positions = np.linspace(start_pos, end_pos, steps)
    quaternions = []
    for t in np.linspace(0, 1, steps):
        quat = slerp_quaternion(start_quat, end_quat, t)
        quaternions.append(quat)
    return positions, np.array(quaternions)


def main():
    # Load UR5 robot description and specify end-effector link
    urdf = load_robot_description("ur5_description")
    target_link_name = "ee_link"  # Confirm in UR5 URDF, often 'ee_link' or 'tool0'

    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollision.from_urdf(urdf)

    # Collision objects: ground plane and a sphere obstacle
    plane_coll = HalfSpace.from_point_and_normal(np.array([0, 0, 0]), np.array([0, 0, 1]))
    # sphere_coll = Sphere.from_center_and_radius(np.array([0.4, 0.3, 0.4]), 0.05)
    sphere_coll = Sphere.from_center_and_radius(np.array([0.0, 0.0, 0.0]), 0.05)

    # Setup visualizer
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Add interactive transform control for sphere obstacle (optional)
    # sphere_handle = server.scene.add_transform_controls(
    #     "/obstacle", scale=0.2, position=tuple(sphere_coll.pose.translation()), wxyz=(0, 0, 1, 0)
    # )

    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.2, position=(0.4, 0.3, 0.4), wxyz=(0, 0, 1, 0))
    
    

    
    server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=sphere_coll.to_trimesh())

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # Define user trajectory: linear path and orientation for end-effector
    steps = 900
    start_position = np.array([0.5, 0.0, 0.3])
    end_position = np.array([0.3, 0.3, 0.5])
    start_quaternion = np.array([0, 0, 0, 1])  # Identity quaternion (x,y,z,w)
    end_quaternion = np.array([0, 0, 1, 0])    # Example rotation

    positions, quaternions = generate_trajectory(
        start_position, end_position, start_quaternion, end_quaternion, steps
    )

    for i in range(steps):
        # Update obstacle collision position if moved interactively
        sphere_coll_current = sphere_coll.transform_from_wxyz_position(
            wxyz=np.array(sphere_handle.wxyz), position=np.array(sphere_handle.position)
        )
        world_coll_list = [plane_coll, sphere_coll_current]

        start_time = time.time()
        solution = pks.solve_ik_with_collision(
            robot=robot,
            coll=robot_coll,
            world_coll_list=world_coll_list,
            target_link_name=target_link_name,
            target_position=positions[i],
            target_wxyz=quaternions[i],
        )
        elapsed = (time.time() - start_time) * 1000
        timing_handle.value = timing_handle.value * 0.99 + 0.01 * elapsed

        urdf_vis.update_cfg(solution)

        time.sleep(0.02)  # for a smooth visual update (~50 Hz)

    # Keep GUI alive
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
