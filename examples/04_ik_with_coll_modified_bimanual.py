import time
import numpy as np
import pyroki as pk
import viser
from pyroki.collision import HalfSpace, RobotCollision, Sphere
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
import pyroki_snippets as pks
from scipy.spatial.transform import Rotation as R, Slerp
import xml.etree.ElementTree as ET
from trimesh import creation
import jax
from functools import partial

def slerp_quaternion(q1, q2, t):
    """Interpolate between two quaternions (in [w,x,y,z] format)"""
    q1_xyzw = np.array([q1[1], q1[2], q1[3], q1[0]])
    q2_xyzw = np.array([q2[1], q2[2], q2[3], q2[0]])
    r1 = R.from_quat(q1_xyzw)
    r2 = R.from_quat(q2_xyzw)
    slerp = Slerp([0, 1], R.concatenate([r1, r2]))
    interp_rot = slerp([t])
    interp_quat_xyzw = interp_rot.as_quat()[0]
    return np.array([interp_quat_xyzw[3], interp_quat_xyzw[0], interp_quat_xyzw[1], interp_quat_xyzw[2]])

def generate_trajectory(start_pos, end_pos, start_quat, end_quat, steps):
    """Generate smooth trajectory in task space"""
    positions = np.linspace(start_pos, end_pos, steps)
    quats = [slerp_quaternion(start_quat, end_quat, t) for t in np.linspace(0, 1, steps)]
    return positions, quats

def approximate_box_with_spheres(center, half_extents):
    """Approximate box collision with spheres"""
    spheres = []
    min_r = min(half_extents)
    spheres.append(Sphere.from_center_and_radius(center, min_r))
    for dx in [-half_extents[0], half_extents[0]]:
        for dy in [-half_extents[1], half_extents[1]]:
            for dz in [-half_extents[2], half_extents[2]]:
                corner = center + np.array([dx, dy, dz])
                spheres.append(Sphere.from_center_and_radius(corner, min_r / 2))
    return spheres

def set_mesh_color(mesh, rgb, alpha=1.0):
    color = np.array([int(c * 255) for c in rgb] + [int(alpha * 255)], dtype=np.uint8)
    mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))

def load_mjcf_to_scene(xml_path, server):
    root = ET.parse(xml_path).getroot()
    collision_objs = []
    body_poses = {}

    for body in root.findall('body'):
        body_name = body.get('name', None)
        body_pos = np.array([float(x) for x in body.get('pos', '0 0 0').split()])
        quat_wxyz = np.array([float(x) for x in body.get('quat', '1 0 0 0').split()])
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rot = R.from_quat(quat_xyzw)
        rot_mat = rot.as_matrix()

        if body_name is not None:
            body_poses[body_name] = (body_pos, quat_xyzw)

        for geom in body.findall('geom'):
            geom_pos = np.array([float(x) for x in geom.get('pos', '0 0 0').split()])
            rgba = [float(x) for x in geom.get('rgba', '1 1 1 1').split()]
            size = np.array([float(x) for x in geom.get('size', '0 0 0').split()])
            geom_type = geom.get('type', 'box')

            abs_pos = body_pos + rot.apply(geom_pos)
            mesh = None

            if geom_type == 'box':
                extents = 2 * size
                mesh = creation.box(extents=extents)
                collision_objs.extend(approximate_box_with_spheres(abs_pos, size))
            elif geom_type == 'sphere':
                radius = size[0]
                mesh = creation.icosphere(radius=radius)
                collision_objs.append(Sphere.from_center_and_radius(abs_pos, radius))
            else:
                continue

            if mesh is not None:
                transform = np.eye(4)
                transform[:3, :3] = rot_mat
                transform[:3, 3] = abs_pos
                mesh.apply_transform(transform)
                set_mesh_color(mesh, rgba[:3], rgba[3])

                server.scene.add_mesh_trimesh(
                    f"/mjcf_objects/{geom.get('name', 'geom')}",
                    mesh=mesh,
                )

    return collision_objs, body_poses

def transform_world_to_local(world_pos, world_quat_wxyz, base_pos, base_quat_wxyz):
    """Convert world pose to robot's local frame"""
    world_quat_xyzw = np.array([world_quat_wxyz[1], world_quat_wxyz[2], world_quat_wxyz[3], world_quat_wxyz[0]])
    base_quat_xyzw = np.array([base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]])

    base_rot = R.from_quat(base_quat_xyzw)
    base_rot_inv = base_rot.inv()
    
    local_pos = base_rot_inv.apply(world_pos - base_pos)
    local_quat_xyzw = (base_rot_inv * R.from_quat(world_quat_xyzw)).as_quat()
    
    local_quat_wxyz = np.array([local_quat_xyzw[3], local_quat_xyzw[0], local_quat_xyzw[1], local_quat_xyzw[2]])
    
    return local_pos, local_quat_wxyz





def _make_fk(robot, link_name):
    @jax.jit
    def _fk(config):
        return robot.forward_kinematics(config, link_name=link_name)
    return _fk






def main():
    # Load robots
    urdf1 = load_robot_description("ur5_description")
    urdf2 = load_robot_description("ur5_description")
    robot1 = pk.Robot.from_urdf(urdf1)
    robot2 = pk.Robot.from_urdf(urdf2)
    robot_coll1 = RobotCollision.from_urdf(urdf1)
    robot_coll2 = RobotCollision.from_urdf(urdf2)

    # Setup visualization
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=3, height=3, cell_size=0.1)
    
    # Robot bases
    base1 = server.scene.add_transform_controls(
        "/robot1_base", scale=0.1, position=(0.5, -0.5, 0), wxyz=(0, 0, 0, 1)
    )
    base2 = server.scene.add_transform_controls(
        "/robot2_base", scale=0.1, position=(0.5, 0.5, 0), wxyz=(0, 0, 0, 1)
    )
    
    urdf_vis1 = ViserUrdf(server, urdf1, root_node_name="/robot1_base")
    urdf_vis2 = ViserUrdf(server, urdf2, root_node_name="/robot2_base")

    plane_coll = HalfSpace.from_point_and_normal(np.array([0, 0, 0]), np.array([0, 0, 1]))


    # Load target and obstacles from MJCF
    mjcf_collision_objs, body_poses = load_mjcf_to_scene("scene/scene.xml", server)
    target_world_pos_0, target_world_quat_0 = body_poses["target_0"]
    target_world_pos_1, target_world_quat_1 = body_poses["target_1"]

    print("mjcf_collision_objs:", mjcf_collision_objs)
    
    # Visualize target
    server.scene.add_frame(
        "/target_frame_0",
        position=tuple(target_world_pos_0),
        wxyz=tuple(target_world_quat_0),
        axes_length=0.2,
        axes_radius=0.01,
    )

    server.scene.add_frame(
        "/target_frame_1",
        position=tuple(target_world_pos_1),
        wxyz=tuple(target_world_quat_1),
        axes_length=0.2,
        axes_radius=0.01,
    )

    # Define start configuration (UR5 home position)
    start_config = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
    
    # Set initial robot positions
    # urdf_vis1.update_cfg(start_config)
    # urdf_vis2.update_cfg(start_config)
    # time.sleep(1.0)

    # Create JIT-compiled FK functions
    # fk1 = create_fk_function(robot1)
    # fk2 = create_fk_function(robot2)

    # fk1 = _make_fk(robot1, "ee_link")
    # fk2 = _make_fk(robot2, "ee_link")

    
    # Get starting poses from FK
    # start_pos1, start_rot1 = fk1(start_config)
    # start_pos2, start_rot2 = fk2(start_config)
    # start_quat1 = np.array([start_rot1.as_quat()[3], *start_rot1.as_quat()[:3]])
    # start_quat2 = np.array([start_rot2.as_quat()[3], *start_rot2.as_quat()[:3]])

    start_pos1, start_quat1 = [0, 0, 0], [1, 0, 0, 0]
    start_pos2, start_quat2 = [0, 0, 0], [1, 0, 0, 0]

    # Transform target to local frames
    base1_pos = np.array(base1.position)
    base1_quat = np.array(base1.wxyz)
    target_pos1, target_quat1 = transform_world_to_local(
        target_world_pos_0, target_world_quat_0, base1_pos, base1_quat
    )
    
    base2_pos = np.array(base2.position)
    base2_quat = np.array(base2.wxyz)
    target_pos2, target_quat2 = transform_world_to_local(
        target_world_pos_1, target_world_quat_1, base2_pos, base2_quat
    )

    # Generate trajectories
    steps = 100
    positions1, quats1 = generate_trajectory(
        start_pos1, target_pos1, start_quat1, target_quat1, steps
    )
    positions2, quats2 = generate_trajectory(
        start_pos2, target_pos2, start_quat2, target_quat2, steps
    )

    # Main loop
    for i in range(steps):
        # Solve IK for both robots

        world_coll_list = [plane_coll] + mjcf_collision_objs
        # world_coll_list = mjcf_collision_objs

        sol1 = pks.solve_ik_with_collision(
            robot=robot1,
            coll=robot_coll1,
            world_coll_list=world_coll_list,
            target_link_name="ee_link",
            target_position=positions1[i],
            target_wxyz=quats1[i],
        )

        sol2 = pks.solve_ik_with_collision(
            robot=robot2,
            coll=robot_coll2,
            world_coll_list=world_coll_list,
            target_link_name="ee_link",
            target_position=positions2[i],
            target_wxyz=quats2[i],
        )

        if sol1 is None:
            print(f"Step {i}: Robot 1 IK failed! Using previous solution")
            continue
        if sol2 is None:
            print(f"Step {i}: Robot 2 IK failed! Using previous solution")
            continue

        urdf_vis1.update_cfg(sol1)
        urdf_vis2.update_cfg(sol2)
        time.sleep(0.05)

    print("Trajectory complete for both robots!")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()