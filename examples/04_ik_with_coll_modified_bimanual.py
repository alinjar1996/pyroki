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


def approximate_box_with_spheres(center, half_extents):
    """
    Approximate a box by spheres: one at center + spheres at corners.
    Returns a list of Sphere collision objects.
    """
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
    """
    Load MJCF XML, create trimesh visual meshes with correct transforms and colors,
    approximate box collisions with spheres, and return all collision objects.
    """
    root = ET.parse(xml_path).getroot()
    collision_objs = []

    for body in root.findall('body'):
        body_pos = np.array([float(x) for x in body.get('pos', '0 0 0').split()])
        quat_wxyz = np.array([float(x) for x in body.get('quat', '1 0 0 0').split()])
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rot = R.from_quat(quat_xyzw)
        rot_mat = rot.as_matrix()

        for geom in body.findall('geom'):
            geom_pos = np.array([float(x) for x in geom.get('pos', '0 0 0').split()])
            rgba = [float(x) for x in geom.get('rgba', '1 1 1 1').split()]
            size = np.array([float(x) for x in geom.get('size', '0 0 0').split()])
            geom_type = geom.get('type', 'box')

            abs_pos = body_pos + rot.apply(geom_pos)
            mesh = None

            if geom_type == 'box':
                extents = 2 * size  # full box size
                mesh = creation.box(extents=extents)
                collision_objs.extend(approximate_box_with_spheres(abs_pos, size))
            elif geom_type == 'sphere':
                radius = size[0]
                mesh = creation.icosphere(radius=radius)
                collision_objs.append(Sphere.from_center_and_radius(abs_pos, radius))
            else:
                continue  # skip unknown types

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

    return collision_objs


def main():
    urdf1 = load_robot_description("ur5_description")
    urdf2 = load_robot_description("ur5_description")

    robot1 = pk.Robot.from_urdf(urdf1)
    robot2 = pk.Robot.from_urdf(urdf2)

    robot_coll1 = RobotCollision.from_urdf(urdf1)
    robot_coll2 = RobotCollision.from_urdf(urdf2)

    target_link1 = "ee_link"
    target_link2 = "ee_link"

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=3, height=3, cell_size=0.1)

    base1 = server.scene.add_transform_controls(
        "/robot1_base", scale=0.1, position=(0, -0.2, 0), wxyz=(0, 0, 0, 1)
    )
    base2 = server.scene.add_transform_controls(
        "/robot2_base", scale=0.1, position=(0.7, 0.2, 0), wxyz=(0, 0, 0, 1)
    )

    urdf_vis1 = ViserUrdf(server, urdf1, root_node_name="/robot1_base")
    urdf_vis2 = ViserUrdf(server, urdf2, root_node_name="/robot2_base")

    plane_coll = HalfSpace.from_point_and_normal(np.array([0, 0, 0]), np.array([0, 0, 1]))
    sphere_center = np.array([0.2, 0.2, 0.2])
    sphere_radius = 0.1
    sphere_coll = Sphere.from_center_and_radius(sphere_center, sphere_radius)

    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.2, position=tuple(sphere_center), wxyz=(0, 0, 0, 1)
    )
    server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=sphere_coll.to_trimesh())

    mjcf_collision_objs = load_mjcf_to_scene("scene/scene.xml", server)

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    steps = 900

    start_pos1, end_pos1 = np.array([0.5, 0.0, 0.3]), np.array([0.3, 0.3, 0.0])
    start_quat1, end_quat1 = np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0])

    start_pos2, end_pos2 = np.array([0.5, 0.0, 0.3]), np.array([0.3, -0.3, 0.0])
    start_quat2, end_quat2 = np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0])

    positions1, quats1 = generate_trajectory(start_pos1, end_pos1, start_quat1, end_quat1, steps)
    positions2, quats2 = generate_trajectory(start_pos2, end_pos2, start_quat2, end_quat2, steps)

    for i in range(steps):
        sphere_coll_curr = sphere_coll.transform_from_wxyz_position(
            wxyz=np.array(sphere_handle.wxyz),
            position=np.array(sphere_handle.position),
        )

        world_coll_list = [plane_coll, sphere_coll_curr] + mjcf_collision_objs

        start_time = time.time()

        sol1 = pks.solve_ik_with_collision(
            robot=robot1,
            coll=robot_coll1,
            world_coll_list=world_coll_list,
            target_link_name=target_link1,
            target_position=positions1[i],
            target_wxyz=quats1[i],
        )

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

        urdf_vis1.update_cfg(sol1)
        urdf_vis2.update_cfg(sol2)

        time.sleep(0.02)  # ~50Hz smooth animation

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
