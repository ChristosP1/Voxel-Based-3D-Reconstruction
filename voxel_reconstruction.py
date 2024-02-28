import numpy as np
import cv2
import os
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


def load_config_info(config_info_path="data/cam", config_input_filename="config.xml"):
    """
    Loads intrinsic (camera matrix, distortion coefficients) and extrinsic (rotation vector, translation vector) camera
    parameters from config file.

    :param config_info_path: config xml file directory path
    :param config_input_filename: config xml file name
    :return: camera matrix
    """
    # Select tags for loaded nodes and their types
    node_tags = ["CameraMatrix", "DistortionCoeffs", "RotationVector", "TranslationVector"]
    node_types = ["mat" for _ in range(len(node_tags))]

    # Load nodes
    nodes = utils.load_xml_nodes(config_info_path, config_input_filename, node_tags, node_types)

    # Parse config
    mtx = nodes.get("CameraMatrix")
    dist = nodes.get("DistortionCoeffs")
    rvecs = nodes.get("RotationVector")
    tvecs = nodes.get("TranslationVector")

    return mtx, dist, rvecs, tvecs


def create_voxel_volume(num_voxels_x=128, num_voxels_y=128, num_voxels_z=128, x_min=-512, x_max=1024, y_min=-1024,
                        y_max=1024, z_min=-2048, z_max=512):
    """
    Creates voxel volume points given dimensions and linear spaces.

    :param num_voxels_x: number of voxels in x range
    :param num_voxels_y: number of voxels in y range
    :param num_voxels_z: number of voxels in z range
    :param x_min: min x for sampling x range
    :param x_max: max x for sampling x range
    :param y_min: min y for sampling y range
    :param y_max: max y for sampling y range
    :param z_min: min z for sampling z range
    :param z_max: max z for sampling z range
    :return: voxel volume points
    """
    # Sample ranges
    x_range = np.linspace(x_min, x_max, num=num_voxels_x)
    y_range = np.linspace(y_min, y_max, num=num_voxels_y)
    z_range = np.linspace(z_min, z_max, num=num_voxels_z)

    # Generate points
    voxel_points = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)

    return voxel_points


def create_lookup_table(voxel_points, num_cameras, cam_input_path="data", config_input_filename="config.xml"):
    """
    Creates lookup table to map 3D voxels to 2D points for a number of cameras.

    :param voxel_points: voxel volume points
    :param num_cameras: number of cameras
    :param cam_input_path: camera root directory path
    :param config_input_filename: config file name (found in respective camera folders in cam_input_path)
    :return: dictionary of projected voxel points for every camera (camera id as key, tuple of voxel position tuple and
             points position tuple as value)
    """
    # Lookup entry for every camera
    lookup_table = {camera: [] for camera in range(1, num_cameras+1)}
    for camera in range(1, num_cameras+1):
        # Load camera parameters
        config_path = os.path.join(cam_input_path, "cam" + str(camera))
        mtx, dist, rvecs, tvecs = load_config_info(config_path, config_input_filename)

        # Project 3D voxel points to image plane and store them
        image_points_voxels, _ = cv2.projectPoints(voxel_points, rvecs, tvecs, mtx, dist)
        for voxel, img_point in zip(voxel_points, image_points_voxels):
            x, y = img_point[0]
            lookup_table[camera].append((tuple(map(int, voxel)), (x, y)))

    return lookup_table


def update_visible_voxels_and_extract_colors(lookup_table, fg_masks, images):
    """
    Updates visibility for voxels by checking if they are visible by a camera view and extracts colors for every camera
    view.

    :param lookup_table: lookup table of projected voxel image points per camera
    :param fg_masks: foreground masks for every camera
    :param images: images to get colors from for every camera
    :return: dictionary of visible corners and whether they are seen by a camera and dictionary of colors per camera
             for the visible voxels
    """
    # Storing voxel points and their colors per camera
    voxels_visible = {}
    voxels_visible_colors = {}

    # Go through voxels of every camera
    for camera, voxel_list in lookup_table.items():
        camera_arr_idx = camera-1
        # Go through image points of every voxel
        for voxel, (x, y) in voxel_list:
            # Check if voxel projection is within image boundaries
            if 0 <= y < fg_masks[camera_arr_idx].shape[0] and 0 <= x < fg_masks[camera_arr_idx].shape[1]:
                # Check if voxel is visible from camera
                if fg_masks[camera_arr_idx][int(y), int(x)] > 0:
                    # Update visibility for current camera
                    if voxel not in voxels_visible:
                        voxels_visible[voxel] = {}
                    voxels_visible[voxel][camera] = True

                    # Store color from frame for current camera
                    color = images[camera_arr_idx][int(y), int(x), :]
                    if voxel not in voxels_visible_colors:
                        voxels_visible_colors[voxel] = {}
                    voxels_visible_colors[voxel][camera] = np.array(color)

    return voxels_visible, voxels_visible_colors


def plot_marching_cubes(voxels_status, rotate=True, plot_output_path="plots",
                        plot_output_filename="marching_cubes.png"):
    """
    Runs marching cubes algorithm on activated voxels and plots results.

    :param voxels_status: 3D array with statuses of voxels (True for ON, False for OFF)
    :param rotate: if True then rotates figure in plot to see the front, otherwise seeing the back of figure
    :param plot_output_path: plot output directory path
    :param plot_output_filename: plot output file name (including extension)
    """
    # Change orientation
    if rotate:
        voxels_status = np.rot90(voxels_status, 2)

    # Run marching cubes
    verts, faces, normals, values = measure.marching_cubes(voxels_status, 0)

    # Plot results from algorithm
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor("k")
    ax.add_collection3d(mesh)

    # Set axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("z-axis")
    ax.set_xlim(0, voxels_status.shape[2])
    ax.set_ylim(0, voxels_status.shape[1])
    ax.set_zlim(0, voxels_status.shape[0])

    # Adjust plot
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(plot_output_path, plot_output_filename))
