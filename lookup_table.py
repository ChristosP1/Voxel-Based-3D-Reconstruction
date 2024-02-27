import cv2
import numpy as np
import ctypes
import os
import pickle
from utils import load_cameras_xml, find_file_paths

def compute_world_points(vol_center, vol_size, calibration_scale, step, chunk_size):
    """
    This function computes the real world coordinates in mm for each voxel point
    :param chunk_size: The size of a volume chunk
    :return: The real world points and the number of chunks per dimension
    """

    # Compute volume location and ranges
    center = np.array(vol_center, dtype=np.float32) * calibration_scale
    size = np.array(vol_size, dtype=np.float32) * calibration_scale

    sx, sy, _ = center - size / 2.0
    ex, ey, _ = center + size / 2.0
    sz, ez = center[2], size[2]

    samples_size = (size / step).astype(ctypes.c_int32)

    print("The requested look-up table will have size", samples_size)

    # Sample points
    X = np.linspace(sx, ex, samples_size[0], dtype=np.float32)
    Y = np.linspace(sy, ey, samples_size[1], dtype=np.float32)
    Z = np.linspace(sz, ez, samples_size[2], dtype=np.float32)

    chunks = np.ceil(samples_size / chunk_size).astype(np.int32)

    return X, Y, Z, chunks

def compute_voxels_chunk(lx, ly, lz, cx, cy, cz, size):
    """
    Computes a chunk of real-world voxel coordinates.
    :param lx, ly, lz: Lists of real-world points along the x, y, z axes.
    :param cx, cy, cz: Chunk indices along the x, y, z axes.
    :param size: Size of the chunk.
    :return: A list of voxel coordinates in the chunk.
    """
    llx = lx[(cx * size):((cx + 1) * size)]
    lly = ly[(cy * size):((cy + 1) * size)]
    llz = lz[(cz * size):((cz + 1) * size)]

    chunk = []
    for x in llx:
        for y in lly:
            for z in llz:
                chunk.append([x, y, z])
    
    return chunk

def project_points(chunk, camera):
    """
    Projects 3D world points into 2D image points for a given camera.
    :param chunk: List of 3D world points (voxels).
    :param camera: Dictionary containing camera parameters (intrinsics and extrinsics).
    :return: List of 2D image points corresponding to the 3D world points.
    """
    # Extract camera parameters
    rotation_vector = camera["RotationMatrix"]
    translation_vector = camera["TranslationMatrix"]
    camera_matrix = camera["CameraMatrix"]
    distortion_coeffs = camera["DistortionCoeffs"]

    # Convert chunk to a NumPy array
    chunk_array = np.array(chunk, dtype=np.float32)

    # Project 3D points to 2D using OpenCV's projectPoints function
    image_points, _ = cv2.projectPoints(chunk_array, rotation_vector, translation_vector, camera_matrix, distortion_coeffs)

    return image_points

def is_valid_pixel(pixel, shape):
    return not np.any(np.isinf(pixel)) and \
        -1 < pixel[0] < shape[1] and \
        -1 < pixel[1] < shape[0]


# Determine the coordinates and scales of the 3D reconstruction
volumn_center = [5, 3.8, 0]
volumn_size = [10, 12, 15]
calibration_scale = 115
step_size = 64   
frame_shape = [644,486]
# Determine chunk size for multithreading
chunk_size = 128
# Cameras
cameras = None
X, Y, Z, chunks_count = None, None, None, None
sort_by_camera_distance = 1
scaling_factor = 1/calibration_scale

lookup_save_path = ''
cameras_path = find_file_paths('data', 'config.xml')
print(cameras_path)


def main_process_exclusive_work():
    global volumn_center, volumn_size, calibration_scale, step_size, chunk_size, cameras, X, Y, Z
    
    # Load global data
    cameras = load_cameras_xml(cameras_path, scaling_factor)
    X, Y, Z, chunks_count = compute_world_points(volumn_center, volumn_size, calibration_scale, step_size, chunk_size)

    lookup = {i: dict() for i in range(len(cameras))}

    print("Starting look-up table computation!")
    print("Total number of chunks: ", chunks_count[0] * chunks_count[1] * chunks_count[2])

    for cx in range(chunks_count[0]):
        for cy in range(chunks_count[1]):
            for cz in range(chunks_count[2]):
                # Process the chunk for each camera
                for camera_index, camera in enumerate(cameras):
                    chunk = compute_voxels_chunk(X, Y, Z, cx, cy, cz, chunk_size)
                    pixels = project_points(chunk, camera)

                    # Link 3D voxel to a pixel for a given camera
                    for i, pixel in enumerate(pixels):
                        pixel_t = tuple(pixel[0])
                        if is_valid_pixel(pixel_t, frame_shape):
                            world_voxel = (cx * chunk_size + i % chunk_size,
                                           cy * chunk_size + (i // chunk_size) % chunk_size,
                                           cz * chunk_size + i // (chunk_size ** 2))
                            lookup[camera_index].setdefault(pixel_t, []).append(world_voxel)

                    # Optionally sort by camera distance
                    if sort_by_camera_distance:
                        for pixel in lookup[camera_index].keys():
                            lookup[camera_index][pixel] = sorted(lookup[camera_index][pixel],
                                                                 key=lambda v: np.sum(np.square(
                                                                     np.array(v) - camera["RescaledWorldPosition"])))

    # Save the look-up table in a binary file
    for i in lookup:
        print(i)
    pickle.dump(lookup, open(os.path.join(lookup_save_path, "lookup"), "wb"))

if __name__ == '__main__':
    main_process_exclusive_work()


