import cv2
import numpy as np
import pickle
from utils import load_cameras_xml, find_file_paths




# Step 2: Prepare Camera Data
def load_calibration_data(scaling_factor=1):
    cameras_paths = find_file_paths('data', 'config.xml')
    cameras_data = load_cameras_xml(cameras_paths, scaling_factor)
    return cameras_data

# Step 3: Compute 3D World Points
def compute_world_points(volume_center, volume_size, calibration_scale, step_size, chunk_size):
    # Convert lists to NumPy arrays for element-wise operations
    volume_center = np.array(volume_center)
    volume_size = np.array(volume_size)

    # Compute volume extents
    min_extent = volume_center - volume_size / 2.0 * calibration_scale
    max_extent = volume_center + volume_size / 2.0 * calibration_scale

    # Generate grid points
    X = np.arange(min_extent[0], max_extent[0], step_size)
    Y = np.arange(min_extent[1], max_extent[1], step_size)
    Z = np.arange(min_extent[2], max_extent[2], step_size)

    # Create 3D grid (all combinations of X, Y, Z)
    world_points = np.array(np.meshgrid(X, Y, Z)).T.reshape(-1, 3)

    # If needed, split the grid into chunks
    chunks = [world_points[i:i + chunk_size] for i in range(0, len(world_points), chunk_size)]
    
    return chunks


# Step 4: Project 3D Points to 2D Image Planes
def project_3D_points_to_2D(chunk, camera_data):
    """
    Projects 3D world points onto a camera's 2D image plane.

    :param chunk: Chunk of 3D world points.
    :param camera_data: Dictionary containing camera's calibration data.
    :return: Dictionary mapping 2D image pixels to 3D world points.
    """
    mapping = {}

    rotation_vector, translation_vector = camera_data['RotationMatrix'], camera_data['TranslationMatrix']
    camera_matrix, distortion_coeffs = camera_data['CameraMatrix'], camera_data['DistortionCoeffs']

    # Convert rotation matrix to rotation vector
    rotation_vector, _ = cv2.Rodrigues(rotation_vector) if rotation_vector.shape == (3, 3) else (rotation_vector, None)

    # Project the 3D points
    image_points, _ = cv2.projectPoints(np.array(chunk), rotation_vector, translation_vector, camera_matrix, distortion_coeffs)

    # Map the 2D image points to the corresponding 3D world points
    for point_2d, point_3d in zip(image_points, chunk):
        mapping[tuple(point_2d[0])] = point_3d

    return mapping


# Step 5: Create the Lookup Table
def create_lookup_table(chunks, cameras_data):
    lookup_table = {camera_index: {} for camera_index in range(len(cameras_data))}

    for chunk in chunks:
        for camera_index, camera_data in enumerate(cameras_data):
            mapping = project_3D_points_to_2D(chunk, camera_data)
            lookup_table[camera_index].update(mapping)
    
    total_mappings = 0
    for i in range(len(lookup_table)): total_mappings += len(lookup_table[i])
    
    return lookup_table, total_mappings

# Step 6: Handle Segmented Images
def load_segmentation_masks():
    masks_paths = find_file_paths('data', 'mask_MOG.jpg')
    
    masks = []
    for path in masks_paths:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        masks.append(mask)
    return masks

def refine_lookup_table(lookup_table):
    segmentation_masks = load_segmentation_masks()
    refined_lookup = {camera_index: {} for camera_index in lookup_table.keys()}
    
    for camera_index, mappings in lookup_table.items():
        mask = segmentation_masks[camera_index]
        mask_height, mask_width = mask.shape

        for pixel, world_points in mappings.items():
            x, y = int(pixel[0]), int(pixel[1])
            
            # Check if pixel coordinates are within the mask dimensions
            if x >= 0 and x < mask_width and y >= 0 and y < mask_height:
                if mask[y, x] == 255:  
                    refined_lookup[camera_index][pixel] = world_points
                
    total_mappings = 0
    for i in range(len(refined_lookup)): total_mappings += len(refined_lookup[i])
    
    return refined_lookup, total_mappings

# Step 7: Save the Lookup Table
def save_lookup_table(lookup_table, name):
    with open(name, 'wb') as f:
        pickle.dump(lookup_table, f)


# Step 1: Define 3D Space
volume_center = [5, 3.8, 0]
volume_size = [10, 12, 15]
calibration_scale = 115
step_size = 32
chunk_size = 128
scaling_factor = 1/calibration_scale

# Main Execution
if __name__ == '__main__':
    # Execute the steps
    ############################ LOAD CAMERA DATA ############################
    cameras_data = load_calibration_data(scaling_factor)
    print("## Calibration data of the 4 cameras are extracted ##")
    
    ############################ COMPUTE CHUNKS ############################
    chunks = compute_world_points(volume_center, volume_size, calibration_scale, step_size, chunk_size)
    print(f"## World points in chunks are computed. Chunks: {len(chunks)} ##")
    
    ############################ CREATE LOOKUP TABLE ############################
    lookup_table, total_mappings = create_lookup_table(chunks, cameras_data)
    print(f"## Lookup table was created. Total mappings: {total_mappings} ##")
    
    ############################ REFINE LOOKUP TABLE ############################
    refined_lookup_table, total_mappings = refine_lookup_table(lookup_table)
    print(f"## Lookup table was refined. Total mappings: {total_mappings} ##")
    
    ############################ SAVE LOOKUP TABLES ############################
    save_lookup_table(lookup_table, 'lookup_table.pkl')
    save_lookup_table(refined_lookup_table, 'refined_lookup_table.pkl')
    print("*** Both the original and refined lookup tables are saved ***")
    
    exit(0)
    
    
    
    