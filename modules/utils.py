import numpy as np
import cv2
# import utils


def interpolate_points_from_manual_corners(corners, chessboard_shape, use_perspective_transform=True):
    """
    Interpolates chessboard points from the 4 manual corners given by the user. Function supports flat interpolation
    and interpolation using perspective transform for enhanced accuracy (use_perspective_transform parameter).

    :param corners: array of 4 corner points ([x, y])
    :param chessboard_shape: chessboard number of intersection points vertically and horizontally (vertical, horizontal)
    :param use_perspective_transform: interpolates using homogenous coordinates and perspective transform for enhanced
                                      accuracy if set to True, otherwise uses flat interpolation
    :return: returns array of 2D interpolated image points or None if wrong number of corners given (unequal to 4)
    """
    # if len(corners) != 4:
    #     utils.show_warning("incorrect_num_corners")
    #     return None

    # Sort corners to (top-left, top-right, bottom-right, bottom-left)
    # corners = sort_corners_clockwise(corners, origin="top-left")

    corners = np.array(corners, dtype=np.float32)
    # Calculate the maximum width and height
    max_width = max(np.linalg.norm(corners[0] - corners[1]), np.linalg.norm(corners[3] - corners[2]))
    max_height = max(np.linalg.norm(corners[1] - corners[2]), np.linalg.norm(corners[3] - corners[0]))
    print("Max width:", max_width)
    print("Max height:", max_height)

    # Horizontal and vertical step calculation using chessboard shape
    horizontal_step = max_width / (chessboard_shape[1] - 1)
    vertical_step = max_height / (chessboard_shape[0] - 1)

    interpolated_row = []
    interpolated_points = []
    # Perform perspective transform for accuracy improvement
    if use_perspective_transform:
        # Use maximum width and height to form destination coordinates for perspective transform
        dest_corners = np.float32([[0, 0], [max_width - 1, 0],
                                   [max_width - 1, max_height - 1], [0, max_height - 1]])
        p_matrix = cv2.getPerspectiveTransform(corners, dest_corners)

        # Get inverse matrix for projecting points from the transformed space back to the original image space
        inverted_p_matrix = np.linalg.inv(p_matrix)

        # Compute each projected point
        for y in range(0, chessboard_shape[0]):
            for x in range(0, chessboard_shape[1]):
                # Calculate the position of the current point relative to the grid using homogenous coordinates
                point = np.array([x * horizontal_step, y * vertical_step, 1])

                # Multiply with inverse matrix to project point from transformed space back to original image space
                point = np.matmul(inverted_p_matrix, point)

                # Divide point by its Z
                point /= point[2]

                # Append the X and Y of point to the list of interpolated points in row
                interpolated_row.append(point[:2])
            # Append interpolated points in row to interpolated points
            interpolated_points.append(interpolated_row)
            interpolated_row = []
    # Flat interpolation
    else:
        for y in range(0, chessboard_shape[0]):
            for x in range(0, chessboard_shape[1]):
                # Calculate the position of the current point relative to the grid
                point = np.array([x * horizontal_step, y * vertical_step])

                # Interpolate the position of the current point between the known corners
                point = corners[0] + point

                # Append the point to the list of interpolated points in row
                interpolated_row.append(point)
            # Append interpolated points in row to interpolated points
            interpolated_points.append(interpolated_row)
            interpolated_row = []

    # If change_point_order is True then point order will start from bottom-left and end on top-right
    # moving through rows before changing column
    # if False then point order will start at top-left and end on bottom-right
    # moving through columns before changing row as already saved
    interpolated_points = np.array(interpolated_points, dtype="float32")
    if chessboard_shape[0] > chessboard_shape[1]:
        interpolated_points = np.flip(interpolated_points, axis=0)
        interpolated_points = np.transpose(interpolated_points, (1, 0, 2))

    # Return (MxN, 1, 2) array to match automatic corner detection output
    return np.reshape(interpolated_points, (-1, 1, 2))