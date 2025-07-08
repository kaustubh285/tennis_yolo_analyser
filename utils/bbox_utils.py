def get_bbox_center(bbox):
    """
    Calculate the center of a bounding box.

    Args:
        bbox (list): Bounding box coordinates in the format [x1, y1, x2, y2].

    Returns:
        tuple: Center coordinates (cx, cy).
    """
    x1, y1, x2, y2 = bbox
    cx = int(x1 + x2) / 2
    cy = int(y1 + y2) / 2
    return cx, cy


def measure_distance(pt1, pt2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        pt1 (tuple): First point (x1, y1).
        pt2 (tuple): Second point (x2, y2).

    Returns:
        float: Distance between the two points.
    """
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
