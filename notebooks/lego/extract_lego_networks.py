import networkx as nx
import shapely
import numpy as np
import os

LDRAW_PATH = "/Users/glover.co/Downloads/ldraw/parts/"

def read_dat(file, visited=None):
    """
    Recursively reads a DAT file and extracts brick position and bounding box, including subparts.

    Parameters:
        file (str): The path to the DAT file.
        visited (set): Keeps track of visited files to avoid infinite recursion.

    Returns:
        top_vertices (numpy.ndarray): Top surface vertices.
        bottom_vertices (numpy.ndarray): Bottom surface vertices.
        bounding_box (tuple): ((xmin, ymin, zmin), (xmax, ymax, zmax)).
        size (tuple): (width, depth, height) in LDU.
    """
    if visited is None:
        visited = set()

    vertices = []
    faces = []
    edges = []

    if file in visited:
        return np.array([]), np.array([]), ((0, 0, 0), (0, 0, 0)), (0, 0, 0)  # Prevent infinite loops
    visited.add(file)

    if not os.path.exists(file):
        return np.array([]), np.array([]), ((0, 0, 0), (0, 0, 0)), (0, 0, 0)

    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] == '0':  # Ignore comments
                continue

            command_type = parts[0]

            if command_type == '1':  # Subpart reference
                subpart_name = parts[-1]
                # Replace backslashes with forward slashes for compatibility
                subpart_name = subpart_name.replace('\\', '/')
                subpart_file = os.path.join(LDRAW_PATH, subpart_name)

                # Recursively load subpart
                sub_top, sub_bottom, _, _ = read_dat(subpart_file, visited)
                vertices.extend(sub_top.tolist())
                vertices.extend(sub_bottom.tolist())

            elif command_type in {'2', '3', '4'}:  # Edge, Triangle, or Quad
                vertices_in_line = [tuple(map(float, parts[i:i+3])) for i in range(2, len(parts), 3)]
                for v in vertices_in_line:
                    if v not in vertices:
                        vertices.append(v)
                
                if command_type == '2':  # Edge
                    edges.append((vertices_in_line[0], vertices_in_line[1]))
                elif command_type == '3':  # Triangle
                    faces.append(tuple(vertices_in_line[:3]))
                elif command_type == '4':  # Quad
                    faces.append(tuple(vertices_in_line[:4]))

    if not vertices:
        return np.array([]), np.array([]), ((0, 0, 0), (0, 0, 0)), (0, 0, 0)

    # Convert to numpy array
    vertices = np.array(vertices)

    # Compute bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    bounding_box = (tuple(min_coords), tuple(max_coords))
    size = tuple(max_coords - min_coords)  # (width, depth, height)

    
    # Separate top and bottom vertices based on height (Y-axis)
    ymin = min_coords[1]
    ymax = max_coords[1]
    min_idx = np.where(vertices[:, 1] == ymin)[0]
    max_idx = np.where(vertices[:, 1] == ymax)[0]
    top_vertices = vertices[max_idx]
    bottom_vertices = vertices[min_idx]

    return top_vertices, bottom_vertices, bounding_box, size

def apply_rotation(vertices, rotation_matrix, position):
    """
    Rotates and translates the brick's vertices.

    Parameters:
        vertices (list): A list of vertices of the brick.
        rotation_matrix (numpy.ndarray): A 3x3 rotation matrix.
        position (numpy.ndarray): A 1x3 position vector.
    
    Returns:
        numpy.ndarray: The rotated and translated vertices.
    """
    rotated = (rotation_matrix @ vertices.T).T  # Rotate
    return rotated + position  # Translate

def parse_ldr_lines(lines):
    """
    Parses LDraw lines and extracts bricks and submodels.

    Parameters:
        lines (list): List of lines from an LDraw file.

    Returns:
        bricks (list): List of (name, position, rotation_matrix).
        submodels (dict): Dictionary of submodel names -> list of lines.
    """
    bricks = []
    submodels = {}
    current_submodel = None
    submodel_lines = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) <= 1:  # Skip comments and meta commands
            continue
        if parts[0] == '0' and parts[1] == '!LPUB':
            continue
        if parts[:2] == ['0', 'FILE']:  # Start of a new submodel
            if current_submodel:  # Save the previous submodel
                submodels[current_submodel] = submodel_lines
            current_submodel = parts[2]
            submodel_lines = []
            continue
        
        if parts[0] == '0' and parts[1] == 'NOFILE':  # End of submodel
            if current_submodel:
                submodels[current_submodel] = submodel_lines
            current_submodel = None
            continue

        if current_submodel != 'model.ldr' and current_submodel != None:  # Collecting submodel lines
            if parts[0] == '0' and parts[1] == 'STEP':
                continue
            else:
                submodel_lines.append(line)  # Store submodel lines
        
        else:
            if parts[0] == '1':  # Brick definition
                color = parts[1]
                pos = np.array(parts[2:5], dtype=float)
                rot = np.array(parts[5:14], dtype=float).reshape(3, 3)
                name = parts[14][:-4]

                bricks.append((name, pos, rot))

    return bricks, submodels


def apply_transformation(pos, rot, parent_pos, parent_rot):
    """
    Applies a transformation to a position and rotation.
    
    Parameters:
        pos (numpy array): Local position.
        rot (numpy array): Local rotation matrix.
        parent_pos (numpy array): Parent's global position.
        parent_rot (numpy array): Parent's global rotation matrix.
    
    Returns:
        (new_pos, new_rot): Transformed position and rotation matrix.
    """
    new_pos = parent_rot @ pos + parent_pos
    new_rot = parent_rot @ rot
    return new_pos, new_rot


def expand_submodels(bricks, submodels):
    """
    Replaces submodel references with actual bricks.

    Parameters:
        bricks (list): List of (name, position, rotation_matrix).
        submodels (dict): Dictionary of submodel names -> list of lines.

    Returns:
        expanded_bricks (list): Flattened list of all bricks with global positions.
    """
    expanded = []
    def process_brick(name, pos, rot):
        if f'{name}.ldr' in submodels: 
            # If it's a submodel, expand it
            sub_bricks, _ = parse_ldr_lines(submodels[f'{name}.ldr'])  # Get raw bricks inside
            for sub_name, sub_pos, sub_rot in sub_bricks:
                new_pos, new_rot = apply_transformation(sub_pos, sub_rot, pos, rot)
                process_brick(sub_name, new_pos, new_rot)  # Recursive expansion
        else:
            expanded.append((name, pos, rot))  # Normal brick

    for name, pos, rot in bricks:
        process_brick(name, pos, rot)  # Start processing

    return expanded



def read_ldr(file):
    """
    Reads an LDraw .ldr file and returns brick names, positions, and rotation matrices.

    Parameters:
        file (str): Path to the LDraw .ldr file.

    Returns:
        names (list): Brick names.
        positions (list): Brick positions.
        rotations (list): Rotation matrices.
    """
    with open(file, 'r') as f:
        lines = f.readlines()

    bricks, submodels = parse_ldr_lines(lines)
    expanded_bricks = expand_submodels(bricks, submodels)
    names, positions, rotations = zip(*expanded_bricks) if expanded_bricks else ([], [], [])
    return list(names), np.array(positions), np.array(rotations)

def check_contact(box1, box2, min_overlap=0.1):
    """
    Check if two bounding boxes are in contact, i.e., one sits on top of the other.
    
    Parameters:
        box1, box2 (numpy.ndarray): Two bounding boxes, each as a 2x3 array where
                                     the first row is the (xmin, ymin, zmin) and
                                     the second row is the (xmax, ymax, zmax).
    
    Returns:
        bool: True if the two boxes are in contact, False otherwise.
    """
    # Unpack the bounding box coordinates for box1 and box2
    (xmin1, ymin1, zmin1), (xmax1, ymax1, zmax1) = box1
    (xmin2, ymin2, zmin2), (xmax2, ymax2, zmax2) = box2

    # Make shapely boxes
    box1 = shapely.geometry.box(xmin1, zmin1, xmax1, zmax1)
    box2 = shapely.geometry.box(xmin2, zmin2, xmax2, zmax2)

    # Check if the two boxes intersect
    if ymin1 == ymax2 or ymin2 == ymax1:
        intersection_area = box1.intersection(box2).area
        min_area = min(box1.area, box2.area)

        # Ensure the overlap is above a certain threshold
        if min_area > 0:
            if (intersection_area / min_area) >= min_overlap:
                return True
        
    return False

def find_contacting_boxes(bounding_boxes):
    """
    Find all pairs of bounding boxes that are in contact.
    
    Parameters:
        bounding_boxes (list of numpy.ndarray): A list of bounding boxes, each represented by a 2x3 array.
    
    Returns:
        list: A list of pairs of indices where the boxes are in contact.
    """
    contacting_pairs = []
    
    for i in range(len(bounding_boxes)):
        for j in range(i + 1, len(bounding_boxes)):
            if check_contact(bounding_boxes[i], bounding_boxes[j]):
                contacting_pairs.append((i, j))
    
    return contacting_pairs

def create_lego_network(file, with_names=True):
    """
    Create network from LDR lego model.
    This is done by reading the LDR file, extracting the bricks, and 
    finding the contacts between the bricks.

    Parameters:
        file (str): The path to the LDR file.
    
    Returns:
        networkx.Graph: A network representing the LEGO model.
    """
    # Read the LDR file
    names, positions, rotations = read_ldr(file)

    # Create a list of bounding boxes
    bounding_boxes = []
    for name, position, rotation in zip(names, positions, rotations):
        # Get the DAT file for the brick
        dat_file = os.path.join(LDRAW_PATH, f"{name}.dat")
        top_vertices, bottom_vertices, bounding_box, size = read_dat(dat_file)
        
        # Rotate and translate the vertices
        bounding_box = apply_rotation(np.array(bounding_box), rotation, position)
        
        bounding_boxes.append(bounding_box)

    # Find contacting boxes
    contacting_pairs = find_contacting_boxes(bounding_boxes)

    # Create a network
    G = nx.Graph()
    G.add_nodes_from(np.arange(len(names)))
    for i, j in contacting_pairs:
        G.add_edge(i,j)

    # # Add brick name as node attribute
    # for i, name in enumerate(names):
    #     G.nodes[i]['name'] = name
    # Get names in same order as nodes
    unique_names = list(set(names)) 
    num_unique_names = len(unique_names)
    if with_names:
        X = np.zeros((G.number_of_nodes(), num_unique_names))
        for i in range(G.number_of_nodes()):
            X[i, unique_names.index(names[i])] = 1
        return G, X
    return G,names