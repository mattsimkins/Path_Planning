import math

#Constant that scales a grid spacing to the height of a triangle
Y_FACT = math.sqrt(3)/2


def coord_from_ind(ind, node_sp):
    
    '''Calculates a grid coordinate in grid space.
    
    Args:
        ind: A list of 2 integers that correspond to grid indices,
        List[int, int]
        
        node_sp: Spacing of grid nodes, Float
        
    Returns:

        coord: Tuple(Float, Float)
    '''
    
    if (ind[1] % 2) == 0: x = ind[0]*node_sp
    else: x = ind[0]*node_sp + node_sp/2
    y = ind[1]*node_sp*Y_FACT
    coord = (x, y)
    return coord


def dist2node(loc, node_ind, node_sp):
    
    '''Calculates the distance between a trajectory coordinate and a grid
    node that is specified by its indices.
    
    Args:

        loc: Cartesian coordinate, x and y, Tuple(Float, Float)
        
        node_ind: Indices for node of interest, List[Int, Int]
        
        node_sp: Node spacing of grid, Float
    
    Returns:
        
        dist: Positive distance from coordinate to node, Float
    '''
    
    node_loc = coord_from_ind(node_ind, node_sp)
    dist = math.sqrt((loc[0] - node_loc[0])**2 + (loc[1] - node_loc[1])**2)
    return dist


def path_length(traj):
    
    '''Calculates the total path length of a trajectory by summing the
    distance between sucessive coordinates.
    
    Args:
    
        traj: A trajectory, List[Tuple(Float, Float), Tuple(Float, Float),
        ...]
    
    Returns:

        overall_length: Total trajectory length, Float
    '''
    
    last_coord = traj[0]
    overall_length = 0
    for coord in traj:
        dist = math.sqrt((coord[0]-last_coord[0])**2 +\
            (coord[1]-last_coord[1])**2)
        overall_length = overall_length + dist
        last_coord = coord
    return overall_length


def find_index_closest(loc, node_sp):
    
    '''Determines the closest closest grid node given a coordinate.
    
    Args:

        loc: A cartesian coordinate, Tuple(Float, Float)
    
        node_sp: Node spacing of grid, Float
        
    Returns:

        indices: The indices of the closest node, [int, int]
            
    '''
    
    j_closest = round(loc[1]/(node_sp*Y_FACT))
    
    #Determine if closest node is on a shifted row
    if (j_closest % 2) == 0: i_closest = round(loc[0]/node_sp)
    else: i_closest = round(loc[0]/node_sp-1/2)
    indices = (i_closest, j_closest)
    return indices


def find_trident(path_loc, node_sp):
    
    '''Finds the trident of grid node given a trajectroy coordinate. In
    this context, a trident is defined as 3 neighboring nodes that form an
    equallateral triangle. Given the grid geometry, the triangle apex could
    point up or down. The grid geometry is alluded to below with "*" 
    representing nodes.
    
    --- * ---- * ---- * --
      /   \  /   \  /   \
    * ---- * ---- * ---- *
      \   /  \   /  \   /
    --- * ---- * ---- * --
      /   \  /   \  /   \
    * ---- * ---- * ---- *
        
    The trident is the triangle of nodes that enclose the coordinate. The 
    nodes along the tridents horizontal axes are defined as being "left" and
    "rigth". The node at the apex is defined as "center", and the center node
    could be located above or below the left and right nodes.
    
    Args:

        path_loc: Cartesian coordinate within grid space, Tuple(Float, Float)
        
        node_sp: Node spacing of grid, Float
    
    Returns: 

        left_ind: Indices of the left node, [int, int, int]
        
        right_ind: Indices of the right node, [int, int, int]
        
        center_ind: Indices of the center node, [int, int, int]
    '''
    
    #Setup unit vectors for dotting into
    unit_vecs = [(-1, 0), (-0.5, Y_FACT), (0.5, Y_FACT), \
        (1, 0), (0.5, -Y_FACT), (-0.5, -Y_FACT)]
    cls_ind = find_index_closest(path_loc, node_sp)

    #Determine if closest node is on a shifted row in x
    if (cls_ind[1] % 2) == 0: cls_shift = False
    else: cls_shift = True      
        
    #Define 6 candidate quadrants to find the locations of the closest nodes
    dot_dict = {
        "dot_largest": 0,
        "index_largest": 0,
        "dot_2nd_largest": 0,
        "index_2nd_largest": 0
        }
    
    #Determine 3 closest nodes (triangle) using dot product
    coord_closest = coord_from_ind(cls_ind, node_sp)
    for i in range(6):
        dot_check = (path_loc[0]-coord_closest[0])*unit_vecs[i][0] +\
            (path_loc[1] - coord_closest[1])*unit_vecs[i][1]
        if dot_check > dot_dict["dot_2nd_largest"]:
            dot_dict["dot_2nd_largest"] = dot_check
            dot_dict["index_2nd_largest"] = i
        if dot_dict["dot_2nd_largest"] > dot_dict["dot_largest"]:
            dot_largest = dot_dict["dot_2nd_largest"]
            unit_vec_largest = dot_dict["index_2nd_largest"]
            dot_dict["dot_2nd_largest"] = dot_dict["dot_largest"]
            dot_dict["index_2nd_largest"] = dot_dict["index_largest"]
            dot_dict["dot_largest"] = dot_largest
            dot_dict["index_largest"] = unit_vec_largest
     
    # All quadrants equally valid, arbitrarily picking quadrant I
    if (dot_dict["index_largest"] == 0 and\
        dot_dict["index_2nd_largest"] == 0):
        dot_dict["index_2nd_largest"] = 1
    
    #Define quadrant I based on dot product
    if (dot_dict["index_largest"] == 0 and\
        dot_dict["index_2nd_largest"] == 1) or\
            (dot_dict["index_largest"] == 1 and\
                dot_dict["index_2nd_largest"] == 0):
        i_left = cls_ind[0] - 1
        j_left = cls_ind[1]
        i_right = cls_ind[0]
        if cls_shift: i_center = cls_ind[0]
        else: i_center = cls_ind[0] - 1
        j_center = cls_ind[1] + 1

    #Define quadrant II based on dot product
    if (dot_dict["index_largest"] == 1 and\
        dot_dict["index_2nd_largest"] == 2) or\
        (dot_dict["index_largest"] == 2 and\
            dot_dict["index_2nd_largest"] == 1):
        if cls_shift: i_left = cls_ind[0]
        else: i_left = cls_ind[0] - 1
        j_left = cls_ind[1] + 1
        i_right = i_left + 1
        i_center = cls_ind[0]
        j_center = cls_ind[1]

    #Define quadrant III based on dot product
    if (dot_dict["index_largest"] == 2 and\
        dot_dict["index_2nd_largest"] == 3) or\
        (dot_dict["index_largest"] == 3 and\
            dot_dict["index_2nd_largest"] == 2):
        i_left = cls_ind[0]
        j_left = cls_ind[1]
        i_right = cls_ind[0] + 1
        if cls_shift:
            i_center = cls_ind[0] + 1
        else: i_center = cls_ind[0]
        j_center = cls_ind[1] + 1
        
    #Define quadrant IV based on dot product
    if (dot_dict["index_largest"] == 3 and\
        dot_dict["index_2nd_largest"] == 4) or\
        (dot_dict["index_largest"] == 4 and\
            dot_dict["index_2nd_largest"] == 3):
        i_left = cls_ind[0]
        j_left = cls_ind[1]
        i_right = cls_ind[0] + 1
        if cls_shift: i_center = cls_ind[0] + 1
        else: i_center = cls_ind[0]
        j_center = cls_ind[1] - 1
    
    #Define quadrant V based on dot product    
    if (dot_dict["index_largest"] == 4 and\
        dot_dict["index_2nd_largest"] == 5) or\
        (dot_dict["index_largest"] == 5 and\
            dot_dict["index_2nd_largest"] == 4):
        if cls_shift:
            i_left = cls_ind[0]
        else: i_left = cls_ind[0] - 1
        j_left = cls_ind[1] - 1
        i_right = i_left + 1
        i_center = cls_ind[0]
        j_center = cls_ind[1]
        
    #Define quadrant VI based on dot product    
    if (dot_dict["index_largest"] == 5 and\
        dot_dict["index_2nd_largest"] == 0) or\
        (dot_dict["index_largest"] == 0 and\
            dot_dict["index_2nd_largest"] == 5):
        i_left = cls_ind[0] - 1
        j_left = cls_ind[1]
        i_right = cls_ind[0]
        if cls_shift:
            i_center = cls_ind[0]
        else: i_center = cls_ind[0] - 1
        j_center = cls_ind[1] - 1
    left_ind = (i_left, j_left)
    right_ind = (i_right, j_left)
    center_ind = (i_center, j_center)
    
    return (left_ind, right_ind, center_ind)


def traj_metrics(traj):
    
    '''Trajectory metrics are collected for the calculated path operation, 
    see av_trav(). They are intended to prevent that calculation from 
    generating a trajectroy that is too long, or from running indefinitely.
    As that av_trav() proceeds, calculated metrics are compared against
    metrics collected during training.
       
        Args:

            traj: A trajectory from either task space or grid space, List[
                Tuple(Float, Float), Tuple(Float, Float), ...]
                
        Returns:
        
            shortest_segment: Smallest distance between an two successive
            coordinates in a trajectory, Float
            
            coord_count: The number of coordinates in the trajectroy, Int
            
            overall_length: The sum of distances between successive nodes,
            Float 
    '''
    
    #Number of coordinates in trajectory
    coord_count = len(traj)
    
    #Find shortest segment
    shortest_segment = None
    for i in range(coord_count - 1):
        seg_length = math.sqrt((traj[i][0] - traj[i+1][0])**2 + (traj[i][1] -\
            traj[i+1][1])**2)
        if shortest_segment == None:
            shortest_segment = seg_length
        if seg_length < shortest_segment:
            shortest_segment = seg_length
    
    #find trajectory length
    last_coord = traj[0]
    overall_length = 0
    for coord in traj:
        dist = math.sqrt((coord[0]-last_coord[0])**2 +\
            (coord[1]-last_coord[1])**2)
        overall_length = overall_length + dist
        last_coord = coord
    
    return shortest_segment, coord_count, overall_length


if __name__ == "__main__":
    pass