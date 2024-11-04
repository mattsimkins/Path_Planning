import math
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

#Constant that scales a grid spacing to the height of a triangle
Y_FACT = math.sqrt(3)/2


def open_model(file_path):
    
    '''Opens a model with the file name "model.json". The model is assumed
    to have been generated by the caller.
        
    Args:
    
        file_path: Relative path to data directory, String
        
    Returns:
    
        model: Includes a dictionary that should contains parameter
        
        specified by the caller, Dict.
    '''
    
    try:
        with open(file_path + "model.json", "r") as f:
            model = json.load(f)
        #Convert grid from json to numpy array
        grid2np = np.array(model["grid"])
        model["grid"] = grid2np
        return model
    except FileNotFoundError:
        return None


def read_traj(file_path, file_name):
    
    '''Reads in trajectories for training. Trajectories are structured as
    two, space separated floating point values for x and y respectivel,
    followed by an end line. This function parces characters and formats
    them for use by PVF.
    
    Args:
    
        file_path: Gives the relative path to the directory containing
        trajectory files for training, String.
        
        file_name: Gives the trajectory name for the .txt file being
        read in as a trajectory, String.
    
    Returns:

        traj: A data type structured for use by PVF, List[Tuple(Float, Float),
        Tuple(Float, Float), Tuple(Float, Float), ...].
    '''
    
    try:
        traj_file = open(file_path+"/"+file_name, "r")
        traj = []
        content = traj_file.readlines()
        for line_char in content:
            x_done = False
            x_value = ""
            y_value = ""
            for one_char in line_char:
                if one_char == " ":
                    x_done = True
                    pass
                if x_done == False:
                    x_value = x_value + one_char
                if x_done == True:
                    y_value = y_value + one_char
            traj.append((float(x_value), float(y_value)))
            traj_file.close()
    
    except FileNotFoundError:
        print("Trajectory file not found!")
        return None

    except ValueError:
            print(('Trajectory not correctly formatted and will be ignored.'
                   'Should be of the form:\n<x1> <y1>\n<x2> <y2>\n...'))
            return None
        
    return traj


def convert_traj_ts2gs(traj, node_spacing, extents = None,\
    shift2traj_coord = None):
    '''Recieves a trajectory in task space and converts it to a
    a trajectory for use in grid space. Because grid nodes must span grid
    grid space, using an indefinite grid space necessarily implies an infinite
    quantity of nodes. Therefore, grid space must be bounded by extents. This
    is accomplished in one of two ways. Either the user can specify bound
    explicitly, or they are calculated automatically. Automatic extents
    look for the maximum and minimum values in x and y of the trajectory
    being passed, and add some "padding". Additionally, the grid space frame
    has its lower extents at (0, 0), while the trajectory being passed could
    include x and y values < 0. Therefore, this function also provides a
    shift vector that translates grid space trajectories back to task space.
    
    Args:
    
        traj: A task space trajectory, List[Tuple(Float, Float), Tuple(Float,
        Float), ...].
        
        node_spacing: Spacing between nodes along edges of an equalateral
        triangle, Float.
        
        extents: Manually defined coordinate extents in task space given by 
        min x, max x, min y, max y respectively. If None the extents are
        assigned automatically, List[Float, Float, Float, Float].
        
        shift2traj_coord: If passed a shift is passed a pre-existing model was
        loaded and this function is simply using the shift and extents being
        passed to verify that the current training trajectory is in bounds. If
        None is passed the function will calculate one along with new extents,
        List[Float, Float].
        
    Returns:
    
        shifted_traj:
        
        grid_cart_extents: Grid space extents given task space trajectory size
        or task space coordinate frame extents. Values are given as x-min,
        x-max, y-min, and y-max respectively in the list List[Float, Float,
        Float, Float].
        
        traj_shift_gs2ts: The trajectory shift that was required to convert a
        trajectory from grid space back to task space. List[Float, Float].
    '''
       
    #initialize with valid guesses
    min_x = traj[0][0]
    min_y = traj[0][1]
    max_x = traj[0][0]
    max_y = traj[0][1]
    
    #find most extreme values of trajectory
    for coord in traj:
        if coord[0] < min_x:
            min_x = coord[0]
        if coord[0] > max_x:
            max_x = coord[0]
        if coord[1] < min_y:
            min_y = coord[1]
        if coord[1] > max_y:
            max_y = coord[1]

    #construct coordinate extents for grid space coordinate frame
    grid_cart_extents = [None, None]
    traj_shift_gs2ts = [None, None]
    
    #automatic assignment for extents
    padding = 3 #add padding for coordinate frame
    grid_cart_extents[0] = max_x - min_x + 2*padding*node_spacing
    grid_cart_extents[1] = max_y - min_y + 2*padding*node_spacing*Y_FACT
    
    #automatic assignment for shift
    if shift2traj_coord == None:
        traj_shift_gs2ts[0] = min_x - padding*node_spacing
        traj_shift_gs2ts[1] = min_y - padding*node_spacing*Y_FACT
    
    #If extents are not passed it is automatically assignment
    if extents != None:
        diff_x = extents[1] - extents[0]
        diff_y = extents[3] - extents[2]
        if diff_x < grid_cart_extents[0] or diff_y < grid_cart_extents[1]:            
            print(("Trajectory exceeds given extents. Consider retraining usi"
"ng larger exents than:\nmin x<{}\nmax x>{}\nmin y<{}\nmax y>{}, or do not tr"
"ain with this trajectory\n"
.format(extents[0], extents[1], extents[2], extents[3])))
            return None
        
    #for user assigned extents, not using automatic extents assignment
    if extents != None:
        grid_cart_extents[0] = extents[1] - extents[0]
        grid_cart_extents[1] = extents[3] - extents[2]
        traj_shift_gs2ts[0] = extents[0]
        traj_shift_gs2ts[1] = extents[2]
    
    #function is only being used to get a shifted trajectory
    if shift2traj_coord != None:
        traj_shift_gs2ts[0] = shift2traj_coord[0]
        traj_shift_gs2ts[1] = shift2traj_coord[1]
        grid_cart_extents = None
        shift2traj_coord = None
    
    shifted_traj = [] #shift coordinates for used with the model
    for coord in traj:
        shifted_grid_coord = (coord[0] - traj_shift_gs2ts[0], coord[1] -\
            traj_shift_gs2ts[1])
        
        #shifted trajectory to grid coordinates
        shifted_traj.append(shifted_grid_coord)
    return shifted_traj, grid_cart_extents, traj_shift_gs2ts


def check_extents(traj, extents):
    
    '''Consecutive duplicate coordinates in a trajectory are not permitted.
    These can be thought as stalls in the trajectory whereby there are 2 or
    more of the same coordinates. These are removed. Additionally, this
    function checks that the coordinate frame extents are not exceeded.
    
    Args:

        extents: x-min, x-max, y-min, and y-max, List[Float, Float, Float,
        Float].
        
        traj: Trajectory possibly containing consecutive, duplicate
        coordinates, or coordinates that exceed frame extents,
        List[Tuple(Float, Float), Tuple(Float, Float), ...].
                
    Returns:
        
        traj_fixed: Trajectory with duplicates removed, if any, 
        List[Tuple(Float, Float), Tuple(Float, Float), ...].
        
        None: Returned if coordinate frame extents are breached.
    '''
    
    #Check for correct data types and format
    if traj == None or isinstance(traj, list) == False:
        print("Trajectory provided is not a list.")
        return None
    duplicates_index_list = []
    for i in range(len(traj) - 1):
        if traj[i][0] == traj[i+1][0] and traj[i][1] == traj[i+1][1]:
            duplicates_index_list.insert(0, i+1)
        
        #check if coordinates exceed gs extents
        error_msg = "Consider retraining model with larger extents,\nor\
            excluding the most recent trajectory from training.\n"
        if extents != None:
            if traj[i][0] < 0:
                print("Error: tajectory crosses y-axis")
                print(error_msg)
                return None
            if traj[i][1] < 0:
                print("Error: tajectory crosses x-axis")
                print(error_msg)
                return None
            if traj[i][0] > extents[0]:
                print("Error: tajectory exceeds x-axis limit")
                print(error_msg)
                return None
            if traj[i][1] > extents[1]:
                print("Error: tajectory exceeds x-axis limit")
                print(error_msg)
                return None
            
        #cover the case for reaching last coordinate in trajectory
        if i == len(traj)-1:
            if traj[i+1][0] < 0:
                print("Error: tajectory crosses y-axis")
                print(error_msg)
                return None
            if traj[i+1][1] < 0:
                print("Error: tajectory crosses x-axis")
                print(error_msg)
                return None
            if traj[i+1][0] > extents[0]:
                print("Error: tajectory exceeds x-axis limit")
                print(error_msg)
                return None
            if traj[i+1][0] > extents[0]:
                print("Error: tajectory exceeds x-axis limit")
                print(error_msg)
                return None
    
    traj_fixed = traj                    
    if len(duplicates_index_list) != 0:
        print("Trajectory contains", len(duplicates_index_list),\
            "duplicate coordinates that are being removed\n")
        for index in duplicates_index_list:
            removed_coord = traj_fixed.pop(index)
            print("Coordinate", removed_coord, 'was removed.')
    return traj_fixed


def shift_traj(traj, shift):
    
    '''There are two coordinate frames associated with PVF. The first is "task
    space" (TS), the second is "grid space" (GS). GS has its origin at (0,0)
    while TS could have an origin with x or y located at a negative value.
    If a trajectory is calculated in GS it must be shifted to be meaningfull
    in TS. Likewise, if a training trajectory is used to train a model in GS
    it must first be shifted. This function takes a trajectory and shifts it
    by the caller's specified amount.
    
    Args:

        Traj: The trajectory that needs to be shifted, List[tuple(Float,
        Float), tuple(Float, Float), ...].
        
        Shift: A 2D vector specifying the amount of shift desired,
        List[Float, Float].
                
    Returns:
        
        traj_shifted: The shifted tajectory. All points along the trajectory
        are displaced by the shift vector, List[tuple(Float, Float), 
        tuple(Float, Float), ...].
    '''
    
    traj_shifted = []
    for c in traj:
        coord = (c[0] + shift[0], c[1] + shift[1])
        traj_shifted.append(coord)
    return traj_shifted

    
def find_shortest_seg(traj):
    '''Finds the shortest segment in a trajectory. When pseudo average
    trajectories are calculated it can calculate very small segements
    towards the end of the trajectory. This function is usefull for
    preventing the algorithm from calculating extraneous small tails
    when grid nodes reach the end of the trajectories they were trained on.
    Additionally, it helps check if a node spacing is much smaller than
    the smallest segment length so that the caller is warned that the node
    spacing is smaller than necessary for a given training trajectory.
    
    Args:

        traj: The trajectory of interest. This could originate from task space
        or grid space, List[tuple(Float, Float), tuple(Float, Float), ...].
                
    Returns:
        
        shortest_segment: The shortest segment encountered in the passed
        trajectory, Float.
    '''
    shortest_segment = None
    for i in range(len(traj) - 1):
        seg_length = math.sqrt((traj[i][0] - traj[i+1][0])**2 + (traj[i][1] -\
            traj[i+1][1])**2)
        if shortest_segment == None:
            shortest_segment = seg_length
        if seg_length < shortest_segment:
            shortest_segment = seg_length
    return shortest_segment


def plot_trajectory(*arg, title="data", **kwargs):
    '''Plots a trajectory. The title is optional. The trajectory could be from
    task space or grid space. This is purely a visualization tool.
    
    Kwargs:
    
        title: Optional plot titel, None, or String.
        
        extents: Specify plot extents. If None, the plot is
        automatically scaled to fit the trajectory. This is given as x-min,
        x-max, y-min, y-max, List[Float, Float, Float, Float]
        
        image_file: Specifies the image file name to overlay trajectories 
        onto. The image file must be located in the working directory, and it
        was tested using .jpg image types. This feature is usefull for
        visualizing trajectory obsticals, such as, roads or walls. However, 
        the image extents must be properly sized and alligned to the task
        space where the trajectory was generaged, String. 
    
    Args:

        Trajectory to plot. Could originate from task space or grid
        space, List[tuple(Float, Float), tuple(Float, Float), ...].
                
    Returns:
        
        N/A
    '''
    
    
    #Allows user to specify coordinate ext
    extents = kwargs.get("extents")
    
    #Allows user to specify a background image
    img_name = kwargs.get("image_file")    
    
    if img_name != None:
        img = mpimg.imread(img_name)   
        
        # Create a figure and axes
        fig, ax = plt.subplots()
        ax.imshow(img, extent=extents)
        
    else:
        ax = plt

    #Add plots of trajectories 
    for data in arg:
        
        for i in range(len(data) -1): 
            x1 = data[i][0]
            y1 = data[i][1]
            x2 = data[i+1][0]
            y2 = data[i+1][1]
            ax.plot([x1, x2], [y1, y2], c='blue')
            ax.scatter(x1, y1, c='blue')
        
    #Set extents of plots
    if extents != None:
        plt.xlim(extents[0], extents[1])
        plt.ylim(extents[2], extents[3])
    
    plt.title(title)
    plt.show()
    plt.close()


if __name__ == "__main__":

    pass