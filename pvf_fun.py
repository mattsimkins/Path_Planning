import math
import json
import matplotlib.pyplot as plt
import numpy as np

#Constant that scales a grid spacing to the height of a triangle
Y_FACT = math.sqrt(3)/2

def open_model(file_path):
    
    '''Opens a model with the file name "model.json". The model is assumed
    to have been generated by the caller.
        
    Args:
        file_path: Relative path to data directory.
        
    Returns:
        model: Includes a dictionary that should contains parameter
        specified by the caller.
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

#Read in trajectory from .txt file
def read_traj(file_path, file_name):
    '''Reads in trajectories for training. Trajectories are structured as
    two, space separated floating point values for x and y respectivel,
    followed by an end line. This function parces characters and formats
    them for use by PVF.
    
    Args:
    
        file_path: Gives the relative path to the directory containing
        trajectory files for training, String.
        
        file_name: Gives the trajectory name for the .txt file being
        read in as a trajectory
    
    Returns:

        traj: A data type structured for use by PVF, List[tuple(float, float),
        tuple(float, float), tuple(float, float), ...]
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
    
        traj: A task space trajectory, List[Tuple(float, float), Tuple(float,
        float), Tuple(float, float), ...]
        
        node_spacing: Spacing between nodes along edges of an equalateral
        triangle, Float
        
        extents: Manually defined coordinate extents in task space given by 
        min x, max x, min y, max y, List[float, float, float, float]
        
        shift2traj_coord: A shift vector that is ordinarily return by this
        function. If passed, it was found in a pre-existing model.
        
    Returns:
    
        shifted_traj:
        
        grid_cart_extents: 
        
        traj_shift_gs2ts: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

def check(traj, extents):
    
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
        #cover the case for the last coordinate in trajectory
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
                        
    if len(duplicates_index_list) != 0:
        print("Trajectory contains", len(duplicates_index_list),\
            "duplicate coordinates that are being removed\n")
        for index in duplicates_index_list:
            removed_coord = traj.pop(index)
            print("Coordinate", removed_coord, 'was removed.')
    return traj

def shift_traj(traj, shift):
    traj_shifted = []
    for c in traj:
        coord = (c[0] + shift[0], c[1] + shift[1])
        traj_shifted.append(coord)
    return traj_shifted
    
def find_shortest_seg(traj):
    '''!!!!!!!!!!!!!!!!!!
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

def plot_trajectory(data, title="data"):
    '''plots array of trajectory coordinates
    '''
    if data == None:
        pass
    else:
        for i in range(len(data) -1): 
            x1 = data[i][0]
            y1 = data[i][1]
            x2 = data[i+1][0]
            y2 = data[i+1][1]
            plt.plot([x1, x2], [y1, y2], c='blue')
            plt.scatter(x1, y1, c='blue')
        plt.title(title)
        plt.show()
        plt.close()


#need to work on this when done!!!!!!!!!
def standard_error_message():
    print("Usage:\nstart_point=tupel(<coordinate x>, <coordinate y>\nextents=list[]")

if __name__ == "__main__":

    pass