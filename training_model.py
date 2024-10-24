from training_model_fun import *
import math
import numpy as np
import matplotlib.pyplot as plt


class BuildGrid:
    
    '''Trains a model (grid) by iteratively reading in 2D trajectories. After
    training the model the class calculates a quai-average trajectory given
    an arbitrary start location. The calculated trajectory works by navigating
    the trained grid. After training, a start location can be passed. If the
    start location is located on, or near a path of one or more of
    the training trajectories a psuedo-average trajectory is returned. The
    returned path (av_traj) roughly describes an average trajectory in that
    it captures an amalgam of training trajectories. However, it is also not
    a true 'average' because it adheres to a valid path. For example, if one
    training trajectory steers left around an obstacle while the other steers
    right, the av_traj will either go left or right, but it will not steer 
    through the obstacle as a true average would. Additionally, if training
    trajectories contain loops, av_traj will bypass the loop. For this reason
    this algorithm is unsuitable for trajectories that should include loops, 
    or more specifically, trajectores that should cross its own path at
    points.
    Grid spacing is a critical parameter. Small grid spacings with large
    training trajectories result in a larger grid. While the model learns
    and calculates av_traj in linear time, the algorithmic memory complexity
    grows roughly by order n^2 where n relates to the size of the trajectory,
    or inversely to the node spacing size.
    '''
    
    def __init__(self, node_spacing = None):
        
        '''Initialization of variables. Initialization of node spacing depends
        on the trajectory characteristics, and the node spacing value is
        critical to performance. Core to this algorithm is the grid of nodes
        being established. The geometry of the grid is given below with "*"
        representing the nodes, and connecting lines representing the grid
        spacing. The grid resembles a ternary plot, however, index assigment
        was selected so that it more easily integrates with cartesian space.
        The lower left node is at index [0, 0], and has a cartesian coordinate
        of (0, 0). the node to its right its at [1, 0], the node to its upper
        rigth is at [0, 1], and the node directly above it is at [2, 0]. 
        Node spacing provides the context between grid indices and cartesian
        grid space. For example, the node to the imediate right of the lower
        left node is at the location (node_spacing, 0)
        
        --- * ---- * ---- * --
          /   \  /   \  /   \
        * ---- * ---- * ---- *
          \   /  \   /  \   /
        --- * ---- * ---- * --
          /   \  /   \  /   \
        * ---- * ---- * ---- *
        
        Args:
        
            node_spacing: Key word node_spacing = A numerical value specifying
            grid density.
        
        Returns:
        
            traj_av: List[Tuple(float, float), Tuple(float, float), ...].
            
            None: If node_spacing is not provided.
        '''
        
        if node_spacing == None:
            print("Error: Must specify key word node_spacing=<number>")
            return None

        if node_spacing != None:
            self.node_spacing = node_spacing
            #Average length of training paths
            self.average_path_length = 0
            #Number of trajectories used for training
            self.grid_update_count = 0
            #Max quantity of cooordinates among trajectory training set
            self.max_coord_count = 0
            #Shortest distance between coordinates among training set
            self.shortest_segment = self.node_spacing
        

    def set_coord_frame_extents(self, upper_corner_loc):
        
        '''Initialization of coordinate frame. The coordinate frame is
        defined as having an x and y extent, with the origin location
        at (0,0). Inputted trajectories must not exceed the limits of
        this coordinate frame. Moreover, negative x and y coordinates
        are not permitted. If the caller requires training on trajectories
        that include negative coordinates, the caller is responsible for
        defining suitably sized coordinate extents, as well as a vector 
        that shifts trajectories back to their original coordinates.
        
        Args:
            upper_corner_loc: A 2-element list describing the x and y
            location of the upper-right coordinate extent.
        
        Returns:
            N/A
        '''
        
        x = math.ceil(upper_corner_loc[0])
        y = math.ceil(upper_corner_loc[1])
        self.grid_extents = [x, y]
        
        #Calculte number of rows of nodes along x
        nom_node_count_x = self.grid_extents[0]/self.node_spacing
        node_count_x = math.ceil(nom_node_count_x)

        #Calculte number of rows of nodes along y
        node_spacing_in_y = self.node_spacing*Y_FACT
        nom_node_count_y = self.grid_extents[1]/node_spacing_in_y
        node_count_y = math.ceil(nom_node_count_y)
        
        #Parameter list includes vector x and y components
        param_list = 2
        self.grid = np.zeros((node_count_x, node_count_y, param_list))
        
        
    def check_extents(self, loc, check_type):
        
        '''As nodes are used for trajectory training, or for calculating
        a pseudo-average trajectory, it is necessary that indexing stays
        within the extents of the grid.
        
        Args:
        
            loc: A catesian coordinate in grid space, Tupel(Float, Float)
            
            check_type: Accepts "point" or "triangle". If triangle is
            specified, the location of three neighboring nodes is
            evaluated, String
            
        Returns:
        
            exceeded: A True value indicates that the location is invalid,
            True or False
        '''
        
        #Check is a specific location exceeds limits
        exceeded = False
        if check_type == "point":
            if loc[0] <= 0 or loc[0] >= self.grid_extents[0] or\
                      loc[1] <= 0 or loc[1] >= self.grid_extents[1]: 
                          return True
        
        #Checks if a "trident" (three neighboring nodes) exceeds a limit
        if check_type == "triangle":
            
            #Coordinates on the coordinate frame axese are prohibited
            if loc[0] == 0 or loc[1] == 0: return True
            
            #The triangle of 3 nearest nodes is defined here as a "trident"
            [left_ind, right_ind, center_ind] = \
                find_trident(loc, self.node_spacing)
                
            if left_ind[0] < 0 or\
                left_ind[1] + 1 > self.grid.shape[1] or\
                left_ind[1] < 0 or\
                right_ind[0] + 1 > self.grid.shape[0] or\
                center_ind[1] < 0 or\
                center_ind[1] + 1 > self.grid.shape[1]: exceeded = True
        return exceeded
    
    
    def update_node(self, vec, node_indices):
        
        '''Nodes are updated with a vector that goes from one coordinate in a
        trajectory to the next. The node is "visited" when a trajectroy
        coordinate passes within its "trident". If the node was never visited
        then its parameters are populated with the vector components. However,
        if the node was visited previously the update must consider the vector
        components that were already present for that node. In that case, the
        update includes a series of vector operations.
        
        Args:
        
            vec: A vector being used for the update,
            numpy.array([Float, Float])
            
            node_indices: The indices of the node being updated, [int, int].
        
        Returns:
        
            N/A
        '''
        
        ind_i = node_indices[0]
        ind_j = node_indices[1]
        
        #A node's fist visit, populate nodes if empty
        if self.grid[ind_i][ind_j][0] == 0 and \
            self.grid[ind_i][ind_j][1] == 0:
            self.grid[ind_i][ind_j][0] = vec[0]
            self.grid[ind_i][ind_j][1] = vec[1]
        
        #Nodes not empty, update must consider previous value
        else:
            hist_vec = np.array([self.grid[ind_i][ind_j][0],\
                self.grid[ind_i][ind_j][1]])
            len_hist_vec = np.linalg.norm(hist_vec)
            len_vec = np.linalg.norm(vec)
            scale_fact = len_vec/(len_vec + len_hist_vec)
            diff_vec = np.subtract(vec, hist_vec)
            scaled_diff_vec = diff_vec*scale_fact
            updated_vec = np.add(hist_vec, scaled_diff_vec)
            self.grid[ind_i][ind_j][0] = updated_vec[0]
            self.grid[ind_i][ind_j][1] = updated_vec[1]
        
        
    def av_traj(self, loc_start):
        
        '''Calculates a pseudo avrage trajectory using a trained grid. It is
        also called when a grid update occurs. That is because the process of
        calculating a trajectroy may also include additional updates to the
        grid.
        
        Args:
        
            loc_start: The start location to begin calculating the trajectory,
            Tuple(Float, Float)
            
        Returns:

            av_traj: This calculated pseudo-average trajectory is the primary
            output of training, List[Tuple(Float, Float), Tuple(Float, Float),
            ...]
        
            None: If trajectory calculation fails. This implies something is
            wrong with the grid model, or there was a start position located
            on a portion of the grid that had never beeing visited during
            trianing.
        '''
        
        if self.node_spacing == None:
            print(("No existing model found.\n Consider using key word "
                   "node_spacing=<spacing> for instantiation"))
            return None
        
        #Psuedo-average trajectory calculation preparation
        av_traj = [loc_start]    
        loc = loc_start
        running_path_length = 0
        stop_calc = False

        #Continues growing pseudo-average trajectory until a stop is generated
        while stop_calc == False:
            stop_calc = self.check_extents(loc, "triangle")
                
            #Gather indices neighboring nodes
            [left_ind, right_ind, center_ind] =\
                find_trident(av_traj[-1], self.node_spacing)
                
            #Gather vectors recorded in triad of neighboring nodes
            vec_left = np.array([self.grid[left_ind[0]][left_ind[1]][0],\
                self.grid[left_ind[0]][left_ind[1]][1]])
            vec_right = np.array([self.grid[right_ind[0]][right_ind[1]][0],\
                self.grid[right_ind[0]][right_ind[1]][1]])
            vec_center = np.array([self.grid[center_ind[0]]\
                [center_ind[1]][0],\
                    self.grid[center_ind[0]][center_ind[1]][1]])
            left_visited = 0
            right_visited = 0
            center_visited = 0
            
            #Vector length of 0 indicates the node was unvisited
            if np.linalg.norm(vec_left) != 0: left_visited = 1
            if np.linalg.norm(vec_right) != 0: right_visited = 1
            if np.linalg.norm(vec_center) != 0: center_visited = 1
            
            #Calculated pseudo average proceeds based on 3 cases
            num_nodes_visited = left_visited + right_visited + center_visited
            
            #Case 1 - All nodes are empty so abort, model fails
            if num_nodes_visited == 0:
                return None
            
            #Case 2 - Two of three nodes are empty, but recoverable
            if num_nodes_visited == 1:
                
                #Calculate location of nodes
                loc_left = coord_from_ind(left_ind, self.node_spacing)
                loc_right = coord_from_ind(right_ind, self.node_spacing)
                loc_center = coord_from_ind(center_ind, self.node_spacing)
                
                #Right and center nodes were empty, use left
                if left_visited == 1 and right_visited == 0 and\
                    center_visited == 0:
                    loc_left_points_to = np.add(loc_left, vec_left)
                    vec_right = np.subtract(loc_left_points_to, loc_right)
                    vec_center = np.subtract(loc_left_points_to, loc_center)
                
                #Left and center nodes were empty, use right
                if left_visited == 0 and right_visited == 1 and\
                    center_visited == 0:
                    loc_right_points_to = np.add(loc_right, vec_right)
                    vec_left = np.subtract(loc_right_points_to, loc_left)
                    vec_center = np.subtract(loc_right_points_to, loc_center)
                
                #Left and right nodes were empty, use center
                if left_visited == 0 and right_visited == 0 and\
                    center_visited == 1:
                    loc_center_points_to = np.add(loc_center, vec_center)
                    vec_right = np.subtract(loc_center_points_to, loc_right)
                    vec_left = np.subtract(loc_center_points_to, loc_left)
                
                #Populate any empty nodes
                self.update_node(vec_left, left_ind)
                self.update_node(vec_right, right_ind)
                self.update_node(vec_center, center_ind)
                
                #All vectors point to same place, use of left is arbitrary
                loc_array = loc + vec_left
                    
            #Case 3 - One of three nodes were zero
            if num_nodes_visited == 2:
                
                #Find distances to neighboring grid nodes
                dist2right = dist2node(av_traj[-1],\
                    right_ind, self.node_spacing)
                dist2left = dist2node(av_traj[-1],\
                    left_ind, self.node_spacing)
                dist2center = dist2node(av_traj[-1],\
                    center_ind, self.node_spacing)
                
                #Left node empty, use right and center
                if left_visited == 0 and right_visited == 1 and\
                    center_visited == 1:
                                        
                    #Determine weights based on distances to nodes
                    den = dist2right + dist2center
                    weight_right = dist2right/den
                    weight_center = dist2center/den
                    
                    #Weight the vectors
                    vec_right_weighted = weight_right*vec_right
                    vec_center_weighted = weight_center*vec_center
                    
                    #Calculate sum of weighted vectors
                    vec_r_plus_c = np.add(vec_right_weighted,\
                        vec_center_weighted)
                    loc_array = loc + vec_r_plus_c
                    
                    #Populate empty node
                    loc_left = coord_from_ind(left_ind, self.node_spacing)
                    vec_left = np.subtract(loc_array, loc_left)
                    self.update_node(vec_left, left_ind)
                    
                #Right node empty, use center and left
                if left_visited == 1 and right_visited == 0 and \
                    center_visited == 1:
                    
                    #Determine weights based on distances to nodes
                    den = dist2left + dist2center
                    weight_left = dist2left/den
                    weight_center = dist2center/den
                    
                    #Weight the vectors
                    vec_left_weighted = weight_left*vec_left
                    vec_center_weighted = weight_center*vec_center
                    
                    #Calculate sum of weighted vectors
                    vec_l_plus_c = np.add(vec_left_weighted,\
                        vec_center_weighted)
                    loc_array = loc + vec_l_plus_c
                    
                    #Populate empty node
                    loc_right = coord_from_ind(right_ind, self.node_spacing)
                    vec_right = np.subtract(loc_array, loc_right)
                    self.update_node(vec_right, right_ind)
                    
                #Center node empty, use left and right
                if left_visited == 1 and right_visited == 1 and\
                    center_visited == 0:
                    
                    #Determine weights based on distances to nodes
                    den = dist2left + dist2right
                    weight_left = dist2left/den
                    weight_right = dist2right/den
                    
                    #Weight the vectors
                    vec_left_weighted = weight_left*vec_left
                    vec_right_weighted = weight_right*vec_right
                    
                    #Calculate sum of weighted vectors
                    vec_l_plus_r = np.add(vec_left_weighted,\
                        vec_right_weighted)
                    loc_array = loc + vec_l_plus_r
                    
                    #Populate empty node
                    loc_center = coord_from_ind(center_ind, self.node_spacing)
                    vec_center = np.subtract(loc_array, loc_center)
                    self.update_node(vec_center, center_ind)
                
            #case 4 - All 3 nodes are non-zero (best case)
            else:
                #determine distances to neighboring grid nodes
                dist2left = dist2node(av_traj[-1], left_ind,\
                    self.node_spacing)
                dist2right = dist2node(av_traj[-1], right_ind,\
                    self.node_spacing)
                dist2center = dist2node(av_traj[-1], center_ind,\
                    self.node_spacing)
                
                #determine weights based on distances to nodes
                den = dist2left + dist2center + dist2right
                weight_left = (dist2center + dist2right - dist2left)/den
                weight_right = (dist2center + dist2left - dist2right)/den
                weight_center = (dist2left + dist2right - dist2center)/den
                
                #Weight the vectors
                vec_left_weighted = weight_left*vec_left
                vec_right_weighted = weight_right*vec_right
                vec_center_weighted = weight_center*vec_center
                
                #Sum weighted vectors
                vec_l_plus_r = np.add(vec_left_weighted, vec_right_weighted)
                vec_sum = np.add(vec_l_plus_r, vec_center_weighted)
                loc_array = loc + vec_sum
            
            loc = (loc_array[0].tolist(), loc_array[1].tolist())
            
            #Update running path length to ensure path does not run on forever
            new_length = math.sqrt((loc[0]-av_traj[-1][0])**2 + \
                (loc[1]-av_traj[-1][1])**2)
            running_path_length = running_path_length + new_length
            
            #Check if done
            if loc == av_traj[-1]:
                break
            
            #Margin% applied to comparisons against training history
            margin = 50
            
            #Check for excessive coordinate count
            if len(av_traj) > self.max_coord_count +\
                round(0.01*margin*self.max_coord_count):
                stop_calc = True
            
            #Check for excessively short segements
            if self.shortest_segment > new_length +\
                round(0.01*margin*new_length): stop_calc = True
            
            #check that trajectory path is not much longer than average            
            if self.average_path_length < running_path_length:
                stop_calc = True

            #Grow trajectory by one coordinate
            av_traj.append(loc)
            
        return av_traj

    
    def update_grid(self, traj):
        
        '''Updates all nodes of the grid given a training trajectory.
        
        Args:
        
            traj: A training trajectory, List[Tuple(Float, Float), 
            Tuple(Float, Float), ...]
            
        Returns:
        
            N/A
        '''
        
        try:
            if traj == None:
                raise ValueError
        except ValueError:            
            print('Error: Trajectory is type None')
            return None
        
        [shortest_segment, coord_count, path_length] = traj_metrics(traj)
        
        #update class member with funning path length average
        self.grid_update_count += 1
        self.average_path_length = self.average_path_length*\
            (self.grid_update_count-1)/self.grid_update_count +\
                path_length/self.grid_update_count
        
        #track shortest length trajectory segement encountered during training
        if self.shortest_segment == None: #bootstrap
            self.shortest_segment = shortest_segment

        if self.shortest_segment > shortest_segment:
            self.shortest_segment = shortest_segment
        
        #track coordinate counts
        if coord_count > self.max_coord_count:
            self.max_coord_count = coord_count
        
        #check later to see if there are any zero length segments
        zero_length_seg_present = False
        next_ind = 0
        for loc in traj:
            
            #Occurrs for consecutive repeated coordinates
            if shortest_segment == 0:
                print("Invalid trajectory with zero length segment")
                zero_length_seg_present = True
            
            #check that location in trajectory is valid
            exceeded = self.check_extents(loc, "triangle")
            if exceeded == True or zero_length_seg_present == True: break
            else:
                #Reached end of traj list
                if len(traj) == next_ind + 1: break
                
                #Vector to next location
                loc_current = np.array([loc[0], loc[1]])
                next_ind += 1
                loc_next = np.array([traj[next_ind][0], traj[next_ind][1]])
                
                #vector from current location to next along trajectory
                vec2next_traj_pt = np.subtract(loc_next, loc_current)
                
                #length of current vector
                vec_len_vec2next_traj_pt = np.linalg.norm(vec2next_traj_pt)
                
                #Next location is close, no nodes between to update
                if vec_len_vec2next_traj_pt < self.node_spacing:
                    [left_ind, right_ind, center_ind] =\
                        find_trident(loc_current, self.node_spacing)
                    
                    #Find location of nodes
                    loc_left = coord_from_ind(left_ind, self.node_spacing)
                    loc_right = coord_from_ind(right_ind, self.node_spacing)
                    loc_center = coord_from_ind(center_ind, self.node_spacing)
                    
                    #Calculate vector from nodes to trajectory
                    vec_left2traj_next = np.subtract(loc_next, loc_left)
                    vec_right2traj_next = np.subtract(loc_next, loc_right)
                    vec_center2traj_next = np.subtract(loc_next, loc_center)
                    
                    #Upate nodes with trajectory
                    self.update_node(vec_left2traj_next, left_ind)
                    self.update_node(vec_right2traj_next, right_ind)
                    self.update_node(vec_center2traj_next, center_ind)
                    
                
                #Next location is far, must update nodes in between
                else:
                    #find direction of increment
                    vec_hat_vec2next_traj_pt =\
                        vec2next_traj_pt/vec_len_vec2next_traj_pt
                    
                    #Divide current vector into peices to increment
                    n_inc = math.floor(\
                        vec_len_vec2next_traj_pt/self.node_spacing)                    
                    
                    #Creep along long distance, populate nodes along the way   
                    for j in range(n_inc):
                        len_of_vec_increment = (j)*self.node_spacing
                        loc_of_increment = np.add(loc_current,\
                            vec_hat_vec2next_traj_pt*len_of_vec_increment)
                        [left_ind, right_ind, center_ind] = \
                            find_trident(loc_of_increment, self.node_spacing)
                        
                        loc_left = coord_from_ind(left_ind, self.node_spacing)
                        loc_right = coord_from_ind(right_ind,\
                            self.node_spacing)
                        loc_center = coord_from_ind(center_ind,\
                            self.node_spacing)
                        
                        vec_left2traj_next = np.subtract(loc_next, loc_left)
                        vec_right2traj_next = np.subtract(loc_next, loc_right)
                        vec_center2traj_next =\
                            np.subtract(loc_next, loc_center)
                        
                        self.update_node(vec_left2traj_next, left_ind)
                        self.update_node(vec_right2traj_next, right_ind)
                        self.update_node(vec_center2traj_next, center_ind)
                
                #Run calculated average to boaden grid along trajectory
                traj_test = self.av_traj(traj[0])
                
                #Encountered a trident of unvisited nodes
                if traj_test == None:
                    print("Warning: Model failed to calculate trajecoty.\n")
                    pass
               
                
    def plot_grid(self):
        
        '''A visualization tool for evaluating a trained grid. This is also
        useful for providing intuition about how the model works.
        
        Args:
        
            N/A
        
        Returns:
            N/A
        '''
        
        #For large grids the program may appear to hang
        print("Warning, for large grids this can take a long time to plot.\n")
        
        vec_list = []
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                V = np.array([self.grid[i][j][0], self.grid[i][j][1]])
                vec_list.append(V)
        fig, ax = plt.subplots()

        # Add vectors V and W to the plot
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                node_loc = coord_from_ind((i, j), self.node_spacing)
                list_index = j+i*self.grid.shape[1]
                if vec_list[list_index][0] == 0 and\
                    vec_list[list_index][1] == 0:
                    color = 'b'
                else:
                    color = 'r'
                ax.quiver(node_loc[0], node_loc[1], vec_list[list_index][0],\
                    vec_list[list_index][1], angles='xy', scale_units='xy',\
                        scale=1, color=color)
        node_loc = coord_from_ind((self.grid.shape[0], self.grid.shape[1]),\
            self.node_spacing)
        plt.title("Trained Path Vector Field")
        ax.set_xlim([0, node_loc[0]])
        ax.set_ylim([0, node_loc[1]])
        plt.grid()
        plt.show()

if __name__ == "__main__":
    pass