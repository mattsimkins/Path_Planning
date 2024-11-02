from pvf_fun import *
from training_model import BuildGrid
import json


class TrainModel(BuildGrid):
    '''This is a wrapper for the class updateGrid. It provides an interface
    for reading in trajectories, training a model, saving those model, and
    generating pseudo average trajectories based on the models. Documented 
    behavior describes "trajecotry space" (ts) and "grid space" (gs).
    Trajectory space describes the cartesian space in which trajectories
    are defined. Grid space describes the bounded coordinate frame that
    is meaningfull to the grid, i.e. trainin model. Because the coordinate
    frame in grid space has its orgin and lower extents at (0,0), trajectories
    from task space must be shifted to grid space, and vice versa.
    Shifts from one space to the other use the same vector, but with opposite
    sign.  
    '''
    
    
    def save_model(self):
        '''Saves parameters TrainModel object, including the model state. For
        this reason, it is included as a class member function.
        
        Args:
        
            N/A.
            
        Returns:
        
            None: If model could not be found or read in.
        '''
        
        try:
            update_count = self.grid_update_count
        except AttributeError:
            print(("Error: No updates to model.\nNode spacing may not have"
                   "been specified when instantiating object."))
            return None
        
        if self.grid_update_count == 0:
            print("Error: No model to save.")
        else:
            model = {
                #Defines grid space distance between node along x axis
                "node_spacing": self.node_spacing,
                
                #Coordinate frame extents in grid space, orgin is at (0,0)
                "grid_extents": self.grid_extents,
                
                #task space extents
                "trajectory_extents": self.traj_extents,
                
                #used to shift from gs coordinates to ts coordinates
                "shift": self.shift2coord,
                
                #average length from training trajectories
                "average_path_length": self.average_path_length,
                
                #Number of training trajectories used for saved model
                "update_count": update_count,
                
                #max number of segments encountered training set
                "max_trajectory_count": self.max_coord_count, 
                
                 #shortest segement found among trajectories in training set
                "shortest_segment": self.shortest_segment,
                
                #Average trajectory generated from last saved model
                "model_trajectory": self.av_trajectory,
                
                #Start coordinate from last trajectory used to update model
                "last_used_start_point": self.last_start_coord,
                "grid": self.grid.tolist(),
            }
            
            #Save model to working trajectory
            with open(self.path2data+'model.json', 'w+') as f:
                json.dump(model, f)
                print("Model saved")


    def train_on_trajectory(self, **kwargs):
        '''
        Returns a calculated pseudo-average trajectory
        
        Kwargs:
        
            path: Name of relative directory containing .txt files that are
            used to train a model.
            
            start_coord: The requested x and y coordinate of a start location
            in task space for a calculated trajectory, Tuple(Float, Float).
            
            extents: A coordinate frame that spans all training trajectories in
            task space. If None are passed the extents of the coordinate
            frame is automatically generated to fit the extents of trajectory
            data used for the first instance of training. If manually defined,
            extents include x-min, x-max, y-min, and y-max, and are given by
            List[Float, Float, Float, Float].
            
            traj_name: The name of the .txt trajectory file used for training,
            and are given by a String with the .txt extension, String.
        
        Returns:
        
            traj_av: A trajectory given by a series of x and y coordinates
            in task space, List[Tuple(float, float), Tuple(Float, Float), 
            Tuple(Float, Float), ...].
            
            None: Returned for keyword errors, inconsistencies with model, a
            model failure, or invalid start point.
        '''
                
        path = kwargs.get("path")
        if path != None:
            #Handle the case if a slash was not included
            if path[-1] != "/": path = path + "/"
            self.path2data = path
        
        start_coord = kwargs.get("start_point")
        
        extents = kwargs.get("extents")
        
        traj_name = kwargs.get("file_name")
        
        old_model = open_model(self.path2data)
        
        #Check extents
        if extents != None:
            if isinstance(extents, list) == False or len(extents) != 4:
                print(("Extents must be provided in the form List[<x min" 
                       "num>, <x max num>, <y min num>, <y max num>]."))
                extents = None
            if extents != None and old_model != None:
                print('Saved model already exists. Extents will be ignored.')
                extents = None
        
        #Prevent further execution if node spacing doesn't match model
        if old_model != None and old_model["node_spacing"] != self.node_spacing:
            print("Error: A saved model was found having a node spacing of "
                  "{}, but the node spacing specified was {}."
                  .format(old_model["node_spacing"], self.node_spacing))
            old_model = None
            return None        
        
         #Assume user just wants to generate a trajectory by passing nothing  
        if traj_name == None and old_model == None:
            print("No model exists.\nspecify a file name to train on.")
            return None
        
        #Initializes model none exists to read in
        if old_model != None and self.grid_update_count == 0:
            self.node_spacing = old_model["node_spacing"]
            self.grid_extents = old_model["grid_extents"]
            self.traj_extents = old_model["trajectory_extents"]
            self.shift2coord = old_model["shift"]
            self.average_path_length = old_model["average_path_length"]
            self.grid_update_count= old_model["update_count"]
            self.max_coord_count = old_model["max_trajectory_count"]
            self.shortest_segment = old_model["shortest_segment"]
            self.av_trajectory = old_model["model_trajectory"]
            self.last_start_coord = old_model["last_used_start_point"]
            self.grid = old_model["grid"]
            print("Old model was read in.")
        
        #User passes nothing and wants to see last saved average trajectory
        if traj_name == None and old_model != None and\
            start_coord == None:
            return old_model["model_trajectory"]
        
        #Model trained, user wants average trajectory based on start point
        if traj_name == None and old_model\
            != None and start_coord != None:
                
            #Check that start coordinate is within training extents
            if start_coord[0] < self.traj_extents[0]:
                print(("x-coordinate in start point is too low.\nUse a value "
                       "> {}.".format(self.traj_extents[0])))
                return None
            if start_coord[0] > self.traj_extents[1]:
                print("x-coordinate in start point is too high.\nUse a value "
                      "< {}".format(self.traj_extents[1]))
                return None
            if start_coord[1] < self.traj_extents[2]:
                print("y-coordinate in start point is too low.\nUse a value "
                      "> {}.".format(self.traj_extents[2]))
                return None
            if start_coord[1] > self.traj_extents[3]:
                print("y-coordinate in start point is too high.\nUse a value "
                      "< {}.".format(self.traj_extents[3]))
                return None
            
            #Shift start point to grid space coordinate
            grid_start_point = (start_coord[0] -\
                self.shift2coord[0],start_coord[1] -\
                    self.shift2coord[1])
            
            #Calculate a pseudo average trajectory based on start point
            av_traj = self.av_traj(grid_start_point)
            
            #Ensure that model returned a trajectory
            if av_traj == None:
                print(("Bad start point. Consider using a start point closer "
                       "to {}.".format(self.last_start_coord)))
                return None
            else:
                #Convert trajectory from grid space to task space
                av_traj_ts = shift_traj(av_traj, self.shift2coord)
                return av_traj_ts

        #Check for valid model
        try:
            update_count = self.grid_update_count
        except AttributeError:
            print(("Error: No training was found. Model may not save."
                   "Verify node spacing was specified when instantiatig "
                   "object."))
            return None
        
        #Using an existing model, no new one is created
        if old_model != None or (old_model == None and update_count != 0):
            traj_ts = read_traj(self.path2data, traj_name)
            
            #Check for valid structuring of trajectory
            if traj_ts == None:
                print("Most recently provided trajectory is invalid.")
                return None
            
            #convert from task space to grid space
            traj_ts2gs = convert_traj_ts2gs(traj_ts, self.node_spacing,\
                extents, self.shift2coord)
            
            #Check for duplicate coordinates
            traj_gs = check_extents(traj_ts2gs[0], self.grid_extents)
            if traj_gs == None:
                return None
            
            #Update model with trajectory
            self.update_grid(traj_gs)
            
            #Save, or overwrite the last calculated trajectory
            av_traj_gs = self.av_traj((traj_gs[0][0], traj_gs[0][1]))

            #If model could not generate a trajectory provide a warning
            if av_traj_gs == None:
                print(("Warning: Training trajectory {} compromised model.\n"
                      "Consider not saving model and using a node spacing > "
                      "{}.".format(traj_name, self.node_spacing)))
                return None
                
            else: #Convert trajectory from grid space back to trjectory space
                self.av_trajectory = shift_traj(av_traj_gs, self.shift2coord)
                
                #Update to the last start point
                self.last_start_coord = self.av_trajectory[0]

                return self.av_trajectory
                

        #Section creating a new model, first read in trajecory
        traj_ts = read_traj(self.path2data, traj_name)
        
        #After trajectory is read in, convert to grid space
        traj_ts2gs = convert_traj_ts2gs(\
            traj_ts, self.node_spacing, extents, None)
        
        #Check for duplicate coordinates and exceeded coordinate frame bounds
        traj_gs = check_extents(traj_ts2gs[0], traj_ts2gs[1])
        if traj_gs == None: return None
        
        #Trajectory is valid, update class member variables
        extents_gs = traj_ts2gs[1]
        self.shift2coord = traj_ts2gs[2]
        self.traj_extents = [self.shift2coord[0], extents_gs[0] +\
            self.shift2coord[0],\
            self.shift2coord[1], extents_gs[1] + self.shift2coord[1]]
        
        #Check that no unsaved model exists using update count
        if old_model == None and self.grid_update_count == 0:
            
            #Inform user that a new model is being created.
            print(("No existing model found. Creating one with "
                   "specified node spacing.\n"))
            
            #Check if trajectory segments > node spacing 
            shortest_segment = find_shortest_seg(traj_gs)
            if shortest_segment < self.node_spacing:
                print(("Warning: Trajectory averaging may round off bends. "
                       "Consider node spacing <".format(shortest_segment)))
            self.set_coord_frame_extents(extents_gs)
            self.update_grid(traj_gs)

            #calculate a trajectory for user coordinate frame
            grid_av_traj = self.av_traj((traj_gs[0][0], traj_gs[0][1]))
            
            #Training may proceed, but the most recent data broke the model
            if grid_av_traj == None:
                print("Warning: Training trajectory {} "
                      "compromised model.\nConsider using a node spacing "
                      "> {}.".format(traj_name, self.node_spacing))
                return None
            
            #convert grid space trajectory back to tajectory space
            else:
                self.av_trajectory =\
                    shift_traj(grid_av_traj, self.shift2coord)
                self.last_start_coord = self.av_trajectory[0]
                return self.av_trajectory


if __name__ == "__main__":
    pass
