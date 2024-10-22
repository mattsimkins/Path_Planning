# Path_Planning
A novel tool using a “Path Vector Field” to calculates a pseudo-average trajectory from one or more training trajectories. The calculated path is “pseudo” because it skips loops and and avoids collisions from the training set.

# Description
This project was motivated by a “Learning by Demonstration” project for industrial robots. The goal was for a robot to observe multiple attempts by humans to complete a task. Specifically, the robot would track hand trajectories, calculate an average trajectory, and then perform that task with an end effector. Originally, the approach was to calculate an average trajectory from multiple trajectory observations. However, no algorithm exists that calculates a true average, and calculating a true average trajectory in a statistically rigorous way appears unachievable. Moreover, algorithms that do calculate something approximating an average have inherent problems. For example, if two vehicles take alternative paths around an obstacle, one to the left, the other to the right, the average trajectory will take the center path, directly into the obstacle!
This project uses a novel technique called a “Path Vector Field” (PVF) to solve this problem. Rather than apply a statistical treatment, PVF uses a machine learning approach wherein training trajectories are iteratively used to generate, and improve a PVF model. Model history is captured in a grid of “nodes” that span task space. Inspiration for this work derives from brain physiology in the entorhinal cortex that allows animals to navigate mazes, so-called “grid cells”. 
Model training does not require a collection of trajectories that are processed together. Instead, the model is updated and improved iteratively. The model is update-able even as a trajectory is being formed in real time. The qualifier “pseudo” is used because the calculated path is not a true average in a statistical sense. It will avoid obstacles by adhering to an approximate average of a valid path. Moreover, PVF automatically cuts out trajectory loops, or equivalently, cases where the trajectory crosses itself. While this is advantageous for many use-cases, it is not suitable for trajectories where loops are desired. Finally, PVF takes into account changing trajectory velocity. If different trajectories are captured with a constant sampling rate, higher and lower velocities will result in more or less distance respectively between succeeding coordinate points. PVF approximately calculates average speed across portions of the trajectory. In terms of algorithmic complexity, training and trajectory estimation are in linear time with respect to trajectory count and coordinate count. In terms of space complexity, PVF is O(n<sup>2</sup>) with respect to the density of nodes being used to span the trajectory space, and the size of that space.
At the time of this writing, PVF works for 2D cases, but a 3D version is being contemplated. Notwithstanding, PVF efficiently solves certain problems, such as finding the correct path through a maze, or avoiding obstacles. Use cases may include, but are not limited to training by demonstration for plainer moves of a SCARA robot, path planing for AGVs, navigating a maze, or traffic flow modeling.

# Getting Started
## Dependencies
* Linux, developed using Ubuntu 24.04
* Python 3.12.3, Matplotlib 3.9, NumPy 1.26.4, JSON
* Optionally, Turtle
## Installing
* Download and unzip (or clone) pvf.py, pvf_fun.py, training_model.py, and  training_model_fun.py into your working directory
* In the working directory, create a directory for putting your training trajectories. The repository contains a sample of trajectories within a directory called “Dataset”.
## Executing program
### Creating and Training a New Model
* At the top of your main program:
```
import pvf
```
* Create a training object with the desired node spacing. Assuming the desired node spacing is 60:
```
T = pvf.TrainModel(60)
```
* Train on a first trajectory by specifying the name of the directory containing the trajectory. The returned trajectory is the pseudo average trajectory. Assuming the directory is named “Dataset” and the trajectory file is named “data1.txt’:
```
traj = T.train_on_trajectory(path='Dataset', file_name=”data1.txt”)
```
* Alternatively, the coordinate frame extents are automatically assigned based on the trajectory size. If it is known that the trajectory is significantly smaller than future training trajectories the extents can be specified manually by providing a 4-element list comprised of the minimum x, maximum x, minimum y, and maximum y:
```
traj = T.train_on_trajectory(path='Dataset', file_name="data1.txt", extents=[-700, 700, -700, 600])
```
* Train on a second trajectory. This step can be repeated for as many trajectory files as are in the training directory. Assuming the next file name is “data2.txt”:

```
traj = T.train_on_trajectory(file_name=”data2.txt”)
```
* Each returned trajectory captures the model calculated pseudo-average trajectory based on all the data trained on until that point. You can visualize the model trajectory as follows:

```
pvf.plot_trajectory(traj)
```
* The function train_on_trajectory() returns a calculated trajectory based on the starting point from the last trained trajectory. However, it can also return a trajectory from any start point, provided that the start point is close to any portion of any trajectory the model was trained on. In this example, a calculated trajectory is generated from start point x = -76, y = 84:
```
traj = T.train_on_trajectory(path='Dataset', start_point=(-76, 84))
pvf.plot_trajectory(traj, "different start point")
```
* If the model looks acceptable, save it. After execution of the following line, a file called “model.json’ will populate in the dataset directory:
```
T.save_model()
```
### Loading and Training a Saved Model
* This section assumes a model was saved  previously to the dataset directory and additional training is desired. The node spacing initialization must match the spacing used in the model target model. Node spacing is observable by opening the target json model. Create a new object called S:
```
S = pvf.TrainModel(60)
```
* Train on the next data file:
```
traj = S.train_on_trajectory(path='Dataset', file_name="data3.txt")
```
* Re-plot the trajectory to see how it has been adjusted:
```
pvf.plot_trajectory(traj, "re-training")
```
* The Path Vector Field associated with the model is viewable with the following command:
```
S.plot_grid()
```
### Concluding Remarks
Grid spacing selection has a large effect on model performance. If the spacing is too small (dense grid) a larger training set is required. If the grid spacing is too large the calculated trajectory will tend to round off sharper trajectory curves, akin to high frequency filtering.
Because training trajectories may start and end at different locations, it is difficult to specify the calculated trajectory’s end point. This version of PVF estimates the end point based on average estimates of the training data, and at times, a small tail may appear at the end of a calculated trajectory. This is caused by an overestimation of average path length and is normal.
After training on a small number of trajectories, the calculated trajectories are sometimes cut short. This becomes less common with added training on a larger dataset. Training on the first trajectory is critical as the model parameters are difficult to adjust thereafter. That said, model training is rapid so some iteration and tuning is recommended. 
Included in the repository is a file named “trajectory_creator.py”. This tool allows users to generate their own trajectories quickly by hand for trying PVF.  A code snippet that iterates rapidly over a potentially large data set of .txt files follows:
```
      import pvf
      import os
      
      #Create PVF model with node spaing=60
      T = pvf.TrainModel(60)
      
      #Read in all .txt files from dataset directory
      file_list = os.listdir("Dataset")
      for file_name in file_list:
          if file_name[-4:] != '.txt':
              file_list.remove(file_name)
      for file_name in file_list:
          traj = T.train_on_trajectory(path='Dataset', file_name=file_name)
      
      #Calculate trajectory
      pvf.plot_trajectory(traj, "Pseudo-Average Trajectory")
      
      #Save model to directory containing training data
      T.save_model()
      
      #For large grid sizes this plot can be skipped
      T.plot_grid()
```
# Author
Matt Simkins
simkinsmatt@hotmail.com
# Version History
* 0.1, Initial Release
# License
GNU General Public License v3.0
