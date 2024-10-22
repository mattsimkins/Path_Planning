import turtle
from pathlib import Path


def save_trajectory(traj):
    
    '''A handy program for generating trajectories manually with a mouse
    or touchpad. This program opens a window. Move the cursor to the next
    intended point along a trajectory and left click. When the window is
    closed the user is prompted to select a file name. It is recommended
    that the window be made full screen in order to obtain greater trajectory
    resolution.
    '''
    
    default_file_name = "data.txt"
    print("Default file name is :", str(default_file_name))
    input_name = input("Enter an alternate file name or type enter: \n")
    if input_name != "": file_name = input_name + ".txt"
    else: file_name = default_file_name
    print("Writing " + file_name + " to working directory.\n")

    # Using Path from pathlib
    file_path = Path(file_name)

    # Open the file in write mode using with statement
    with file_path.open('w') as file:
        # Write content to the file
        for coord in traj:
            x = coord[0]
            y = coord[1]
            file.write(f"{x} {y}")
            file.write("\n")
        
    print("File", file_path, "created and saved successfully")


def generate_trajectory():
    # defining instance of turtle
    pen=turtle.Turtle()
    head=turtle.Turtle()
    head.penup()
    head.hideturtle()
    head.goto(0, 260)
    head.write("Coordinate position",
            align = "center", 
            font = ("courier", 10, "normal"))

    trajectory = []

    # giving circle shape to pen i.e. instance of turtle
    pen.shape("circle")

    # giving colour to turtle
    pen.color("red")

    # define function to make the turtle move 
    # to the position which is clicked by the mouse
    def btnclick(x, y):
        pen.goto(x, y)
        head.clear()    
        head.write(f"x coordinate = {x}, y coordinate = {y}",
                align = "center", font = ("courier", 10, "normal"))
        coordinate = (x, y)
        trajectory.append(coordinate)
        
    # this function will call btnclick whenever mouse is clicked    
    turtle.onscreenclick(btnclick, 1)
    turtle.listen()
    turtle.mainloop()
    save_trajectory(trajectory)

if __name__ == "__main__":
    
    generate_trajectory()
