import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import env
from param import Parameter as p
import csv
import numpy as np
import util
from scipy.interpolate import Akima1DInterpolator

def path_interpolation(xs, ys, point=100):
    t = np.linspace(0, 1, len(xs))
            
    fx = Akima1DInterpolator(t, xs)
    fy = Akima1DInterpolator(t,ys)
        
    dfx_dt = fx.derivative()
    dfy_dt = fy.derivative()
    
    t = np.linspace(0, 1, point)

    spline_xs, spline_ys = fx(t), fy(t)
    thetas = np.arctan(dfy_dt(t)/dfx_dt(t))
    
    return spline_xs, spline_ys, thetas

def plot_path(ax, x, y, theta, start, goal):
    ax.scatter(x, y, marker='x', color='blue', s=5)
    
    env_data = env.Env()
    wall_list = env_data.obs_boundary
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    #wallを配置
    for k in range(len(wall_list)):
        wall = patches.Rectangle((wall_list[k][0], wall_list[k][1]), wall_list[k][2], wall_list[k][3], linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(wall)
    
    #障害物を配置
    for k in range(len(obs_rectangle)):
        x0, y0, w, h = obs_rectangle[k][0], obs_rectangle[k][1], obs_rectangle[k][2], obs_rectangle[k][3]
        rectangle_obstacle = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='black', facecolor='gray')
        ax.add_patch(rectangle_obstacle)
        
    for k in range(len(obs_circle)):
        x_o, y_o, r_o = obs_circle[k][0], obs_circle[k][1], obs_circle[k][2],
        circle_obstacle = patches.Circle((x_o, y_o), radius=r_o, edgecolor='black', facecolor='gray')
        ax.add_patch(circle_obstacle)
    
    ax.set_xlabel(r'$x$[m]')
    ax.set_ylabel(r'$y$[m]')
    ax.set_xlim([p.x_min - p.margin, p.x_max + p.margin])
    ax.set_ylim([p.y_min - p.margin, p.y_max + p.margin])
    
    #startとgoalを配置
    ax.scatter([start[0]], [start[1]], marker='v', color='green', label='start')
    ax.scatter([goal[0]], [goal[1]], marker='^', color='green', label='goal')
    
    ax.quiver(x[-1], y[-1], np.cos(theta[-1]), np.sin(theta[-1]))
    
    ax.set_aspect('equal')
    


def gen_movie(xs, ys):
    fig = plt.figure()
    frames = []
    start = [xs[0], ys[0]]
    goal = [xs[-1], ys[-1]]
    xs, ys, thetas = path_interpolation(xs, ys, 50)
    for i in range(1, len(xs) + 1):
        ax = plt.axes()
        x, y, theta = xs[:i], ys[:i], thetas[:i]
        plot_path(ax, x, y, theta, start, goal)
        frames.append([ax])
    ani = animation.ArtistAnimation(fig=fig, artists=frames, interval=100)
    ani.save("plot_path.mp4")
        
if __name__ == '__main__':
    #csvからnetwork情報を取得
    with open("network_circle.csv") as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        trajectory_vectors = [row for row in reader]
        
    path_index = 5
    trajectory_vector = trajectory_vectors[path_index]
    trajectory_vector = np.array(trajectory_vector)
    trajectory_matrix = util.vector_to_matrix(trajectory_vector)
    xs, ys = trajectory_matrix[0], trajectory_matrix[1]
    gen_movie(xs, ys)