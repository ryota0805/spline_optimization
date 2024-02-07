import numpy as np
from scipy.interpolate import Akima1DInterpolator
import scipy.optimize as optimize
import csv
import util
import plot
import matplotlib.pyplot as plt

start = [0, 0]
theta0 = np.pi/4
x0, y0 = start[0], start[1]
#calc y1
def calc_y1(x, theta, x0, y0):
    y1 = np.tan(theta) * (x - x0) + y0
    return y1

#calc y2
def calc_y2(x, theta, x3, y3):
    y2 = np.tan(theta) * (x - x3) + y3
    return y2

def generate_path_trajectory(middle_path):
    x_path = []
    y_path = []
    
    for i in range(len(middle_path)):
        x_path.append(middle_path[i][0])
        y_path.append(middle_path[i][1])
        
    t = np.linspace(0, 1, len(middle_path))
        
    fx = Akima1DInterpolator(t, x_path)
    fy = Akima1DInterpolator(t, y_path)
        
    dfx_dt = fx.derivative()
    dfy_dt = fy.derivative()
    
    return fx, fy, dfx_dt, dfy_dt

#objective
def objective(x, *args):
    fx, fy, dfx_dt, dfy_dt = args[0], args[1], args[2], args[3]
    
    x1, x2, t = x[0], x[1], x[2]
    
    y1 = calc_y1(x1, theta0, x0, y0)
    
    x3, y3 = fx(t), fy(t)

    theta3 = np.arctan(dfy_dt(t)/dfx_dt(t))
    
    y2 = calc_y2(x2, theta3, x3, y3)
    
    t = np.linspace(0, 1, 100)
    
    dx_dt = (-3*x0 + 9*x1 - 9*x2 + 3*x3)*t**2 + (6*x0 - 12*x1 + 6*x2)*t + (-3*x0 + 3*x1)
    dy_dt = (-3*y0 + 9*y1 - 9*y2 + 3*y3)*t**2 + (6*y0 - 12*y1 + 6*y2)*t + (-3*y0 + 3*y1)
    
    d2x_dt2 = (-6*x0 + 18*x1 - 18*x2 + 6*x3)*t + (6*x0 - 12*x1 + 6*x2)
    d2y_dt2 = (-6*y0 + 18*y1 - 18*y2 + 6*y3)*t + (6*y0 - 12*y1 + 6*y2)
    
    array = dx_dt**2 + dy_dt**2
    
    zero_indices = array == 0
    array[zero_indices] = 0.0001
    
    curvature_power = (dx_dt*d2y_dt2 - dy_dt*d2x_dt2)**2/(array**3)
    
    cubicX = (1-t)**3*x0 + 3*(1-t)**2*t*x1 + 3*(1-t)*t**2*x2 + t**3*x3
    cubicY = (1-t)**3*y0 + 3*(1-t)**2*t*y1 + 3*(1-t)*t**2*y2 + t**3*y3
    
    length = 0
    for i in range(len(cubicX)-1):
        length += ((cubicX[i+1] - cubicX[i])**2 + (cubicY[i+1] - cubicY[i])**2)**(0.5)
        
    w1, w2 = 1, 1
    return w1 * sum(curvature_power) + w2 * length

def optimization(middle_path):
    fx, fy, dfx_dt, dfy_dt = generate_path_trajectory(middle_path)
    args = (fx, fy, dfx_dt, dfy_dt)
    initial_x = [1, 1, 0.5] #x1. x2, t
    fun = objective
    options = {'maxiter':100000}
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0, 1)]
    result = optimize.minimize(fun, initial_x, method='SLSQP', args = args, bounds = bounds, options=options)
    
    return result


def generate_bezier(x, *args):
    fx, fy, dfx_dt, dfy_dt = args[0], args[1], args[2], args[3]
    
    x1, x2, t = x[0], x[1], x[2]
    
    y1 = calc_y1(x1, theta0, x0, y0)
    
    x3, y3 = fx(t), fy(t)

    theta3 = np.arctan(dfy_dt(t)/dfx_dt(t))
    
    y2 = calc_y2(x2, theta3, x3, y3)
    
    t = np.linspace(0, 1, 100)
    
    bezier_x = (1-t)**3*x0 + 3*(1-t)**2*t*x1 + 3*(1-t)*t**2*x2 + t**3*x3
    bezier_y = (1-t)**3*y0 + 3*(1-t)**2*t*y1 + 3*(1-t)*t**2*y2 + t**3*y3
    
    return bezier_x, bezier_y

def main():
    #csvからnetwork情報を取得
    with open("network_circle.csv") as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        trajectory_vectors = [row for row in reader]
        
    #csvから得られた情報からnetworkを構成
    network = []

    for trajectory_vector in trajectory_vectors:
        trajectory_vector = np.array(trajectory_vector)
        trajectory_matrix = util.vector_to_matrix(trajectory_vector)
        x, y = trajectory_matrix[0], trajectory_matrix[1]
        path = []
        for i in range(len(x)):
            path.append([x[i], y[i]])
        network.append(path)
    
    result_list = []
    value_list = []
    for middle_path in network:
        result = optimization(middle_path)
        result_list.append(result.x)
        value_list.append(result.fun)
        
    label = [i for i in range(len(value_list))]
    plt.bar(label, value_list, tick_label=label, align="center")
    plt.show()
    
    for index in range(len(network)):
        x = result_list[index]
        fx, fy, dfx_dt, dfy_dt = generate_path_trajectory(network[index])
        args = (fx, fy, dfx_dt, dfy_dt)
        t = np.linspace(0, 1, 100)
        middle_x = fx(t)
        middle_y = fy(t)
        
        bezier_x, bezier_y = generate_bezier(x, *args)

        plot.test_path(middle_x, middle_y, bezier_x, bezier_y)

main()