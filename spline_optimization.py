import numpy as np
from scipy.interpolate import BSpline, interp1d, Akima1DInterpolator
import scipy.optimize as optimize
import csv
import util
import plot
import matplotlib.pyplot as plt

class SplineBasePathPlanning:
    def __init__(self, start, start_theta, network):
        self.start = start
        self.start_theta = start_theta
        self.network = network
        
    def calc_y1(self, x):
        x0, y0 = self.start[0], self.start[1]
        theta = self.start_theta
        
        y1 = np.tan(theta) * (x - x0) + y0
        return y1
    
    def calc_y2(self, x, theta, x3, y3):
        y2 = np.tan(theta) * (x - x3) + y3
        return y2
    
    def generate_path_trajectory(self, middle_path):
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
    
    def generate_bezier(self, x, *args):
        x0, y0 = self.start[0], self.start[1]
        fx, fy, dfx_dt, dfy_dt = args[0], args[1], args[2], args[3]

        x1, x2, t = x[0], x[1], x[2]
        y1 = self.calc_y1(x1)
        
        x3, y3 = fx(t), fy(t)
        
        if dfx_dt(t) == 0:
            theta3 = np.pi/2.001
        else:
            theta3 = np.arctan(dfy_dt(t)/dfx_dt(t))
            
        y2 = self.calc_y2(x2, theta3, x3, y3)
        
        t = np.linspace(0, 1, 100)
        bezier_x = (1-t)**3*x0 + 3*(1-t)**2*t*x1 + 3*(1-t)*t**2*x2 + t**3*x3
        bezier_y = (1-t)**3*y0 + 3*(1-t)**2*t*y1 + 3*(1-t)*t**2*y2 + t**3*y3
        
        return bezier_x, bezier_y
    
    def objective_function(self, x, *args):
        x0, y0 = self.start[0], self.start[1]
        fx, fy, dfx_dt, dfy_dt = args[0], args[1], args[2], args[3]
        x1, x2, t = x[0], x[1], x[2]
        y1 = self.calc_y1(x1)
        
        x3, y3 = fx(t), fy(t)
        if dfx_dt(t) == 0:
            theta3 = np.pi/2.001
        else:
            theta3 = np.arctan(dfy_dt(t)/dfx_dt(t))
        y2 = self.calc_y2(x2, theta3, x3, y3)
        
        t = np.linspace(0, 1, 100)
        
        dx_dt = (-3*x0 + 9*x1 - 9*x2 + 3*x3)*t**2 + (6*x0 - 12*x1 + 6*x2)*t + (-3*x0 + 3*x1)
        dy_dt = (-3*y0 + 9*y1 - 9*y2 + 3*y3)*t**2 + (6*y0 - 12*y1 + 6*y2)*t + (-3*y0 + 3*y1)
        
        d2x_dt2 = (-6*x0 + 18*x1 - 18*x2 + 6*x3)*t + (6*x0 - 12*x1 + 6*x2)
        d2y_dt2 = (-6*y0 + 18*y1 - 18*y2 + 6*y3)*t + (6*y0 - 12*y1 + 6*y2)
        
        curvature_power = (dx_dt*d2y_dt2 - dy_dt*d2x_dt2)**2/((dx_dt**2 + dy_dt**2)**3)
        
        bezier_x = (1-t)**3*x0 + 3*(1-t)**2*t*x1 + 3*(1-t)*t**2*x2 + t**3*x3
        bezier_y = (1-t)**3*y0 + 3*(1-t)**2*t*y1 + 3*(1-t)*t**2*y2 + t**3*y3
        
        length = 0
        for i in range(len(bezier_x)-1):
            length += ((bezier_x[i+1] - bezier_x[i])**2 + (bezier_y[i+1] - bezier_y[i])**2)**(0.5)
            
        w1, w2 = 1, 1
        return w1 * sum(curvature_power) + w2 * length
    
    def optimization(self, middle_path):
        fx, fy, dfx_dt, dfy_dt = self.generate_path_trajectory(middle_path)
        args = (fx, fy, dfx_dt, dfy_dt)
        
        #初期値生成 t=0.5は決め打ち
        t = 0.5
        x0, x3 = self.start[0], fx(t)
        x1, x2 = x0 + (x3 - x0)*1/3, x0 + (x3 - x0)*2/3 
        initial_x = [x1, x2, t] #x1,  x2, t
        fun = self.objective_function
        options = {'maxiter':100000}
        bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0, 1)]
        result = optimize.minimize(fun, initial_x, method='SLSQP', args = args, bounds = bounds, options=options)
        
        return result
    
    def get_optimization_info(self):
        param_list = []
        value_list = []
        for middle_path in self.network:
            result = self.optimization(middle_path)
            param_list.append(result.x)
            value_list.append(result.fun)

        return param_list, value_list
    
    def generate_network(self, file_name):
        #csvからnetwork情報を取得
        with open(file_name) as file:
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
        
        return network
    
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
        
    start, goal, start_theta, goal_theta = [0, 0], [30, 5], 0, 0
    planner1 = SplineBasePathPlanning(start, start_theta, network)
    planner2 = SplineBasePathPlanning(goal, goal_theta, network)
    param_list1, value_list1 = planner1.get_optimization_info()
    param_list2, value_list2 = planner2.get_optimization_info()
    #print(param_list1, value_list1)
    #print(param_list2, value_list2)
    
    sum_value_list = []
    for i in range(len(network)):
        middle_path = network[i]
        x1, x2 = param_list1[i], param_list2[i]
        fx, fy, dfx_dt, dfy_dt = planner1.generate_path_trajectory(middle_path)
        args = (fx, fy, dfx_dt, dfy_dt)
        t = np.linspace(0, 1, 100)
        middle_x, middle_y = fx(t), fy(t)
        bezier_x1, bezier_y1 = planner1.generate_bezier(x1, *args)
        bezier_x2, bezier_y2 = planner2.generate_bezier(x2, *args)
        sum_value = value_list1[i] + value_list2[i]
        sum_value_list.append(sum_value)
        #plot.vis_all_path(middle_x, middle_y, bezier_x1, bezier_y1, bezier_x2, bezier_y2)
        
    label = [i for i in range(len(sum_value_list))]
    plt.bar(label, value_list1, tick_label=label, align="center", label="start→middle")
    plt.bar(label, value_list2, tick_label=label, bottom=value_list1, align="center",label="middle→goal")
    plt.xlabel("Index")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()
    print(value_list1)
    print(value_list2)
    min_index = sum_value_list.index(min(sum_value_list))
    middle_path = network[min_index]
    x1, x2 = param_list1[min_index], param_list2[min_index]
    fx, fy, dfx_dt, dfy_dt = planner1.generate_path_trajectory(middle_path)
    args = (fx, fy, dfx_dt, dfy_dt)
    t = np.linspace(0, 1, 100)
    middle_x, middle_y = fx(t), fy(t)
    bezier_x1, bezier_y1 = planner1.generate_bezier(x1, *args)
    bezier_x2, bezier_y2 = planner2.generate_bezier(x2, *args)
    plot.vis_all_path(middle_x, middle_y, bezier_x1, bezier_y1, bezier_x2, bezier_y2)
    
if __name__ == '__main__':
    main()