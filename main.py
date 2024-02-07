import scipy.optimize as optimize
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import spline_optimization 
import csv
import util
import plot
from param import Parameter as p
import GenerateInitialPath
import objective_function
import constraints

def main():
    #csvからnetwork情報を取得
    with open("network_rectangle.csv") as file:
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
        
    start, goal, start_theta, goal_theta = [5, 3], [30, 5], 0, 0
    
    planner1 = spline_optimization.SplineBasePathPlanning(start, start_theta, network)
    planner2 = spline_optimization.SplineBasePathPlanning(goal, goal_theta, network)
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
    
    #start->middleの最適化
    #paramの設定
    p.initial_x, p.initial_y, p.initial_theta = start[0], start[1], start_theta
    p.terminal_x, p.terminal_y, p.terminal_theta = fx(x1[2]), fy(x1[2]), np.arctan2(dfy_dt(x1[2]), dfx_dt(x1[2]))
    
    p.N = int(((p.terminal_x - p.initial_x)**2 + (p.terminal_y - p.initial_y)**2)**(0.5))*2
    p.dt = 1
    
    #初期解の生成
    initial_x, initial_y = GenerateInitialPath.interp_1d([p.initial_x, p.terminal_x], [p.initial_y, p.terminal_y])
    x, y, theta, phi, v = GenerateInitialPath.generate_initialpath2(initial_x, initial_y)
    #x, y, theta, phi, v = GenerateInitialPath.generate_initialpath_randomly(initial_x, initial_y)
    xs, ys, thetas, phi, v = GenerateInitialPath.initial_zero(0.1)
    print(theta)
    trajectory_matrix = np.array([x, y, theta, phi, v])
    trajectory_vector = util.matrix_to_vector(trajectory_matrix)

    #目的関数の設定
    func = objective_function.objective_function2
    jac_of_objective_function = objective_function.jac_of_objective_function2

    args = (1, 1)

    #制約条件の設定
    cons = constraints.generate_cons_with_jac()

    #変数の範囲の設定
    bounds = constraints.generate_bounds()

    #オプションの設定
    options = {'maxiter':10000, 'ftol': 1e-6}


    #最適化を実行
    result1 = optimize.minimize(func, trajectory_vector, args = args, method='SLSQP', jac = jac_of_objective_function, constraints=cons, bounds=bounds, options=options)
    result_x1, result_y1, result_theta1, _, _ = util.generate_result(result1.x)
    print(result1)
    
    #middle->goalの最適化
    #paramの設定
    p.initial_x, p.initial_y, p.initial_theta = fx(x2[2]), fy(x2[2]), np.arctan2(dfy_dt(x2[2]), dfx_dt(x2[2]))
    p.terminal_x, p.terminal_y, p.terminal_theta = goal[0], goal[1], goal_theta
    
    p.N = int(((p.terminal_x - p.initial_x)**2 + (p.terminal_y - p.initial_y)**2)**(0.5))*2
    p.dt = 1
    
    #初期解の生成
    initial_x, initial_y = GenerateInitialPath.interp_1d([p.initial_x, p.terminal_x], [p.initial_y, p.terminal_y])
    x, y, theta, phi, v = GenerateInitialPath.generate_initialpath2(initial_x, initial_y)
    #x, y, theta, phi, v = GenerateInitialPath.generate_initialpath_randomly(initial_x, initial_y)
    xs, ys, thetas, phi, v = GenerateInitialPath.initial_zero(0.1)
    trajectory_matrix = np.array([x, y, theta, phi, v])
    trajectory_vector = util.matrix_to_vector(trajectory_matrix)

    #目的関数の設定
    func = objective_function.objective_function2
    jac_of_objective_function = objective_function.jac_of_objective_function2

    args = (1, 1)

    #制約条件の設定
    cons = constraints.generate_cons_with_jac()

    #変数の範囲の設定
    bounds = constraints.generate_bounds()

    #オプションの設定
    options = {'maxiter':10000, 'ftol': 1e-6}


    #最適化を実行
    result2 = optimize.minimize(func, trajectory_vector, args = args, method='SLSQP', jac = jac_of_objective_function, constraints=cons, bounds=bounds, options=options)
    result_x2, result_y2, result_theta2, _, _ = util.generate_result(result2.x)
    print(result2)
    
    t = np.linspace(x1[2], x2[2], 100)
    middle_x, middle_y = fx(t), fy(t)
    #結果の表示
    plot.vis_all_path(middle_x, middle_y, result_x1, result_y1, result_x2, result_y2)
    
    trajectory_matrix = util.vector_to_matrix(np.array(trajectory_vectors[min_index]))
    
    x2, y2, theta2 = trajectory_matrix[0], trajectory_matrix[1], trajectory_matrix[2]
    print(list(np.r_[result_x1, x2[13:-5], result_x2]), list(np.r_[result_y1, y2[13:-5], result_y2]), list(np.r_[result_theta1, theta2[13:-5], result_theta2]))
if __name__ == '__main__':
    main()