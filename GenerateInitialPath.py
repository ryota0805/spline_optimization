#初期パスを生成するファイル

import numpy as np
from scipy import interpolate
from param import Parameter as p
import random

########
#WayPointから3次スプライン関数を生成し、状態量をサンプリングする
########
#直線補間の生成
def interp_1d(xs, ys):
    t = np.linspace(0, 1, 2)
    
    linear_x = interpolate.interp1d(t, xs)
    linear_y = interpolate.interp1d(t, ys)
    
    t = np.linspace(0, 1, p.N)
        
    return linear_x(t), linear_y(t)

#3次スプライン関数の生成
def cubic_spline():   
    x, y = [], []
    for i in range(len(p.WayPoint)):
        x.append(p.WayPoint[i][0])
        y.append(p.WayPoint[i][1])
        
    tck,u = interpolate.splprep([x,y], k=3, s=0) 
    u = np.linspace(0, 1, num=p.N, endpoint=True)
    spline = interpolate.splev(u, tck)
    cubicX = spline[0]
    cubicY = spline[1]
    return cubicX, cubicY

#3次スプライン関数の生成(経路が関数の引数として与えられる場合)
def cubic_spline_by_waypoint(waypoint):   
    x, y = [], []
    for i in range(len(waypoint)):
        x.append(waypoint[i][0])
        y.append(waypoint[i][1])
        
    tck,u = interpolate.splprep([x,y], k=3, s=0) 
    u = np.linspace(0, 1, num=p.N, endpoint=True)
    spline = interpolate.splev(u, tck)
    cubicX = spline[0]
    cubicY = spline[1]
    return cubicX, cubicY

#x, yからΘとφを生成する
def generate_initialpath(cubicX, cubicY):
    #nd.arrayに変換
    x = np.array(cubicX)
    y = np.array(cubicY)
    
    #x, yの差分を計算
    deltax = np.diff(x)
    deltay = np.diff(y)
    
    #x, y の差分からthetaを計算
    #theta[0]を初期値に置き換え、配列の最後に終端状態を追加
    theta = np.arctan(deltay / deltax)
    theta[0] = p.initial_theta
    theta = np.append(theta, p.terminal_theta)
    
    #thetaの差分からphiを計算
    #phi[0]を初期値に置き換え配列の最後に終端状態を追加
    deltatheta = np.diff(theta)
    phi = deltatheta / p.dt
    phi[0] = p.initial_phi
    phi = np.append(phi, p.terminal_phi)
    
    #x,yの差分からvを計算
    #phi[0]を初期値に置き換え配列の最後に終端状態を追加
    v = np.sqrt((deltax ** 2 + deltay ** 2) / p.dt)
    v[0] = p.initial_v
    v = np.append(v, p.terminal_v)
    return x, y, theta, phi, v


#x, yからΘとφを生成する
def generate_initialpath2(cubicX, cubicY):
    t = np.linspace(0, p.N, p.N)
    
    fx = interpolate.Akima1DInterpolator(t, cubicX)
    fy = interpolate.Akima1DInterpolator(t, cubicY)
    
    dfx_dt = fx.derivative()
    dfy_dt = fy.derivative()
    
    #nd.arrayに変換
    x = fx(t)
    y = fy(t)
    
    #x, yの差分を計算
    dx_dt = dfx_dt(t)
    dy_dt = dfy_dt(t)
    
    #x, y の差分からthetaを計算
    #theta[0]を初期値に置き換え、配列の最後に終端状態を追加
    theta = np.arctan(dy_dt / dx_dt)
    
    #thetaの差分からphiを計算
    #phi[0]を初期値に置き換え配列の最後に終端状態を追加
    deltatheta = np.diff(theta)
    phi = deltatheta / p.dt
    phi[0] = p.initial_phi
    phi = np.append(phi, p.terminal_phi)
    
    #x,yの差分からvを計算
    #phi[0]を初期値に置き換え配列の最後に終端状態を追加
    v = np.sqrt((dx_dt ** 2 + dy_dt ** 2))
    return x, y, theta, phi, v


#theta, phi, vの初期値をランダムに生成
def generate_initialpath_randomly(cubicX, cubicY):
    t = np.linspace(0, p.N, p.N)
    
    fx = interpolate.Akima1DInterpolator(t, cubicX)
    fy = interpolate.Akima1DInterpolator(t, cubicY)
    
    #nd.arrayに変換
    x = fx(t)
    y = fy(t)
    
    theta = np.array([random.uniform(p.theta_min, p.theta_max) for i in range(p.N)])
    phi = np.array([random.uniform(p.phi_min, p.phi_max) for i in range(p.N)])
    v = np.array([random.uniform(p.v_min, p.v_max) for i in range(p.N)])
    
    return x, y, theta, phi, v

def initial_zero(a):
    return np.array([a for i in range(p.N)]), np.array([a for i in range(p.N)]), np.array([a for i in range(p.N)]), np.array([a for i in range(p.N)]), np.array([a for i in range(p.N)])
