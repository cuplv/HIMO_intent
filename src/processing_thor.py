import numpy as np
import matplotlib.image as mpimg
import random
import pandas as pd
import copy

def rotate_data(x, y):
    theta = np.radians(-41)
    rotation_mat = [[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0,0,1]]

    x = np.array(x)
    y = np.array(y)    

    xy = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
    xyz = np.hstack([xy, np.zeros([xy.shape[0], 1])])
    rotated_data = np.array(list(map(lambda x: np.matmul(rotation_mat, x), xyz)))

    return rotated_data[:,0], rotated_data[:,1]        

def change_size(x, y):
    xlim = 18000
    ylim = xlim/2

    x = np.array(x)
    y = np.array(y)

    xy = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
    xy = xy[((xy[:,0] < xlim) & (xy[:,1] < ylim)) | (np.isnan(xy[:,0]))]

    x = (xy[:,0] / xlim) * 100
    y = (xy[:,1] / (2*ylim)) * 100

    return x, y

def digitize(x, y):
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]    
    
    x = np.digitize(x, bins=range(0,100,1))
    y = np.digitize(y, bins=range(0,100,1))
    return x, y

def fill_in_the_gap(observation):
    [observation.pop(idx) for idx, _ in enumerate(observation) if idx < len(observation)-1 and observation[idx] == observation[idx+1]]
    
    cnt = 0
    for idx, p in enumerate(observation[1:]):
        current_x, current_y = p[0], p[1]
        s_x, s_y = 1, 1
        prev_x, prev_y = observation[cnt+idx][0], observation[cnt+idx][1]
        if current_x < prev_x:
            s_x = -1
        if current_y < prev_y:
            s_y = -1        

        if prev_x == current_x:
            xlist = [prev_x]
        else:
            xlist = list(range(prev_x, current_x, 1*s_x))
        if prev_y == current_y:
            ylist = [prev_y]          
        else:
            ylist = list(range(prev_y, current_y, 1*s_y))

        xlist_rightmost = xlist[-1]
        ylist_rightmost = ylist[-1]    

        while len(xlist) != len(ylist):
            if len(xlist) < len(ylist):
                xlist.insert(len(xlist), xlist_rightmost)
            if len(ylist) < len(xlist):
                ylist.insert(len(ylist), ylist_rightmost)                
        observation.pop(cnt+idx)
        [observation.insert(cnt+idx+i, x) for i, x in enumerate(list(zip(xlist, ylist)))]
        cnt += len(list(zip(xlist, ylist)))-1
    return observation

class MapImg():
    def __init__(self, exp_id):
        # Load
        if exp_id == 1 or exp_id == 2:
            img = mpimg.imread('./Thor/orebro_map.png')
        elif exp_id == 3:
            img = mpimg.imread('./Thor/orebro_map_exp3.png')

        img = img[:,:,0]*255 # 0-1 to 0-255
        img = np.flipud(img).T 

        # Scale up
        img_x, img_y = np.where(img!=0)
        img_x = img_x*10 - 11000
        img_y = img_y*10 - 10000

        # Sampling
        sampling_rate = 1
        sample_idx = random.sample(range(img_x.shape[0]), int(img_x.shape[0]/sampling_rate))
        img_x = img_x[sample_idx]
        img_y = img_y[sample_idx]

        # Rotation
        img_x, img_y = rotate_data(img_x, img_y)
        
        # Move left bottom to (0,0)
        self.min_x = min(img_x)
        self.min_y = min(img_y)
        img_x -= self.min_x
        img_y -= self.min_y
        
        # Change the size
        img_x, img_y = change_size(img_x, img_y)
        
        self.x = img_x
        self.y = img_y
        
        # Digitize
        img_x_digit, img_y_digit = digitize(img_x, img_y)

        self.xd = img_x_digit
        self.yd = img_y_digit
        
class Trajectory():
    def __init__(self, exp_id, run_id, map_img, agent_names):
        self.map_img = map_img
        self.csv_filename_6d = f'./Thor/Exp_{exp_id}_run_{run_id}_6D.tsv'
        self.csv_filename_3d = f'./Thor/Exp_{exp_id}_run_{run_id}.tsv'

        self.agent_names  = agent_names
        self.load_data()
        
    def load_data(self):
        traj_data = dict()
        traj_data_d = dict()
        traj_data_3d = dict()
        data_6d = pd.read_csv(self.csv_filename_6d, skiprows = 10, sep='\t')
        data_3d = pd.read_csv(self.csv_filename_3d, skiprows = 10, sep='\t')

        for idx, name in enumerate(self.agent_names):
            if name.startswith("lidar"):
                traj_data[name] = data_6d.loc[:,'Velodyne X':'Y.11'].values
            elif name.startswith("robot"):
                traj_data[name] = data_6d.loc[:,'Citi_1 X':'Y.12'].values
            else:
                traj_data[name] = data_6d.loc[:,f'Helmet_{idx+2} X':f'Y.{idx+2}'].values

                x = data_3d.loc[:,[f'Helmet_{idx+2} - 1 X', f'Helmet_{idx+2} - 2 X', f'Helmet_{idx+2} - 3 X', f'Helmet_{idx+2} - 4 X']].values
                y = data_3d.loc[:,[f'Helmet_{idx+2} - 1 Y', f'Helmet_{idx+2} - 2 Y', f'Helmet_{idx+2} - 3 Y', f'Helmet_{idx+2} - 4 Y']].values
                x = [x/s if s!=0 else 0 for x, s in zip(x.sum(axis=1), (x!=0).sum(axis=1))]
                y = [y/s if s!=0 else 0 for y, s in zip(y.sum(axis=1), (y!=0).sum(axis=1))]
                traj_data_3d[name] = np.vstack([x, y]).T

            # remove (0,0)
            traj_data[name][(traj_data[name][:,0] == 0) & (traj_data[name][:,1] == 0)] = None

            # Combine 3d and 6d data
            if not name.startswith("lidar") and not name.startswith("robot"):
                traj_data_3d[name][(traj_data_3d[name][:,0] == 0) & (traj_data_3d[name][:,1] == 0)] = None
                traj_data_3d[name][~np.isnan(traj_data[name][:,0]),:] = traj_data[name][~np.isnan(traj_data[name][:,0]),:]
                traj_data[name] = traj_data_3d[name]

            # rotate
            traj_data[name][:,0], traj_data[name][:,1] = rotate_data(traj_data[name][:,0], traj_data[name][:,1])

            # translation
            traj_data[name][:,0] -= self.map_img.min_x
            traj_data[name][:,1] -= self.map_img.min_y

            # change size
            x, y = change_size(traj_data[name][:,0], traj_data[name][:,1])
            traj_data[name] = np.hstack([x.reshape(-1,1),y.reshape(-1,1)])

            traj_data_d[name] = copy.deepcopy(traj_data[name])
            
            # Digitize ---------------------------------------------
            x, y = traj_data_d[name][:,0], traj_data_d[name][:,1]
            x, y = digitize(x, y)
            xy = np.vstack([x, y]).T
                        
            # no robot in exp 1
            if len(x) == 0:
                continue

            # unique values
            xy = [list(xy[0])] + [[x,y] for idx, (x, y) in enumerate(xy[:-1]) if (xy[idx+1,0] != xy[idx,0]) or (xy[idx+1,1] != xy[idx,1])]
            xy = np.array(xy)
            if xy.shape[0] > 1 and (xy[0] == xy[1]).all():
                xy = xy[1:]
            traj_data_d[name] = xy
            # -------------------------------------------------------
            
            # Restore the data
            x = traj_data_d[name][:,0]
            y = traj_data_d[name][:,1]
            traj_data_d[name] = np.array(fill_in_the_gap(list(zip(x, y))))        
        
        self.data = traj_data
        self.data_d = traj_data_d
        
    