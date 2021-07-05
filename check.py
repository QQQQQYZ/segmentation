import numpy as np
import os
ROOT = "/media/qing/DATA3/Learning-Based-Terrain-Traversability-Analysis-master/img_velo_label_327-002"
data = np.load(ROOT+'/label/02/pts_l000000.npz')
np.set_printoptions(threshold=np.inf)
print(data['pts_l'].shape)