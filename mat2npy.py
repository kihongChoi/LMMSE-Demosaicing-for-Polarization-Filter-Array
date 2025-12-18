"""
Data/D_Matrix.mat 를 Data/D_Matrix.npy 로 변환
"""

import scipy.io as sio
import numpy as np

d_matrix_data = sio.loadmat('Data/D_Matrix.mat')
D = d_matrix_data['D']
np.save('Data/D_Matrix.npy', D)