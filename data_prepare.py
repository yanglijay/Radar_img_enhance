import numpy as np
import scipy.io as scio
import sys
import matplotlib.pyplot as plt
from IPython.core.display import display
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
def data_prepare(data_folder, file_idx):
#    data_folder='/home/caffe/Desktop/Radar_enhance/Code/cnn_ljyang/radar_sim/dataset/'
#    file_idx=1

    sys.path.insert(0,data_folder)

    data_norm_out = np.array([])
    expect_img_norm_out = np.array([])

    idx = 0
    while idx<np.alen(file_idx):
        display(('Current reading circle = '+str(idx)))
        display(('Total reading circle = '+str(-1+np.alen(file_idx))))
        data_file = data_folder+'dataset_'+str(file_idx[idx])+'.mat'
        data_mat=scio.loadmat(data_file)
        data=data_mat['data_in']
        expect_img=data_mat['data_object']

        #normalizing data
        data_norm=(data-np.average(data))/np.std(data)
        data_norm=abs(data_norm)#只保留数据的幅度，舍去相位。注意只在设计RVCNN的时候有效，当设计CVCNN的时候，这行要去掉！
        expect_img_norm=(expect_img-np.average(expect_img))/np.std(expect_img)

        data_norm_out = np.append(data_norm_out, data_norm)
        expect_img_norm_out = np.append(expect_img_norm_out, expect_img_norm)
        idx += 1
    data_norm_out=data_norm_out.reshape(np.alen(file_idx), data.shape[0], data.shape[1],1)
    expect_img_norm_out=expect_img_norm_out.reshape(np.alen(file_idx), data.shape[0], data.shape[1],1)
    '''
    data_dim=np.shape(data_norm)
    fig=plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    x=np.arange(1,data_dim[0]+1,1)
    y=np.arange(1,data_dim[1]+1,1)
    y, x = np.meshgrid(y, x)
    ax.plot_surface(x,y,10*np.log(abs(data_norm)),cmap=cm.gnuplot)

    fig=plt.figure()
    ax2 = fig.add_subplot(111,projection='3d')
    ax2.plot_surface(x,y,10*np.log(abs(expect_img_norm)),cmap=cm.gnuplot)
    '''
    return [data_norm_out, expect_img_norm_out]