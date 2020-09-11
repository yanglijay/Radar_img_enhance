#此函数用于读取matlab产生的数据
import numpy as np
import scipy.io as scio
import sys
import matplotlib.pyplot as plt
from IPython.core.display import display
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from data_std_norm_dist import data_std_norm_dist
from data_value_confine import data_value_confine

def pic_prepare(data_folder, file_idx):
#    data_folder='/home/caffe/Desktop/Radar_enhance/Code/cnn_ljyang/radar_sim/dataset_20200909/'
#    file_idx=1

    sys.path.insert(0, data_folder)

    data_norm_mag_out = np.array([])
    data_norm_pha_out = np.array([])
    expect_img_norm_out = np.array([])

    idx = 0
    while idx < np.alen(file_idx):#对file_idx对应的每一份文件依次进行处理
        display(('Current reading circle = '+str(idx)))
        display(('Total reading circle = '+str(-1+np.alen(file_idx))))
        data_file = data_folder+'dataset_'+str(file_idx[idx])+'.mat'#待处理的文件名
        data_mat = scio.loadmat(data_file)#读取文件数据
        data = data_mat['pic_in']#神经网络的输入图像数据
        expect_img = data_mat['pic_expect']#神经网络的目标图像数据

        ###########二选一##############
        [data_norm_mag_out, data_norm_pha_out, expect_img_norm_out] = data_value_confine(data, expect_img, data_norm_mag_out, data_norm_pha_out, expect_img_norm_out)
       #[data_norm_mag_out, data_norm_pha_out, expect_img_norm_out] = data_std_norm_dist(data, expect_img, data_norm_mag_out, data_norm_pha_out, expect_img_norm_out)
        ##############################

        '''
        #normalizing data
        data_mag=abs(data) #取模
        data_pha=np.angle(data) #取相

        for k in range(len(data)):
            data_norm_mag=(data_mag[k]-np.average(data_mag[k]))/np.std(data_mag[k])#对每一张图的模进行正态归一化
            data_norm_pha=(data_pha[k]-np.average(data_pha[k]))/np.std(data_pha[k])#对每一张图的相位进行正态归一化
            expect_img_norm=(expect_img[k]-np.average(expect_img[k]))/np.std(expect_img[k])#对每一张图的预期图像进行正态归一化
            data_norm_mag_out = np.append(data_norm_mag_out, data_norm_mag)#拼接数据
            data_norm_pha_out = np.append(data_norm_pha_out, data_norm_pha)#拼接数据
            expect_img_norm_out = np.append(expect_img_norm_out, expect_img_norm)#拼接数据
        '''
        idx += 1 #读取下一份文件
    data_norm_mag_out=data_norm_mag_out.reshape(np.alen(file_idx)*data.shape[0], data.shape[1], data.shape[2])#重排，第一个维度是图像数目，第二、三维度是图片尺寸
    data_norm_pha_out=data_norm_pha_out.reshape(np.alen(file_idx)*data.shape[0], data.shape[1], data.shape[2])#重排，第一个维度是图像数目，第二、三维度是图片尺寸

    data_norm_out = np.empty(shape=[np.alen(file_idx)*data.shape[0], data.shape[1], data.shape[2], 2])#生成网络的输入图像，第一个通道是模，第二个通道是相位
    data_norm_out[:, :, :, 0] = data_norm_mag_out
    data_norm_out[:, :, :, 1] = data_norm_pha_out
    #data_norm_out[:, :, :, 1] = data_norm_mag_out

    expect_img_norm_out=expect_img_norm_out.reshape(np.alen(file_idx)*data.shape[0], data.shape[1], data.shape[2], 1)#生成网络的预期图像

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