#此函数用于绘图，可以绘制网络的输入、输出，以及预期图像。其中input为图像，维度是四维；idx是input数据集中要画的图像索引
import matplotlib.pyplot as plt
import numpy as np

def figure(input,idx, Title):

    data_dim = np.shape(input)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(1, data_dim[1] + 1, 1)
    y = np.arange(1, data_dim[2] + 1, 1)
    y, x = np.meshgrid(y, x)
    ax.plot_surface(x, y, ((input[idx, :, :, 0])))
    plt.title(Title)
