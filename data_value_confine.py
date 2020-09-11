#此函数用于把数据限制在[0,1]区间内
import numpy as np
def data_value_confine(data, expect_img, data_norm_mag_out, data_norm_pha_out, expect_img_norm_out):

    data_mag = abs(data)  # 取模
    data_pha = np.angle(data)  # 取相

    data_norm_mag = data_mag-np.min(data_mag)# 对每一张图的模进行[0,1]归一化
    data_norm_mag = data_norm_mag/np.max(data_norm_mag)# 对每一张图的模进行[0,1]归一化
#    data_norm_mag = 2*data_norm_mag-1 # 最后对每一张图的模进行[-1,1]归一化

    data_norm_pha = (data_pha+np.pi)/(2*np.pi)# 对每一张图的相位进行[0,1]归一化
#    data_norm_pha = 2*data_norm_pha-1 # 最后对每一张图的模进行[-1,1]归一化

    expect_img_norm = expect_img-np.min(expect_img)# 对每一张图的预期图像进行[0,1]归一化
    expect_img_norm = expect_img_norm/np.max(expect_img_norm)# 对每一张图的预期图像进行[0,1]归一化
#    expect_img_norm = 2*expect_img_norm-1 # 最后对每一张图的模进行[-1,1]归一化

    data_norm_mag_out = np.append(data_norm_mag_out, data_norm_mag)  # 拼接数据
    data_norm_pha_out = np.append(data_norm_pha_out, data_norm_pha)  # 拼接数据
    expect_img_norm_out = np.append(expect_img_norm_out, expect_img_norm)  # 拼接数据
    return [data_norm_mag_out, data_norm_pha_out, expect_img_norm_out]