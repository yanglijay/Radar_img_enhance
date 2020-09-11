#此函数用于对数据进行标准正态分布处理
import numpy as np
def data_std_norm_dist(data, expect_img, data_norm_mag_out, data_norm_pha_out, expect_img_norm_out):
    # normalizing data
    data_mag = abs(data)  # 取模
    data_pha = np.angle(data)  # 取相

    for k in range(len(data)):
        data_norm_mag = (data_mag[k] - np.average(data_mag[k])) / np.std(data_mag[k])  # 对每一张图的模进行正态归一化
        data_norm_pha = (data_pha[k] - np.average(data_pha[k])) / np.std(data_pha[k])  # 对每一张图的相位进行正态归一化
        expect_img_norm = (expect_img[k] - np.average(expect_img[k])) / np.std(expect_img[k])  # 对每一张图的预期图像进行正态归一化
        data_norm_mag_out = np.append(data_norm_mag_out, data_norm_mag)  # 拼接数据
        data_norm_pha_out = np.append(data_norm_pha_out, data_norm_pha)  # 拼接数据
        expect_img_norm_out = np.append(expect_img_norm_out, expect_img_norm)  # 拼接数据
    return [data_norm_mag_out, data_norm_pha_out, expect_img_norm_out]