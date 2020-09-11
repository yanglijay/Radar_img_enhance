#雷达图像处理，回归问题

#释放GPU内存
#from numba import cuda
#    cuda.select_device(0)
#    cuda.close()

#指定GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.models import *
from keras.layers import *
from keras import optimizers
import random
import matplotlib.pyplot as plt
from figure import figure

epochs = 50
batch_size = 50

pics_train=500#用于训练的图像数目
pics_val=100  #用于验证的图像数目
pics_test=100 #用于测试的图像数目

pics_per_file = 50 #每一个文件里面的图像数目，等于matlab仿真中的target.num

files_for_train = int(np.ceil(pics_train/pics_per_file))#CNN训练所需的文件数目
files_for_val = int(np.ceil(pics_val/pics_per_file))#CNN验证所需的文件数目
files_for_test = int(np.ceil(pics_test/pics_per_file))#CNN测试所需的文件数目

from pic_prepare import pic_prepare

data_folder = '/home/caffe/Desktop/Radar_enhance/Code/cnn_ljyang/radar_sim/dataset_20200910/'#保存mat数据的文件夹目录
file_num = 500 #data_folder文件夹内的文件总数目
file_shuffled = random.sample(range(1, file_num+1), file_num)#产生不重复的文件索引。

file_train = file_shuffled[0:files_for_train]#选取用于训练的文件编号
file_val = file_shuffled[files_for_train:files_for_train+files_for_val]#选取用于验证的文件编号
file_test = file_shuffled[files_for_train+files_for_val:files_for_train+files_for_val+files_for_test]#选取用于测试的文件编号

[data_train, expect_train] = pic_prepare(data_folder, file_train)#训练集，得到一幅归一化后的输入数据及预期图像
[data_val, expect_val] = pic_prepare(data_folder, file_val)#验证集，得到一幅归一化后的输入数据及预期图像
[data_test, expect_test] = pic_prepare(data_folder, file_test)#测试集，得到一幅归一化后的输入数据及预期图像

model = Sequential()

model.add(Conv2D(filters=8,
                 kernel_size=(3, 3),
                 padding='Same',
                 kernel_initializer=initializers.RandomUniform(minval=0.05, maxval=0.95, seed=None),#指定下边界和上边界的均匀分布初始化
                 activation='relu',
                 input_shape=(data_train.shape[1], data_train.shape[2], data_train.shape[3])#输入图像的长、宽、通道数
                 )
          )
model.add(Conv2D(filters=16,
                 kernel_size=(5, 5),
                 padding='Same',
                 kernel_initializer=initializers.RandomUniform(minval=0.05, maxval=0.95, seed=None),#指定下边界和上边界的均匀分布初始化
                 activation='relu'
                 )
          )
model.add(Conv2D(filters=1,
                 kernel_size=(3, 3),
                 padding='Same',
                 kernel_initializer=initializers.RandomUniform(minval=0.05, maxval=0.95, seed=None),#指定下边界和上边界的均匀分布初始化
                 )
          )
'''
model.add(Conv2D(filters=8,
                 kernel_size=(5, 5),
                 padding='Same',
                 kernel_initializer=initializers.RandomUniform(minval=0.05, maxval=0.95, seed=None),#指定下边界和上边界的均匀分布初始化
                 activation='relu',
                 input_shape=(data_train.shape[1], data_train.shape[2], data_train.shape[3])#输入图像的长、宽、通道数
                 )
          )
model.add(Conv2D(filters=16,
                 kernel_size=(5, 5),
                 padding='Same',
                 kernel_initializer=initializers.RandomUniform(minval=0.05, maxval=0.95, seed=None),#指定下边界和上边界的均匀分布初始化
                 activation='relu'
                 )
          )
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 padding='Same',
                 kernel_initializer=initializers.RandomUniform(minval=0.05, maxval=0.95, seed=None),#指定下边界和上边界的均匀分布初始化
                 activation='relu'
                 )
          )
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 padding='Same',
                 kernel_initializer=initializers.RandomUniform(minval=0.05, maxval=0.95, seed=None),#指定下边界和上边界的均匀分布初始化
                 activation='relu'
                 )
          )
model.add(Conv2D(filters=1,
                 kernel_size=(3, 3),
                 padding='Same',
                 kernel_initializer=initializers.RandomUniform(minval=0.05, maxval=0.95, seed=None),#指定下边界和上边界的均匀分布初始化
                 )
          )
'''
optimizer = optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)

#optimizer=optimizers.SGD(lr=0.001,  decay=0.001, nesterov=False)

#model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
model.compile(optimizer=optimizer, loss='mean_absolute_error')

train_process = model.fit(x=data_train, y=expect_train, validation_data=(data_val, expect_val), epochs=epochs, batch_size=batch_size)

#weight_conv2d_1, bias_conv2d_1 = model.get_layer('conv2d_1').get_weights()
#weight_conv2d_2, bias_conv2d_2 = model.get_layer('conv2d_2').get_weights()
#weight_conv2d_3, bias_conv2d_3 = model.get_layer('conv2d_3').get_weights()
#weight_conv2d_4, bias_conv2d_4 = model.get_layer('conv2d_4').get_weights()
#weight_conv2d_5, bias_conv2d_5 = model.get_layer('conv2d_5').get_weights()

plt.figure()
p1, = plt.plot(train_process.history['loss'])
p2, = plt.plot(train_process.history['val_loss'])
plt.legend([p1, p2], ['train_loss', 'val_loss'])

#val_process = model.evaluate(data_val, expect_val)
#predict_result = model.predict(data_test, batch_size=data_test.shape[0], verbose=0)
predict_result = model.predict(data_test, batch_size=data_val.shape[0], verbose=0)

from sklearn.metrics import mean_absolute_error
MAE_predict = mean_absolute_error(predict_result.flatten(), expect_test.flatten())

figure(data_test, 50, 'Data test')
figure(predict_result, 50, 'Predict result')
figure(expect_test, 50, 'Expect test')
'''
####################显示每一层神经层的输出############################
layer_outputs = [layer.output for layer in model.layers[:5]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(data_test)
##################################################################
'''
display('Project end. Good luck !!!')
