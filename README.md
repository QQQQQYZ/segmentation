#预处理数据的代码在data_utils/fhy4_datautils_test.py, 提取点云和label的方法在get_pts_l函数中，提取点云和速度信息的方法在get_pts_vel中  
#预加载数据的代码在data_utils/fhy4_Sem_Loader.py，归一化，随机噪声，训练集划分均在里面。SemKITTI_Loader类和PointsAndVel_Loader类用于生成训练数据    
#语义分割训练运行fhy_pcdseg.py  
#fhy_pcdvis.py可将分割后的点云投影到图片上并显示  
#速度预测训练运行qyz_train.py  
#pointnet模型存在/model/fhy_pointnet1.py中，回归模型存在/model/qyz_mlp.py中
