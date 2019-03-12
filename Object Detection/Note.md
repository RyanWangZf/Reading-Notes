# Note of Object Detection Paper Reading #

## Region-based CNN (R-CNN) ##
Ref: [https://blog.csdn.net/briblue/article/details/82012575]  
### Main Steps
1. 生成候选框图  
R-CNN 使用了 Selective Search 的方法进行bounding box的生成，这是一种Region Proposal的方法。  
生成的2000个Bbox使用NMS计算IoU指标剔除重叠的位置。  
2. 针对候选图作embedding的抽取  
Bbox直接Resize为227*227供AlexNet的输入，再Resize之前对所有BBox进行padding。  
3. 使用分类器对embedding训练和分类  
### Training
使用TL，在VOC数据集上进行fine-tune。原始ImageNet上训练的网络能预测1000类，这里采用了20类加背景一共21类的输出方式。  
### Identification



