# Cat vs Dog Classifier Using PyTorch CNN

Developed a CNN-based image classifier to distinguish cats and dogs using PyTorch. Designed two versions of the model: a full model with high-resolution inputs for production-level training, and a lightweight variant with smaller input size and reduced parameters for rapid iteration and debugging. The lightweight model accelerated development by over 70%, making it easier to validate data pipelines and training logic. Implemented dynamic flatten size calculation to enhance model flexibility across different image resolutions. Achieved 90%+ training accuracy and supported inference on new images. Optimized the workflow to align with real-world ML development practices.

### model.py     Full model 模型构建
### light_model.py        Lightweight CNN
### train.py              Full training 负责模型训练，并保存模型文件
### train_debug.py        Fast debug version
### predict.py            Predict new image 加载模型，对单张图片做推理
### evaluate.py           加载训练好的模型，对整个数据集（train或val）进行评估，输出准确率、混淆矩阵等评估指标
### dataset.py            Data loading logic 数据准备
### cat_dog_cnn.pth       Trained model，训练并保存模型 
### streamlit_app.py      基于 Streamlit 的轻量级 Web 应用，图像识别项目，可以上传图片识别猫狗


### Lightweight CNN Version for Rapid Prototyping

To accelerate development and debugging, a lightweight CNN model was designed alongside the main architecture. This version uses:
1. Smaller input size (64×64 instead of 224×224)
2. Fewer convolutional layers (2 instead of 3)
3. Smaller fully connected layers

This approach significantly reduces training time and is ideal for:
1. Verifying data pipelines
2. Testing training logic
3. Debugging without GPU

Once verified, the full version of the model (with 224×224 input and 3 convolutional layers) is used for final training and evaluation
优化：
轻量版项目建议：
step 1: 新建一个轻量调试版（跑得快、方便调试，效果一般即可）：
快速调试版优化点：light_model.py轻量cnn模型
        original_model vs light_model
项目	    原值	    调试值	    原因
图片尺寸	224x224	    64x64   	极大减少计算量
模型层数	3 层卷积	  2 层  	 简化模型结构
批次大小	32	         16	        内存低时更稳
epoch 数	5	        保持 5     	少于 5 可能没学习效果
GPU/CPU	    任意	        不变	            用 GPU 更快（有则用）


Step2: 数据处理时将图像 resize 成 64x64
在 dataset.py 里加一个参数开关：
def get_cat_dog_dataloader(batch_size=16, image_size=64):  # 新增 image_size
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),   # 改为动态尺寸
        transforms.ToTensor(),
    ])
    ...

Step3: add train_debug.py test version

### Features
1. Implemented with PyTorch
2. Dynamic flatten size calculation for flexible input
3. Two versions: full model and lightweight model
4. Supports prediction on new images


# 基于 PyTorch 的猫狗图像分类器（支持网页上传预测）
这是一个使用 PyTorch 实现的图像二分类项目，旨在判断上传的图片中是“猫”还是“狗”。整个项目完整覆盖了深度学习项目的全流程，包括数据处理、模型构建、训练评估、模型保存与加载、命令行预测、以及最终的网页交互式图片上传预测功能，具备良好的工程结构和可扩展性。

### 实现功能（按开发流程划分）：
1. 数据预处理与加载（dataset.py）
    1.1. 基于 CIFAR-10 数据集中筛选猫（label=3）和狗（label=5）
    1.2. 使用 torchvision 的 transforms 进行图像归一化、缩放
    1.3. 封装 get_cat_dog_dataloader() 函数，统一训练/预测数据入口

2. 模型设计（model.py）
    2.1. 使用 3 层卷积神经网络（CNN）提取图像特征
    2.2. 使用 AdaptiveAvgPool2d + 动态 flatten，支持任意输入尺寸
    2.3. 全连接层进行二分类输出（Cat / Dog）

3. 模型训练与保存（train.py）
    3.1.使用交叉熵损失函数 + Adam 优化器
    3.2.每轮输出平均损失（Loss）
    3.3 训练完成后自动保存模型权重为 cat_dog_cnn.pth

4. 模型评估（evaluate.py）
    4.1. 加载模型并在训练集进行评估
    4.2. 输出准确率、分类报告（precision, recall, F1）
    4.3. 可视化混淆矩阵（使用 matplotlib + seaborn）

5. 命令行预测（predict.py）
    支持传入 --image 参数运行预测：

6. 网页上传图片预测（streamlit_app.py）
    6.1 使用 Streamlit 快速构建图像上传页面
    6.2 支持任意尺寸图片上传并实时展示预测结果
    6.3 用户友好界面，适合部署与演示（未来可部署至 Hugging Face Spaces）