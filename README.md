# Gesture Recognize in Caffe<br> 

Using trained caffe model and openCV core funtions,recognize frames from camera with caffe net method(forward culculate). it's a try for applying deep learning network that there are no trains but classification. It has python and cpp two versions,the reason why cpp version exits is that pyInstaller failed to packet and cpp is easier to packet and obviously works faster. the work of the project is mainly that<br>
* initillize caffe net and openCV class,<br>
* call camera to get gesture frames,<br>
* put frames into input blob,forward culculate the net,get 61 predictions from output blob,<br>
* rank the predictions,and put some text on the frames classified.<br>
* save images and video int test folder.<br>

`project status`<br>
Only 8/60 kinds of gestures can be clssified correctly,but we still can study how to set up caffe enviroment and how to use caffe model to achieve thos we want.It's common parts.<br>

### 基于 Caffe 的静态手势识别
### 说明：
openCV调用摄像头获取图像帧，caffe初始化网络模型对帧进行分类(classification)。是对已训练caffe model的应用，有python，C++版本(可打包，可移植)。 <br>
输入是摄像头的获取图像帧，输出是caffe net的输出层数据，进一步处理得到对应61个标签的概率。中间过程是权重网络的初始化，有向图的前向运算。<br>

`项目当前状态说明`<br>
60个手势大概有8个可以比较好地分类出来(概率>0.5)，原因归结于测试数据与训练数据不一致以及训练数据的局限性。
所以本项目的主要目的改作：学习caffe框架，搭建caffe环境，利用caffe模型，实现模型的功能。<br>


## Dependency DownLoad:
With github's 100M-limit,have to put some important files into Baidu Yun.after free downloading,zap and set under root path.<br>
[BaiduYun](https://pan.baidu.com/s/1f8JUHpxMMmxRQ7Ej_DTHog) `kb4x`<br>

### 下载：
C++版本主要依赖文件(include,data,lib,thirdparty,部分超过100M限制)：<br>
* include：caffe框架的头文件，可以调用caffe所有类、方法、数据结构。<br>
* lib：已编译好的静态库libcaffe，与include头文件搭配编译使用。<br>
* data：只用于分类的权重模型、标签文件。<br>
* thirdparty：用于运行时调用的dll库。<br>
解压到根目录下，对应项目相对目录。<br>
[链接](https://pan.baidu.com/s/1f8JUHpxMMmxRQ7Ej_DTHog,"kb4x") `kb4x`<br>

## Release_v1 Download:
about 200M,can be used directly on x86_64 Windows.<br>
[BaiduYun](https://pan.baidu.com/s/1prBpO7BGj-9Ds4jGvh4QAA) `eypc`<br>

### 第一发布版本下载：
cpp版本项目已打包(源代码未重构，可正常使用旧版)，解压后约200M，仅包括需要的依赖项。<br>
[链接](https://pan.baidu.com/s/1prBpO7BGj-9Ds4jGvh4QAA,"eypc") `eypc`<br>

***
## Goals

> - [x] study caffe
> - [x] acheive gesture recognize
> - [ ] reconstruct with design pattern
>> - [ ] strategy pattern
>>>GestureRecognizer
>>>>FrameCapture  
>>>>FrameAnalyzer
>> - [ ] fatory pattern
>>>Caffe frame
>> - [ ] sigleton pattern
>>>only one instance
> - [ ] can be used easily for other caffe models

___
`caffe` `gesture recognize` `C++` `design pattern`
