-Gesture Recognize on Caffe-

Resume:
using trained caffe model and openCV core funtions,recognize frames from camera with caffe net method(forward culculate). it's a try for applying deep learning network that there are no trains but classification. it has python and cpp two versions,the reason why cpp version exits is that pyInstaller failed to packet and cpp is easier to packet and obviously works faster. the work of the project is mainly that

1. initillize caffe net and openCV class,
2. call camera to get gesture frames,
3. put frames into input blob,forward culculate the net,get 61 predictions from output blob,
4. rank the predictions,and put some text on the frames classified.
5. save images and video int test folder.

说明：
openCV调用摄像头获取图像帧，caffe初始化网络模型对帧进行分类(classification)。是对已训练caffe model的应用，有python，C++版本(可打包，可移植)。 
输入是摄像头的获取图像帧，输出是caffe net的输出层数据，进一步处理得到对应61个标签的概率。中间过程是权重网络的初始化，有向图的前向运算。
