#pragma once
#include <caffe/caffe.hpp>
#include <string>
#include <opencv2/opencv.hpp>
#define IF_TEST_MODE(x) x;

using namespace caffe;
using namespace std;

typedef pair<string, float> Prediction;	//标签预测结果是一对键值。

////static 函数 为静态非全局函数，仅供本源码文件使用。
// 函数Argmax()需要用到的子函数，vector排序用
static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}
// 返回预测结果中概率从大到小的前N个预测结果的索引
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);
	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

//@caffe预测工具类
class GestureClassifier2
{
	string model_file;
	string trained_file;
	string mean_file;
	string label_file;

	////可复用参数。
	boost::shared_ptr<Net<float> > net_;	//Dtype为float的Net类，共享所有权智能指针shared_ptr指向该类对象
											//类对象每次引用次数+1，断开连接时-1。引用次数为0时自动析构对象。
											//.调用指针类成员，->调用指针所指对象的类成员。
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	vector<string> labels_;
	vector<cv::Mat> input_channels;



public:
	inline long getNetPtrUseCount() {
		IF_TEST_MODE(cout << "Message: classifier: \tshared_ptr net_ use count:" << net_.use_count() << endl)
		return net_.use_count();
	}

	inline void initFilePath(string model = R"(.\data\hand_net.prototxt)",
		string trained = R"(.\data\1miohands-v2.caffemodel)",
		string mean = R"(.\data\227x227-TRAIN-allImages-forFeatures-0label-227x227handpatch.mean)",
		string label = R"(.\data\s1.txt)") {
		//// 定义模型配置文件，模型文件，均值文件，标签文件路径
		model_file = model;
		trained_file = trained;
		mean_file = mean;
		label_file = label;
		
	}

	void init();
	cv::Mat getIntputMat(const string);
	cv::Mat getIntputMat(cv::Mat&);	//直接引用从摄像头获得帧的色彩矩阵
	void classify(string);
	Prediction getBestPrediction(cv::Mat&);


	GestureClassifier2() {		//自动初始化
		initFilePath();
		init();
	};		
	GestureClassifier2(int i) {	//需手动init
		LOG(WARNING) << "GestureClassifier2 need manual init";
	};	
	~GestureClassifier2() {};
};

