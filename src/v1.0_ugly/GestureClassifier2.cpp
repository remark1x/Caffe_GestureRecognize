#include "GestureClassifier2.h"



void GestureClassifier2::init()
{
	if (model_file.empty()) {
		LOG(WARNING)<< " Should call initFilePath first or set file path first";
		return;
	}

	//// 定义变量
	Blob<float>* input_layer;

	Caffe::set_mode(Caffe::CPU); // 使用

	net_.reset(new Net<float>(model_file, TEST));	// 加载配置文件，设定模式为分类，实例化net对象
	net_->CopyTrainedLayersFrom(trained_file);		// 根据训练好的模型修改模型参数

													//// 输入层信息
	input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	LOG(INFO) << "classifier: num_channels_: " << num_channels_;
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());	//cv::mat shape_[3] and  shape_[2].

	//// 处理均值文件，得到均值图像
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; i++)
	{
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	cv::Mat mean;
	merge(channels, mean);
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);

	//// 获取标签
	std::ifstream labels(label_file.c_str());
	string line;
	while (getline(labels, line))
		labels_.push_back(string(line));
	//判断标签的类数和模型输出的类数是否相同
	Blob<float>* output_layer = net_->output_blobs()[0];
	LOG(INFO) << "output_layer dimension: " << output_layer->channels()
		<< "; labels num ber: " << labels_.size();

	// 预测图像信息
	input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
	net_->Reshape();

	//将input_channels指向模型的输入层相关位置
	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); i++)
	{
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += width * height;
	}
}

cv::Mat GestureClassifier2::getIntputMat(const string img_file_path)
{
	//// 改变图像的大小、通道、数据类型，去均值等
	cv::Mat img = cv::imread(img_file_path);
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)	cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)	cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)	cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)	cv::resize(sample, sample_resized, input_geometry_);
	else									sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)		sample_resized.convertTo(sample_float, CV_32FC3);
	else						sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	return sample_normalized;
}

cv::Mat GestureClassifier2::getIntputMat(cv::Mat & img)
{
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)	cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)	cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)	cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)	cv::resize(sample, sample_resized, input_geometry_);
	else									sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)		sample_resized.convertTo(sample_float, CV_32FC3);
	else						sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	return sample_normalized;
	
}

void GestureClassifier2::classify(string img_file)
{
	//// 处理好的数据保存在输入层（指针指向实现）
	cv::split(getIntputMat(img_file), input_channels);

	//// 调用模型进行预测
	net_->Forward();

	//// 获得输出层
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	vector<float> output = vector<float>(begin, end);

	//// 显示概率前N大的结果
	int N = 10;
	N = std::min<int>(labels_.size(), N);
	std::vector<int> maxN = Argmax(output, N);


	std::vector<Prediction> predictions;
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels_[idx], output[idx]));
	} 

	for (size_t i = 0; i < predictions.size(); ++i) {
		Prediction p = predictions[i];
		std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
			<< p.first << "\"" << std::endl;
	}
}

Prediction GestureClassifier2::getBestPrediction(cv::Mat & img)
{
	//// 处理好的数据保存在输入层（指针指向实现）
	cv::split(getIntputMat(img), input_channels);

	//// 调用模型进行预测
	net_->Forward();

	//// 获得输出层
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	vector<float> output = vector<float>(begin, end);

	//// 显示概率前N大的结果	【此处可改进为单一排序】
	int N = 1;
	N = std::min<int>(labels_.size(), N);
	std::vector<int> maxN = Argmax(output, N);

	Prediction bestP = std::make_pair(labels_[maxN[0]], output[maxN[0]]);

	return bestP;
}


