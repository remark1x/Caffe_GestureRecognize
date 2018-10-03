
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "GestureClassifier.h"
using namespace std;

class cv2Capture{
	cv::VideoCapture* capture;
	cv::VideoWriter* outputVideo;
	string outputVideoPath;
	string outputImagePath;
	int fps;
	string windowName;
	GestureClassifier* classifier;

public:
	inline bool init(string outPath = ".\\data\\output.avi") {
		outputVideoPath = outPath;
		capture = new cv::VideoCapture(0);
		//获取当前摄像头的视频信息
		cv::Size S = cv::Size((int)capture->get(CV_CAP_PROP_FRAME_WIDTH),
			(int)capture->get(CV_CAP_PROP_FRAME_HEIGHT));
		outputVideo = new cv::VideoWriter;
		outputVideo->open(outPath, -1, 30.0, S, true);
		//CV_FOURCC('P','I','M','1')	//视频编码格式

		if (!outputVideo->isOpened()) { 
			cout << "Error: cv2Capture>>init>>outputVideo: Fail to open.Video path:" << outPath << endl;
			return false;
		}
		cout << "Process: cv2Capture>>init: Initilized cv2Capture." << endl;

		classifier = new GestureClassifier();
		classifier->initFilePath();
		classifier->init();
		cout << "Process: cv2Capture>>init: Initilized caffe Cesture Classifier." << endl;
		return true;
	}
	inline void captureVideo(bool saveVideo = true,bool mirrorEffect = true ) {
		cv::Mat frame;
		int count = 0;
		if (windowName.empty()) windowName = "outputFrames:";
		while (true) {
			//读取当前帧
			*capture >> frame;

			//处理当前帧
			if (mirrorEffect) cv::flip(frame, frame, 1);
			if (frame.empty()) break;
			++count;
			//输出当前帧
			cv::imshow(windowName, frame);

			//保存当前帧
			if (saveVideo) *outputVideo << frame;

			if (char(cv::waitKey(1)) == 'q') break;
		}
		cout << "Process: cv2Capture>>captureVideo: Finished.Total frames count:" << count << endl;

	}
	inline void classifyCapture() {		//【test】
		cv::Mat frame;
		int count = 0;
		Prediction bestPre;
		int font = cv::FONT_HERSHEY_TRIPLEX;
		string text;
		cv::Point point(200, 30);
		cv::Scalar redColor(255, 0, 0);
		cv::Scalar greenColor(0, 255, 0);
		cv::Scalar whiteColor(255, 255, 255);
		cv::Scalar blackColor(0, 0, 0);
		cv::Scalar color;
		while (true) {
			//读取当前帧
			*capture >> frame;

			//处理当前帧
			cv::flip(frame, frame, 1);
			if (frame.empty()) break;
			++count;
			if (count % 10 == 0) {
				bestPre = classifier->getBestPrediction(frame);
				text = bestPre.first + ':' + to_string(bestPre.second);
				if (bestPre.second > 0.5) color = greenColor;
				else color = whiteColor;
				if (bestPre.first == "garbage") color = blackColor;
				cv::putText(frame, text, point, font, 1.2, color, 2);
				cv::imwrite(R"(.\data\test_save_image\)" + to_string(count) + ".png", frame);
				cout << "\ntest:count[" << count << "]:" << bestPre.first << "\t" << bestPre.second << endl;
			}
			//输出当前帧
			cv::imshow("test for caffe classify and put best prediction", frame);

			//保存当前帧
			 *outputVideo << frame;

			if (char(cv::waitKey(1)) == 'q') break;
		}
	}

	cv2Capture(int i = 0) { init(); }
	~cv2Capture() {
	}

};