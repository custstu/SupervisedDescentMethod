
/**
 * @author: laughteroverflow
 * @date: Sept. 14 2017
 */

#include "FaceAligner.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>

void train(const std::string &list_path)
{
	const int num_levels = 5;

	std::vector<std::vector<int>> key_points =
	{ { CHIN, EYEBROW_LEFT_CENTER, EYEBROW_RIGHT_CENTER, EYE_LEFT_OUTER, EYE_LEFT_INNER, EYE_RIGHT_OUTER, EYE_RIGHT_INNER, NOSE_TIP, MOUTH_LEFT, MOUTH_RIGHT },
	  { CHIN, EYEBROW_LEFT_CENTER, EYEBROW_RIGHT_CENTER, EYE_LEFT_OUTER, EYE_LEFT_INNER, EYE_RIGHT_OUTER, EYE_RIGHT_INNER, NOSE_TIP, MOUTH_LEFT, MOUTH_RIGHT }, 
	  { CHIN, EYEBROW_LEFT_CENTER, EYEBROW_RIGHT_CENTER, EYE_LEFT_OUTER, EYE_LEFT_INNER, EYE_RIGHT_OUTER, EYE_RIGHT_INNER, NOSE_TIP, MOUTH_LEFT, MOUTH_RIGHT }, 
	  { CHIN, EYEBROW_LEFT_CENTER, EYEBROW_RIGHT_CENTER, EYE_LEFT_OUTER, EYE_LEFT_INNER, EYE_RIGHT_OUTER, EYE_RIGHT_INNER, NOSE_TIP, MOUTH_LEFT, MOUTH_RIGHT, EYE_LEFT_1, EYE_LEFT_2, EYE_LEFT_3, EYE_LEFT_4, EYE_RIGHT_1, EYE_RIGHT_2, EYE_RIGHT_3, EYE_RIGHT_4, MOUTH_TOP, MOUTH_BOTTOM }, 
	  { CHIN, EYEBROW_LEFT_CENTER, EYEBROW_RIGHT_CENTER, EYE_LEFT_OUTER, EYE_LEFT_INNER, EYE_RIGHT_OUTER, EYE_RIGHT_INNER, NOSE_TIP, MOUTH_LEFT, MOUTH_RIGHT, EYE_LEFT_1, EYE_LEFT_2, EYE_LEFT_3, EYE_LEFT_4, EYE_RIGHT_1, EYE_RIGHT_2, EYE_RIGHT_3, EYE_RIGHT_4, MOUTH_TOP, MOUTH_BOTTOM } };

	std::vector<HOGParam> hog_params = { HOGParam(VlHogVariantUoctti, 12, 4, 4, 1.f),
										 HOGParam(VlHogVariantUoctti, 10, 4, 4, 0.8f),
										 HOGParam(VlHogVariantUoctti, 8, 4, 4, 0.6f), 
										 HOGParam(VlHogVariantUoctti, 7, 4, 4, 0.5f), 
										 HOGParam(VlHogVariantUoctti, 6, 4, 4, 0.4f)};

	FaceAligner fa;
	if (fa.train(list_path, num_levels, key_points, hog_params))
	{
		std::cout << "Saving model..." << std::endl;
		if (fa.saveModel("../models/fa.bin"))
			std::cout << "Completed" << std::endl;
		else
			std::cout << "Fail" << std::endl;
	}
}

void test(const std::string &image_path)
{
	cv::CascadeClassifier fd;
	if(!fd.load("../models/haarcascade_frontalface_alt.xml"))
	{
		std::cout<<"Fail to load FD model"<<std::endl;
		return;
	}

	FaceAligner fa;
	if(!fa.loadModel("../models/fa.bin"))
	{
		std::cout<<"Fail to load FA model"<<std::endl;
		return;
	}

	cv::Mat color, gray;
	color = cv::imread(image_path);
	if(color.empty())
	{
		std::cout<<"Fail to open image "<<image_path<<std::endl;
		return;
	}

	cv::cvtColor(color, gray, CV_BGR2GRAY);

	std::vector<cv::Rect> faces;
	fd.detectMultiScale(gray, faces, 1.2, 3, 0, cv::Size(64, 64));
	if (faces.size() > 0)
	{
		for(const auto &face : faces)
		{
			cv::Mat shape = fa.predict(gray, face);
			if (!shape.empty())
			{
				for (int i = 0; i < TOTAL_PTS; ++i)
					cv::circle(color, cv::Point(shape.at<float>(i), shape.at<float>(i + TOTAL_PTS)), 3, cv::Scalar(0, 255, 0), -1);
			}
		}
	}

	cv::imwrite("out.jpg", color);

	cv::imshow("show", color);
	cv::waitKey(0);
}

void demo(const std::string &video_path = std::string())
{
	cv::CascadeClassifier fd;
	if(!fd.load("../models/haarcascade_frontalface_alt.xml"))
	{
		std::cout<<"Fail to load FD model"<<std::endl;
		return;
	}

	FaceAligner fa;
	if(!fa.loadModel("../models/fa.bin"))
	{
		std::cout<<"Fail to load FA model"<<std::endl;
		return;
	}

	cv::VideoCapture cap;
	if (video_path.empty())
	{
		cap.open(0);
	}	
	else
		cap.open(video_path);
	if(!cap.isOpened())
	{
		if(video_path.empty())
			std::cout << "Fail to open camera 0" << std::endl;
		else
			std::cout<<"Fail to open video file "<< video_path << std::endl;
		return;
	}

	bool isWait = false;
	cv::Mat color, gray;
	while (cap.read(color))
	{
		if(color.empty())
			break;

        cv::resize(color, color, cv::Size(1280, 720));
		cv::cvtColor(color, gray, CV_BGR2GRAY);

		std::vector<cv::Rect> faces;
		fd.detectMultiScale(gray, faces, 1.2, 3, 0, cv::Size(64, 64));
		if (faces.size() > 0)
		{
			double t = cv::getTickCount();
			cv::Mat shape = fa.predict(gray, faces[0]);
			std::cout << "FA: " << (cv::getTickCount() - t) * 1000 / cv::getTickFrequency() << " ms" << std::endl;

			if (!shape.empty())
			{
				for (int i = 0; i < TOTAL_PTS; ++i)
					cv::circle(color, cv::Point(shape.at<float>(i), shape.at<float>(i + TOTAL_PTS)), 3, cv::Scalar(0, 255, 0), -1);
			}
		}

		cv::imshow("show", color);
		auto key = cv::waitKey(!isWait);
		switch (key)
		{
		case 13:
			isWait = !isWait;
			break;

		case 27:
			return;
		}
	}
}

void usage(int argc, char **argv)
{
	std::cout << argv[0] << " MODE PATH" << std::endl;
	std::cout << "MODE: [train] for training, [test] for evaluating image, [demo] for evaluating video" << std::endl;
}

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		usage(argc, argv);
		return -1;
	}

	const std::string cmd(argv[1]);
	
	if (cmd != "demo" && argc < 3)
	{
		usage(argc, argv);
		return -1;
	}

	if (cmd == "train")
		train(argv[2]);
	else if (cmd == "test")
		test(argv[2]);
	else if (cmd == "demo")
	{
		if (argc < 3)
			demo();
		else
			demo(argv[2]);
	}
	else
	{
		usage(argc, argv);
		return -1;
	}

	return 0;
}
