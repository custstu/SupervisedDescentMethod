
/**
 * @author: laughteroverflow
 * @date: Sept. 14 2017
 */
 
#include "FaceAligner.hpp"

FaceAligner::FaceAligner()
{
	e_.seed((unsigned int)time(nullptr));
	d_ = std::uniform_int_distribution<int>(-10, 10);
}

FaceAligner::~FaceAligner()
{
}

bool FaceAligner::train(const std::string &list_path,
						const int num_levels,
						const std::vector<std::vector<int>> &key_points,
						const std::vector<HOGParam> &hog_params)
{
	//Check parameters
	if (key_points.size() < num_levels || hog_params.size() < num_levels)
	{
		std::cout << "Bad parameters" << std::endl;
		return false;
	}

	//Set parameters
	num_levels_ = num_levels;
	key_points_ = key_points;
	hog_params_ = hog_params;

	//Load training samples
	std::vector<Sample> samples;
	if (loadSamples(list_path, samples) <= 0)
	{
		std::cout << "Fail to load training samples, abort!" << std::endl;
		return false;
	}

	//Put ground truth into cv::Mat, one sample per row
	cv::Mat shape_gt(int(samples.size()), TOTAL_PTS * 2, CV_32FC1);
	for (int sample_no = 0; sample_no < samples.size(); ++sample_no)
	{
		for (int pt_no = 0; pt_no < TOTAL_PTS; ++pt_no)
		{
			shape_gt.at<float>(sample_no, pt_no) = samples[sample_no].pts_[pt_no].x;
			shape_gt.at<float>(sample_no, pt_no + TOTAL_PTS) = samples[sample_no].pts_[pt_no].y;
		}
	}

	//Calculate mean
	mean_ = cv::Mat(1, TOTAL_PTS * 2, CV_32FC1, cv::Scalar(0.f));
	for (int sample_no = 0; sample_no < samples.size(); ++sample_no)
	{
		cv::Mat shape_scaled;
		scale(shape_gt.row(sample_no), samples[sample_no].face_, shape_scaled);
		mean_ += shape_scaled;
	}
	mean_ /= float(samples.size());

	//Init current shape as mean
	cv::Mat shape_cur;
	for (int sample_no = 0; sample_no < samples.size(); ++sample_no)
	{
		cv::Mat shape;
		unscale(mean_, samples[sample_no].face_, shape);
		shape_cur.push_back(shape);
	}

	//Calculate init error
	std::cout << "Normalized error: " << calcError(shape_cur, shape_gt) << std::endl << std::endl;

	//Start to train
	regressors_.resize(num_levels);
	for (int level_no = 0; level_no < num_levels; ++level_no)
	{
		std::cout << "Training level No." << level_no << "..." << std::endl;

		cv::Mat eye_distance = calcEyeDistance(shape_cur);
		
		//Extract HOG feature, one sample per row
		cv::Mat hog_features;
		for (int sample_no = 0; sample_no < samples.size(); ++sample_no)
		{
			cv::Mat hog_feature = HOGDescriptor::extractHOGFeatures(samples[sample_no].image_, 
																	shape_cur.row(sample_no), 
																	key_points_[level_no], 
																	hog_params_[level_no], 
																	eye_distance.at<float>(sample_no));
			
			hog_features.push_back(hog_feature);
		}

		//Learn model
		cv::Mat diff = shape_gt - shape_cur;
		for (int sample_no = 0; sample_no < samples.size(); ++sample_no)
			diff.row(sample_no) = diff.row(sample_no) / eye_distance.at<float>(sample_no);
		regressors_[level_no].learn(hog_features, diff);
		cv::Mat update_step = regressors_[level_no].predict(hog_features);
		for (int sample_no = 0; sample_no < samples.size(); ++sample_no)
			update_step.row(sample_no) = update_step.row(sample_no) * eye_distance.at<float>(sample_no);
		shape_cur = shape_cur + update_step;

		std::cout << "Normalized error: " << calcError(shape_cur, shape_gt) << std::endl << std::endl;
	}

	std::cout << "Training completed" << std::endl << std::endl;

	return true;
}

cv::Mat FaceAligner::predict(const cv::Mat &image, const cv::Rect &face)
{
	cv::Mat shape;
	unscale(mean_, face, shape);

	for (int level_no = 0; level_no < regressors_.size(); ++level_no)
	{
		float eye_distance = calcEyeDistance(shape).at<float>(0);

		cv::Mat mat_features = HOGDescriptor::extractHOGFeatures(image, shape, key_points_[level_no], hog_params_[level_no], eye_distance);
		cv::Mat update_step = regressors_[level_no].predict(mat_features);

		shape = shape + update_step*eye_distance;
	}

	return shape;
}

bool FaceAligner::saveModel(const std::string &model_path)
{
	int rows, cols;

	std::ofstream fs(model_path, std::ios::binary);
	if (!fs.is_open())
		return false;

	//Number of levels
	fs.write((char*)&num_levels_, sizeof(int));

	//Anchors
	for (int level_no = 0; level_no < num_levels_; ++level_no)
	{
		const auto &key_points = key_points_[level_no];
		int sz = int(key_points.size());

		fs.write((char*)&sz, sizeof(int));
		for (int i = 0; i < sz; ++i)
			fs.write((char*)&key_points[i], sizeof(int));
	}

	//HOG parameters
	for (int level_no = 0; level_no < num_levels_; ++level_no)
	{
		const auto &hog_param = hog_params_[level_no];

		int vlhog_variant = hog_param.vlhog_variant_;
		fs.write((char*)&vlhog_variant, sizeof(int));
		fs.write((char*)&hog_param.num_cells_, sizeof(int));
		fs.write((char*)&hog_param.cell_size_, sizeof(int));
		fs.write((char*)&hog_param.num_bins_, sizeof(int));
		fs.write((char*)&hog_param.relative_patch_size_, sizeof(float));
	}

	//Regressors
	for (int level_no = 0; level_no < num_levels_; ++level_no)
	{
		const auto &regressor = regressors_[level_no];

		//weights
		rows = regressor.weights_.rows;
		cols = regressor.weights_.cols;
		fs.write((char*)&rows, sizeof(int));
		fs.write((char*)&cols, sizeof(int));
		fs.write((char*)regressor.weights_.data, sizeof(float)*rows*cols);
	}

	//mean face
	rows = mean_.rows;
	cols = mean_.cols;
	fs.write((char*)&rows, sizeof(int));
	fs.write((char*)&cols, sizeof(int));
	fs.write((char*)mean_.data, sizeof(float)*rows*cols);

	return true;
}

bool FaceAligner::loadModel(const std::string &model_path)
{
	int rows, cols, n;

	std::ifstream fs(model_path, std::ios::binary);
	if (!fs.is_open())
		return false;

	//Number of levels
	fs.read((char*)&num_levels_, sizeof(int));

	//Anchors
	key_points_.resize(num_levels_);
	for (int level_no = 0; level_no < num_levels_; ++level_no)
	{
		auto &key_points = key_points_[level_no];

		int sz;
		fs.read((char*)&sz, sizeof(int));

		for (int i = 0; i < sz; ++i)
		{
			fs.read((char*)&n, sizeof(int));
			key_points.push_back(n);
		}
	}

	//HOG parameters
	hog_params_.resize(num_levels_);
	for (int level_no = 0; level_no < num_levels_; ++level_no)
	{
		auto &hog_param = hog_params_[level_no];

		fs.read((char*)&n, sizeof(int));
		hog_param.vlhog_variant_ = VlHogVariant(n);
		fs.read((char*)&hog_param.num_cells_, sizeof(int));
		fs.read((char*)&hog_param.cell_size_, sizeof(int));
		fs.read((char*)&hog_param.num_bins_, sizeof(int));
		fs.read((char*)&hog_param.relative_patch_size_, sizeof(float));
	}

	//Regressors
	regressors_.resize(num_levels_);
	for (int level_no = 0; level_no < num_levels_; ++level_no)
	{
		auto &regressor = regressors_[level_no];

		//weights
		fs.read((char*)&rows, sizeof(int));
		fs.read((char*)&cols, sizeof(int));
		regressor.weights_.create(rows, cols, CV_32FC1);
		fs.read((char*)regressor.weights_.data, sizeof(float)*rows*cols);
	}

	//mean face
	fs.read((char*)&rows, sizeof(int));
	fs.read((char*)&cols, sizeof(int));
	mean_.create(rows, cols, CV_32FC1);
	fs.read((char*)mean_.data, sizeof(float)*rows*cols);
	
	return true;
}

int FaceAligner::loadSamples(const std::string &list_path, std::vector<Sample> &samples)
{
	std::cout << "Loading training samples ..." << std::endl;

	//Open list file
	std::ifstream fs(list_path);
	if (!fs.is_open())
	{
		std::cout << "Fail to open list file " << list_path << std::endl;
		return -1;
	}

	//Read paths
	std::string line;
	std::vector<std::string> paths;
	while (std::getline(fs, line))
	{
		if (!line.empty())
			paths.push_back(line);
	}
	fs.close();

	if (paths.size() == 0)
	{
		std::cout << "No sample available" << std::endl;
		return -2;
	}

	//Prepare face detector
	cv::CascadeClassifier fd;
	if (!fd.load("../models/haarcascade_frontalface_alt.xml"))
	{
		std::cout << "Fail to load face detector model" << std::endl;
		return -3;
	}

	//Load samples
	samples.clear();
	samples.reserve(paths.size());
	for (const auto &path : paths)
	{
		Sample sample;

		//Load image
		sample.image_ = cv::imread(path, cv::IMREAD_GRAYSCALE);
		if (sample.image_.empty())
		{
			std::cout << "Fail to load image " << path << std::endl;
			continue;
		}

		//Load facial points
		std::string annotation_path = path.substr(0, path.find_last_of('.')) + ".pts";
		fs.open(annotation_path);
		if (!fs.is_open())
		{
			std::cout << "Fail to open annotation file " << annotation_path << std::endl;
			continue;
		}
		std::string line;
		std::getline(fs, line);
		std::getline(fs, line);
		std::getline(fs, line);
		for (int i = 0; i < TOTAL_PTS; ++i)
			fs >> sample.pts_[i].x >> sample.pts_[i].y;
		if (!fs)
		{
			std::cout << "Bad annotation file " << annotation_path << std::endl;
			continue;
		}
		fs.close();

		//Detect face
		std::vector<cv::Rect> faces;
		fd.detectMultiScale(sample.image_, faces, 1.2, 3, 0, cv::Size(64, 64));
		if (faces.empty())
		{
			std::cout << "No face detected in image " << path << std::endl;
			continue;
		}

		//Check face
		cv::Rect face;
		auto it = faces.begin();
		for (; it != faces.end(); ++it)
		{
			if (it->contains(sample.pts_[EYE_LEFT_OUTER]) && it->contains(sample.pts_[EYE_RIGHT_OUTER]) && it->contains(sample.pts_[MOUTH_BOTTOM]))
			{
				face = *it;
				break;
			}
		}
		if (it == faces.end())
		{
			std::cout << "No valid face in image " << path << std::endl;
			continue;
		}

		//Emplace sample
		for (int i = 0; i < 1; ++i)
		{
			sample.face_ = perturb(face);
			samples.emplace_back(sample);
		}
	}

	std::cout << "Loaded " << samples.size() << " samples from " << paths.size() << " candidates" << std::endl << std::endl;
	return int(samples.size());
}

cv::Rect FaceAligner::perturb(const cv::Rect &face)
{
	float translation_x = d_(e_)*0.01f;
	float translation_y = d_(e_)*0.01f;
	float scaling = 1.0f + d_(e_)*0.015f;

	auto tx_pixel = translation_x * face.width;
	auto ty_pixel = translation_y * face.height;

	auto perturbed_width = face.width * scaling;
	auto perturbed_height = face.height * scaling;

	return cv::Rect(int(face.x + (face.width - perturbed_width) / 2.0f + tx_pixel + 0.5f), 
					int(face.y + (face.height - perturbed_height) / 2.0f + ty_pixel + 0.5f), 
					int(perturbed_width + 0.5f), 
					int(perturbed_height + 0.5f));
}

void FaceAligner::scale(const cv::Mat &shape, const cv::Rect &face, cv::Mat &shape_scaled)
{
	if(shape_scaled.empty())
		shape_scaled.create(shape.rows, shape.cols, CV_32FC1);
	shape_scaled.colRange(0, TOTAL_PTS) = (shape.colRange(0, TOTAL_PTS) - face.x) / face.width - 0.5f;
	shape_scaled.colRange(TOTAL_PTS, TOTAL_PTS * 2) = (shape.colRange(TOTAL_PTS, TOTAL_PTS * 2) - face.y) / face.height - 0.5f;
}

void FaceAligner::unscale(const cv::Mat &shape_scaled, const cv::Rect face, cv::Mat &shape)
{
	if(shape.empty())
		shape.create(shape_scaled.rows, shape_scaled.cols, CV_32FC1);
	shape.colRange(0, TOTAL_PTS) = (shape_scaled.colRange(0, TOTAL_PTS) + 0.5f)*face.width + face.x;
	shape.colRange(TOTAL_PTS, TOTAL_PTS * 2) = (shape_scaled.colRange(TOTAL_PTS, TOTAL_PTS * 2) + 0.5f)*face.height + face.y;
}

cv::Mat FaceAligner::calcEyeDistance(const cv::Mat &shape)
{
	cv::Mat eye_distance(shape.rows, 1, CV_32FC1);
	
	cv::Mat lx = (shape.col(EYE_LEFT_INNER) + shape.col(EYE_LEFT_OUTER))*0.5;
	cv::Mat ly = (shape.col(EYE_LEFT_INNER + TOTAL_PTS) + shape.col(EYE_LEFT_OUTER + TOTAL_PTS))*0.5;
	cv::Mat rx = (shape.col(EYE_RIGHT_INNER) + shape.col(EYE_RIGHT_OUTER))*0.5;
	cv::Mat ry = (shape.col(EYE_RIGHT_INNER + TOTAL_PTS) + shape.col(EYE_RIGHT_OUTER + TOTAL_PTS))*0.5;
	
	cv::Mat dx = rx - lx;
	cv::Mat dy = ry - ly;
	
	eye_distance = dx.mul(dx) + dy.mul(dy);
	for (int i = 0; i < eye_distance.rows; ++i)
		eye_distance.at<float>(i) = sqrt(eye_distance.at<float>(i));
	
	return eye_distance;
}

float FaceAligner::calcError(const cv::Mat &shape_cur, const cv::Mat &shape_gt)
{
	//Calculate distance between eyes in current shape
	cv::Mat eye_distance = calcEyeDistance(shape_cur);
	
	cv::Mat dx = shape_gt.colRange(0, TOTAL_PTS) - shape_cur.colRange(0, TOTAL_PTS);
	cv::Mat dy = shape_gt.colRange(TOTAL_PTS, TOTAL_PTS * 2) - shape_cur.colRange(TOTAL_PTS, TOTAL_PTS * 2);
		
	//dx*dx + dy*dy
	cv::Mat diff = dx.mul(dx) + dy.mul(dy);
	
	float error = 0.f;
	for (int sample_no = 0; sample_no < shape_cur.rows; ++sample_no)
	{
		for (int pt_no = 0; pt_no < TOTAL_PTS; ++pt_no)
			error += sqrt(diff.at<float>(sample_no, pt_no)) / eye_distance.at<float>(sample_no);
	}
	
	return error / shape_cur.rows / TOTAL_PTS;
}
