
/**
 * @author: laughteroverflow
 * @date: Sept. 14 2017
 */

#ifndef _FACE_ALIGNER_HPP
#define _FACE_ALIGNER_HPP

#include <opencv2/opencv.hpp>
#include <random>
#include "HOGDescriptor.hpp"
#include "LinearRegressor.hpp"

/**
 * @brief: total number of points to be estimated
 */ 
#define TOTAL_PTS		(68)

/**
 * @brief: index of IBUG dataset
 */ 
enum POINT_INDEX
{
	CHEEK_RIGHT_OUTER = 4 - 1,
	CHEEK_RIGHT_INNER = 7 - 1,
	CHIN = 9 - 1,
	CHEEK_LEFT_INNER = 11 - 1,
	CHEEK_LEFT_OUTER = 14 - 1,
	EYEBROW_RIGHT_OUTER = 18 - 1,
	EYEBROW_RIGHT_CENTER = 20 - 1,
	EYEBROW_RIGHT_INNER = 22 - 1,
	EYEBROW_LEFT_INNER = 23 - 1,
	EYEBROW_LEFT_CENTER = 25 - 1,
	EYEBROW_LEFT_OUTER = 27 - 1,
	NOSE_TIP = 31 - 1,
	EYE_RIGHT_OUTER = 37 - 1,
	EYE_RIGHT_1 = 38 - 1,
	EYE_RIGHT_2 = 39 - 1,
	EYE_RIGHT_INNER = 40 - 1,
	EYE_RIGHT_3 = 41 - 1,
	EYE_RIGHT_4 = 42 - 1,
	EYE_LEFT_INNER = 43 - 1,
	EYE_LEFT_1 = 44 - 1,
	EYE_LEFT_2 = 45 - 1,
	EYE_LEFT_OUTER = 46 - 1,
	EYE_LEFT_3 = 47 - 1,
	EYE_LEFT_4 = 48 - 1,
	MOUTH_RIGHT = 49 - 1,
	MOUTH_TOP = 52 - 1,
	MOUTH_LEFT = 55 - 1,
	MOUTH_BOTTOM = 58 - 1
};

/**
 * @brief: training sample
 */ 
struct Sample
{
	cv::Mat			image_;
	cv::Point2f		pts_[TOTAL_PTS];
	cv::Rect		face_;
};

/**
 * @brief: interface of SDM algorithm
 */ 
class FaceAligner
{
public:
    FaceAligner();
    ~FaceAligner();

public:
	/**
	 * @brief: train SDM model
	 * @param: list_path [in] a list of training samples, each row contains a full path to image
	 * @param: num_levels [in] the number of regressors
	 * @param: key_points [in] features will be extracted on these points
	 * @param: hog_params [in] parameters for HOG calculation
	 * @return: return true if trainging success
 	 */ 
    bool train(const std::string &list_path, 
               const int num_levels, 
               const std::vector<std::vector<int>> &key_points, 
               const std::vector<HOGParam> &hog_params);

	/**
	 * @brief: predict landmarks
	 * @param: image [in] image to be evaluated
	 * @param: face [in] rectangle of face
	 * @return: landmarks
	 */
    cv::Mat predict(const cv::Mat &image, const cv::Rect &face);

	/**
	 * @brief: save trained model
	 * @param: model_path [in] path to model
	 * @return: return true if success
	 */
    bool saveModel(const std::string &model_path);

	/**
	 * @brief: load model from file
	 * @param: model_path [in] path to model
	 * @return: return true if success
	 */
    bool loadModel(const std::string &model_path);

private:
	/**
	 * @brief: load training samples
	 * @param: list_path [in] list of training samples
	 * @param: samples [out] loaded samples
	 * @return: return negative number if fail, or return the number of loaded samples
	 */
    int loadSamples(const std::string &list_path, std::vector<Sample> &samples);

	/**
	 * @brief: perturb face rectangle
	 * @param: face [in] face rectangle
	 * @return: perturbed face rectangle
	 */
    cv::Rect perturb(const cv::Rect &face);

	/**
	 * @brief: scale landmarks to [-1, 1]
	 * @param: shape [in] landmarks in pixel coordinates
	 * @param: face [in] face rectangle
	 * @param: shape_scaled [out] scaled landmarks
	 */
    void scale(const cv::Mat &shape, const cv::Rect &face, cv::Mat &shape_scaled);
	
	/**
	 * @brief: unscale landmarks from [-1, 1] to pixel coordinates
	 * @param: shape_scaled [in] scaled landmarks
	 * @param: face [in] face rectangle
	 * @param: shape [out] landmarks in pixel coordinates
	 */
    void unscale(const cv::Mat &shape_scaled, const cv::Rect face, cv::Mat &shape);

	/**
	 * @brief: calculate distance of eyes
	 * @param: shape [in] landmarks
	 * @return: distances
	 */
    cv::Mat calcEyeDistance(const cv::Mat &shape);

	/**
	 * @brief: calculate error
	 * @param: shape_cur [in] calculated shape
	 * @param: shape_gt [in] ground truth shape
	 * @return: error
	 */
    float calcError(const cv::Mat &shape_cur, const cv::Mat &shape_gt);

private:
	int									num_levels_;
	std::vector<std::vector<int>>		key_points_;
	std::vector<HOGParam>				hog_params_;
	std::vector<LinearRegressor>		regressors_;
	cv::Mat								mean_;

	/**
	 * @brief: for random generating
	 */
	std::default_random_engine			e_;
    std::uniform_int_distribution<int>	d_;
};

#endif
