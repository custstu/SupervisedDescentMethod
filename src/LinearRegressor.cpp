
/**
 * @author: laughteroverflow
 * @date: Sept. 14 2017
 */
 
#include "LinearRegressor.hpp"

void LinearRegressor::learn(cv::Mat &data, cv::Mat &labels, bool isPCA)
{
    cv::Mat A = data;
	cv::Mat AT = A.t();
	cv::Mat ATA = A.t()*A;
	float lambda = 1.50f * static_cast<float>(cv::norm(ATA)) / static_cast<float>(A.rows);
	cv::Mat regulariser = cv::Mat::eye(ATA.size(), ATA.type())*lambda;
	regulariser.at<float>(regulariser.rows - 1, regulariser.cols - 1) = 0.0f;
	weights_ = (ATA + regulariser).inv(cv::DECOMP_LU)*AT*labels;
}

cv::Mat LinearRegressor::predict(cv::Mat values) const
{
	return  values*weights_;
}
