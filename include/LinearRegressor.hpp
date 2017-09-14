
/**
 * @author: laughteroverflow
 * @date: Sept. 14 2017
 */
 
#ifndef _LINEAR_REGRESSOR_HPP
#define _LINEAR_REGRESSOR_HPP

#include <opencv2/opencv.hpp>

class LinearRegressor
{
public:
    void		learn(cv::Mat &data, cv::Mat &labels, bool isPCA = false);
	cv::Mat		predict(cv::Mat values) const;

public:
    cv::Mat     weights_;
};

#endif
