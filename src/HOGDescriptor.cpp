
/**
 * @author: laughteroverflow
 * @date: Sept. 14 2017
 */
 
#include "HOGDescriptor.hpp"

cv::Mat HOGDescriptor::extractHOGFeatures(const cv::Mat &image, 
                                          const cv::Mat &shape, 
                                          const std::vector<int> &key_points, 
                                          const HOGParam &hog_param, 
                                          const float eye_distance)
{
    cv::Mat features;
	const int patch_width_half = int(hog_param.relative_patch_size_ * eye_distance + 0.5f);
    const int num_landmarks = shape.cols/2;
    const int fixed_roi_size = hog_param.num_cells_ * hog_param.cell_size_;

    //Calculate HOG feature on each key point
    for(auto pt : key_points)
    {
        int x = int(shape.at<float>(pt) + 0.5f);
		int y = int(shape.at<float>(pt + num_landmarks) + 0.5f);

        cv::Mat patch;
        if (x - patch_width_half < 0 || y - patch_width_half < 0 ||
			x + patch_width_half >= image.cols || y + patch_width_half >= image.rows)
		{
			int borderLeft = (x - patch_width_half) < 0 ? std::abs(x - patch_width_half) : 0;
			int borderTop = (y - patch_width_half) < 0 ? std::abs(y - patch_width_half) : 0;
			int borderRight = (x + patch_width_half) >= image.cols ? std::abs(image.cols - (x + patch_width_half)) : 0;
			int borderBottom = (y + patch_width_half) >= image.rows ? std::abs(image.rows - (y + patch_width_half)) : 0;

			cv::Mat extendedImage = image.clone();
			cv::copyMakeBorder(extendedImage, extendedImage, borderTop, borderBottom, borderLeft, borderRight, cv::BORDER_CONSTANT, cv::Scalar(0));
			cv::Rect roi((x - patch_width_half) + borderLeft, (y - patch_width_half) + borderTop, patch_width_half * 2, patch_width_half * 2);
			patch = extendedImage(roi).clone();
		}
		else
		{
			cv::Rect roi(x - patch_width_half, y - patch_width_half, patch_width_half * 2, patch_width_half * 2);
			patch = image(roi).clone();
        }

        //Resize patch to fixed size to get fixed size length of feature
		cv::resize(patch, patch, cv::Size(fixed_roi_size, fixed_roi_size));
        
        //Extract feature
        cv::Mat feature = extractHOGFeature(patch, hog_param);
        features.push_back(feature);
    }

    //Reshape as a row array
    features = features.reshape(1, 1);
    
    //Add a bias row
	cv::Mat bias = cv::Mat::ones(1, 1, CV_32FC1);
	cv::hconcat(features, bias, features);
	return features;
}

cv::Mat HOGDescriptor::extractHOGFeature(const cv::Mat &image, const HOGParam &hog_param)
{
    cv::Mat image_flt;
    image.convertTo(image_flt, CV_32FC1);

    VlHog* hog = vl_hog_new(hog_param.vlhog_variant_, hog_param.num_bins_, false);
    vl_hog_put_image(hog, (float*)image_flt.data, image_flt.cols, image_flt.rows, 1, hog_param.cell_size_);

    auto hog_width = vl_hog_get_width(hog);
    auto hog_height = vl_hog_get_height(hog);
    auto hog_dims = vl_hog_get_dimension(hog);

    cv::Mat feature(1, int(hog_width*hog_height*hog_dims), CV_32FC1);
    vl_hog_extract(hog, feature.ptr<float>(0));
    vl_hog_delete(hog);

    return feature;
}
