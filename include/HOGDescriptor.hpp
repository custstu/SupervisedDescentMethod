
/**
 * @author: laughteroverflow
 * @date: Sept. 14 2017
 */
 
#ifndef _HOG_DESCRIPTOR_HPP
#define _HOG_DESCRIPTOR_HPP

#include <opencv2/opencv.hpp>
extern "C"
{
    #include "hog.h"
}

struct HOGParam
{
    VlHogVariant    vlhog_variant_;
    int             cell_size_;
    int             num_cells_;
    int             num_bins_;
    float			relative_patch_size_;

    HOGParam() = default;
    HOGParam(VlHogVariant vlhog_variant, int cell_size, int num_cells, int num_bins, float relative_patch_size) : 
             vlhog_variant_(vlhog_variant), cell_size_(cell_size), num_cells_(num_cells), num_bins_(num_bins), relative_patch_size_(relative_patch_size)
    {}
};

class HOGDescriptor
{
public:
    static cv::Mat extractHOGFeatures(const cv::Mat &image, 
                                      const cv::Mat &shape, 
                                      const std::vector<int> &key_points, 
                                      const HOGParam &hog_param, 
                                      const float eye_distance);

private:
    static cv::Mat extractHOGFeature(const cv::Mat &image, const HOGParam &hog_param);
};

#endif
