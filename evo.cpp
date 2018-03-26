#include "evo.hpp"

#include <vector>
#include <iostream>

namespace openevo {

EVO::EVO(void) :
		detector(new cv::GridAdaptedFeatureDetector(
				new cv::FastFeatureDetector(1),
				350, // TODO Replace hardcoded parameters // Note: keeps same number of kps per grid cell
				6,
				8)),
		initialized(false)
{}

void EVO::updateImageDepth(const cv::Mat &image, const cv::Mat &depth) {
	if(!this->initialized) {
		image.copyTo(this->prev_img);

		std::vector<cv::KeyPoint> kp;
		this->detector->detect(this->prev_img, kp); // TODO add depth available mask
		cv::KeyPoint::convert(kp, this->prev_pts);

		this->initialized = true;
	}

	// Track detected features
	std::vector<cv::Point2f> pts;
	std::vector<bool> is_tracked;
	cv::Mat err;
	cv::calcOpticalFlowPyrLK(this->prev_img, image, this->prev_pts, pts, is_tracked, err,
			cv::Size(15,15)); // See Kelly et al., 2008 for window size.

	image.copyTo(this->prev_img);
	this->prev_pts = pts;

	// Remove outliers using fundamental matrix
	std::vector<bool> is_inlier;
	cv::findFundamentalMat(this->prev_pts, pts, is_inlier, CV_FM_LMEDS);

	// DEBUG output
	cv::Mat debug;
	image.copyTo(debug);
	for(int i = 0; i < pts.size(); ++i) {
		if(is_tracked[i] && is_inlier[i]) {
			cv::circle(debug, pts[i], 2, cv::Scalar(0, 255, 0));
		}
	}
	cv::imshow("keypoints", debug);
	std::cout << "Keypts: " << pts.size() << std::endl;
}

EVO::~EVO(void) { }

}
