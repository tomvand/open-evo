#include "evo.hpp"

#include <vector>

#ifdef DEBUG
#include <iostream>

#include <cstdio>
#include <unistd.h>
void profile_printf(const char *s) {
	printf("PROFILER: %s", s);
}
#define LIB_PROFILER_IMPLEMENTATION
#define LIB_PROFILER_PRINTF profile_printf
#endif // DEBUG

#include "libProfiler/libProfiler.h" // Included outside DEBUG to provide empty macros

namespace openevo {

EVO::EVO(void) :
		detector(new cv::GridAdaptedFeatureDetector(
				new cv::FastFeatureDetector(1),
				350, // TODO Replace hardcoded parameters // Note: keeps same number of kps per grid cell
				6,
				8)),
		initialized(false)
{
	PROFILER_ENABLE;
}

EVO::~EVO(void) { }

void EVO::updateImageDepth(const cv::Mat &image, const cv::Mat &depth) {
	PROFILER_START(updateImageDepth);

	PROFILER_START(initialize);
	if(!this->initialized) {
		image.copyTo(this->prev_img);

		std::vector<cv::KeyPoint> kp;
		this->detector->detect(this->prev_img, kp); // TODO add depth available mask
		cv::KeyPoint::convert(kp, this->prev_pts);

		this->initialized = true;
	}
	PROFILER_END();

	PROFILER_START(track);
	// Track detected features
	std::vector<cv::Point2f> pts;
	std::vector<bool> is_tracked;
	cv::Mat err;
	cv::calcOpticalFlowPyrLK(this->prev_img, image, this->prev_pts, pts, is_tracked, err,
			cv::Size(15,15)); // See Kelly et al., 2008 for window size.
	PROFILER_END();

	PROFILER_START(copy);
	image.copyTo(this->prev_img);
	this->prev_pts = pts;
	PROFILER_END();

	PROFILER_START(fcheck);
	// Remove outliers using fundamental matrix
	std::vector<bool> is_inlier;
	cv::findFundamentalMat(this->prev_pts, pts, is_inlier, CV_FM_LMEDS);
	PROFILER_END();

	PROFILER_END(); // updateImageDepth

#ifdef DEBUG
	cv::Mat debug;
	image.copyTo(debug);
	for(int i = 0; i < pts.size(); ++i) {
		if(is_tracked[i] && is_inlier[i]) {
			cv::circle(debug, pts[i], 2, cv::Scalar(0, 255, 0));
		}
	}
	cv::imshow("keypoints", debug);
	std::cout << "Keypts: " << pts.size() << std::endl;
#endif

	LogProfiler();
}

} // namespace openevo
