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
				0, // Set in EVO::updateKeyframe
				6,
				8)),
		max_features(350)
{
	PROFILER_ENABLE;
}

EVO::~EVO(void) { }

void EVO::updateImageDepth(const cv::Mat &image, const cv::Mat &depth) {
	PROFILER_START(updateImageDepth);

	if(this->keyframe.size() > 0) {
		// Track features
		//	Remove lost features from lists incl keyframe
		// Compute pose
		// If not new keyframe
		//	Predict features
		//	return; // Skip keyframe update
	}

	if(this->keyframe.size() < 999) { // TODO check below threshold
		this->updateKeyframe(image, depth);
	}

	std::cout << "Keypoints: " << this->keyframe.size() << std::endl;


//	PROFILER_START(track);
//	// Track detected features
//	std::vector<cv::Point2f> pts;
//	std::vector<bool> is_tracked;
//	cv::Mat err;
//	cv::calcOpticalFlowPyrLK(this->prev_img, image, this->prev_pts, pts, is_tracked, err,
//			cv::Size(15,15)); // See Kelly et al., 2008 for window size.
//	PROFILER_END();
//
//	PROFILER_START(fcheck);
//	// Remove outliers using fundamental matrix
//	std::vector<bool> is_inlier;
//	cv::findFundamentalMat(this->prev_pts, pts, is_inlier, CV_FM_LMEDS);
//	PROFILER_END();
//
//	PROFILER_START(copy);
//	image.copyTo(this->prev_img);
//	this->prev_pts = pts;
//	PROFILER_END();

	PROFILER_END(); // updateImageDepth

#ifdef DEBUG
	cv::Mat debug;
	image.copyTo(debug);
	for(int i = 0; i < this->prev_pts.size(); ++i) {
		cv::circle(debug, this->prev_pts[i], 2, cv::Scalar(0, 255, 0));
	}
	cv::imshow("keypoints", debug);
#endif

	LogProfiler();
}

void EVO::updateKeyframe(const cv::Mat &image, const cv::Mat &depth) {
	PROFILER_START(updateKeyframe);

	// Remove remaining keypoints if no depth information is available anymore
	PROFILER_START(depth_mask);
	cv::Mat depth_mask(depth.rows, depth.cols, CV_8UC1);
	cv::Mat image_mask(image.rows, image.cols, CV_8UC1);
	depth_mask = (depth == depth); // Filter unknown (NaN)
	depth_mask &= (depth > 0.0);   // Filter too close (-Inf)
	depth_mask &= (depth < 1.0e6); // Filter too far (+Inf)
	cv::resize(depth_mask, image_mask, image_mask.size(), 0, 0, cv::INTER_NEAREST);
	PROFILER_END();
	PROFILER_START(filter_keyframe);
	int ratio = image.cols / depth.cols; // Assume integer ratio between image and depth map size
	std::vector<bool> depth_available(this->prev_pts.size());
	for(int i = 0; i < this->prev_pts.size(); ++i) {
		int depth_x = this->prev_pts[i].x / ratio;
		int depth_y = this->prev_pts[i].y / ratio;
		depth_available[i] =
				depth_x > 0 && depth_x < depth_mask.cols &&
				depth_y > 0 && depth_y < depth_mask.rows &&
				depth_mask.at<char>(depth_y, depth_x);
	}
	this->filterKeyframe(depth_available);
	PROFILER_END();

	// Detect new keypoints
	PROFILER_START(detect_keypoints);
	int num_new_kp = this->max_features - this->keyframe.size();
	if(num_new_kp <= 0) return;

	std::vector<cv::KeyPoint> new_kp;
	std::vector<cv::Point2f> new_pts;
	this->detector->set("maxTotalKeypoints", num_new_kp);
	this->detector->detect(image, new_kp, image_mask); // TODO add depth mask
	cv::KeyPoint::convert(new_kp, new_pts);
	PROFILER_END();

	// Store 3D coords in keyframe
	PROFILER_START(update_keyframe);
	this->prev_pts.insert(this->prev_pts.end(), new_pts.begin(), new_pts.end());
	this->keyframe.resize(this->prev_pts.size());
	for(int i = 0; i < this->prev_pts.size(); ++i) {
		this->keyframe[i].x = 0.0; // TODO, maybe reprojectto3d?
		this->keyframe[i].y = 0.0;
		this->keyframe[i].z = 0.0;
	}
	PROFILER_END();

	PROFILER_END(); // updateKeyframe
}

void EVO::filterKeyframe(std::vector<bool> keep) {
	PROFILER_START(filterKeyframe);

	assert(keep.size() == this->prev_pts.size());
	assert(keep.size() == this->keyframe.size());

	int ikeep = 0;
	for(int iread = 0; iread != keep.size();){
		if(keep[iread]) {
			this->prev_pts[ikeep] = this->prev_pts[iread];
			this->keyframe[ikeep] = this->keyframe[iread];
			++ikeep;
		}
		++iread;
	}
	this->prev_pts.resize(ikeep); // Note: does not reallocate vector
	this->keyframe.resize(ikeep);

#ifdef DEBUG
	int sum = 0;
	for(int i = 0; i < keep.size(); ++i) {
		if(keep[i]) ++sum;
	}
	assert(ikeep == sum);
#endif
	PROFILER_END(); // filterKeyframe
}

} // namespace openevo
