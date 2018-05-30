#include "evo.hpp"

#include <vector>
#include <iostream>

namespace openevo {

EVO::EVO(void) :
		detector(new cv::GridAdaptedFeatureDetector(
				new cv::FastFeatureDetector(1),
				350, // TODO Replace hardcoded parameters // Note: keeps same number of kps per grid cell
				6,
				8))
{}

void EVO::updateImageDepth(const cv::Mat &image, const cv::Mat &depth) {
	// Detect keypoints
	std::vector<cv::KeyPoint> kp;
	this->detector->detect(image, kp); // TODO add depth available mask

	// DEBUG output
	cv::Mat debug;
	cv::drawKeypoints(image, kp, debug);
	cv::imshow("keypoints", debug);
	std::cout << "Keypts: " << kp.size() << std::endl;
}

EVO::~EVO(void) { }

}
