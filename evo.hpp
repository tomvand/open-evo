#include <opencv2/opencv.hpp>

#include <vector>

namespace openevo {

class EVO {
public:
	EVO(void);
	virtual ~EVO(void);

	void updateIMU(void);

	void updateImageDepth(
			const cv::Mat &image,
			const cv::Mat &depth,
			cv::InputArray intrinsic);
//	void updateImagePair(void); // TODO
//	void updateImageDisparity(void); // TODO

private:
	int max_features;
	double near_clip;

	void updateKeyframe(
			const cv::Mat &image,
			const cv::Mat &depth,
			cv::InputArray intrinsic);

	cv::Ptr<cv::FeatureDetector> detector;

	std::vector<cv::Point3f> keyframe;

	std::vector<cv::Point2f> tracked_pts;
	cv::Mat prev_img;
};

}
