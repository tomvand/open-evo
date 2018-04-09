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

	double keyframe_thres;
	int keyframe_init_num_features;

	void updateKeyframe(
			const cv::Mat &image,
			const cv::Mat &depth,
			cv::InputArray intrinsic);

	cv::Ptr<cv::FeatureDetector> detector;

	std::vector<cv::Point3f> keyframe; // Keypoints in keyframe frame
	cv::Mat key_t, key_r; // Pose of keyframe in world, i.e. p_w = [R(key_r)|key_t] * p_k

	cv::Mat cam_t, cam_r; // Pose of camera in keyframe, i.e. p_k = [R(cam_r)|cam_t] * p_c

	std::vector<cv::Point2f> tracked_pts;
	cv::Mat prev_img;
};

}
