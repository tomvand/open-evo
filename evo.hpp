#include <opencv2/opencv.hpp>

#include <vector>

namespace openevo {

class EVO {
public:
	EVO(void);
	virtual ~EVO(void);

	void getPose(cv::Mat &rvec, cv::Mat &tvec);

	void updateIMU(void);
	void updateImageDepth(
			const cv::Mat &image,
			const cv::Mat &depth,
			cv::InputArray intrinsic,
			double timestamp);
//	void updateImagePair(void); // TODO
//	void updateImageDisparity(void); // TODO

private:
	int target_keypts;
	double near_clip;

	double keyframe_thres;
	double valid_thres;
	int keypts_at_keyframe_init;

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
	double prev_timestamp;

	cv::Mat vel; // Translational velocity in current camera frame
	cv::Mat rates; // Angular velocity in current camera frame
};

}
