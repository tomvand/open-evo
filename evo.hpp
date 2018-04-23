#include <opencv2/opencv.hpp>

#include <vector>

namespace openevo {

class EVO {
public:
	EVO(void);
	virtual ~EVO(void);

	bool getPose(cv::Mat &rvec, cv::Mat &tvec);
	bool getRates(cv::Mat &vel, cv::Mat &rates);

	void updateIMU(const cv::Mat &rates, double timestamp);
	void updateImageDepth(
			const cv::Mat &image,
			const cv::Mat &depth,
			cv::InputArray intrinsic,
			double timestamp);
//	void updateImagePair(void); // TODO
//	void updateImageDisparity(void); // TODO

	inline void setTargetKeypts(int num) { this->target_keypts = num; }
	inline void setMinKeypts(int num) { this->min_keypts = num; }
	inline void setKeyframeThres(double thres) { this->keyframe_thres = thres; }
	inline void setNearClip(double nc) { this->near_clip = nc; }

private:
	int target_keypts;
	int min_keypts;
	double keyframe_thres;
	double near_clip;


	void predictKeypoints(
			cv::InputArray intrinsic,
			std::vector<cv::Point2f> &predicted_pts,
			double dt);
	void trackKeypoints(
			const cv::Mat &image,
			const std::vector<cv::Point2f> &predicted_pts);
	void updateKeyframe(
			const cv::Mat &image,
			const cv::Mat &depth,
			cv::InputArray intrinsic);

	cv::Ptr<cv::FeatureDetector> detector;

	std::vector<cv::Point3f> keyframe; // Keypoints in keyframe frame
	int keyframe_initial_size; // Initial number of points in keyframe
	cv::Mat key_t, key_r; // Pose of keyframe in world, i.e. p_w = [R(key_r)|key_t] * p_k
	cv::Mat cam_t, cam_r; // Pose of camera in keyframe, i.e. p_k = [R(cam_r)|cam_t] * p_c

	std::vector<cv::Point2f> tracked_pts;
	cv::Mat prev_img;
	double prev_timestamp;

	cv::Mat imu_R; // Orientation matrix of IMU in previous camera frame, i.e. p_imu = imu_R * p_c
	cv::Mat imu_bias; // Estimated IMU angular velocity bias
	double imu_bias_gain; // Bias estimator gain [0..1]
	double imu_prev_timestamp;

	cv::Mat vel; // Translational velocity in current camera frame
	cv::Mat rates; // Angular velocity in current camera frame
	bool estimate_valid;
};

}
