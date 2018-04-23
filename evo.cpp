#include "evo.hpp"

#include <vector>


#ifdef DEBUG
#include <sstream>
#include <iostream>
#include <iomanip>
#endif // DEBUG

// Profiler. Placed outside DEBUG to provide empty macro's
#include <unistd.h>
#include <cstdio>
void profile_printf(const char *s) {
	printf("PROFILER: %s", s);
}
#define LIB_PROFILER_IMPLEMENTATION
#define LIB_PROFILER_PRINTF profile_printf
#include "libProfiler/libProfiler.h"


namespace {

template<class T1, class T2>
void filter_vector(std::vector<T1> &vec, const std::vector<T2> &keep) {
	assert(keep.size() == vec.size());
	PROFILER_START(filter_vector);
	int ikeep = 0;
	for(int iread = 0; iread != keep.size();){
		if(keep[iread]) {
			vec[ikeep] = vec[iread];
			++ikeep;
		}
		++iread;
	}
	vec.resize(ikeep); // Note: does not reallocate vector

#ifdef DEBUG
	int sum = 0;
	for(int i = 0; i < keep.size(); ++i) {
		if(keep[i]) ++sum;
	}
	assert(ikeep == sum);
#endif
	PROFILER_END(); // filter_vector
}


template<class T1, class T2>
void select_vector(std::vector<T1> &vec, const std::vector<T2> &select) {
#ifdef DEBUG
	for(int i = 1; i < select.size(); ++i) {
		assert(select[i] > select[i-1]); // select should be sorted, no duplicates
	}
#endif
	int ivec = 0;
	for(int i = 0; i < select.size(); ++i) {
		vec[ivec++] = vec[select[i]]; // Note: select[i] >= ivec when select is sorted
	}
	vec.resize(ivec);
	assert(vec.size() == select.size());
}


void invert_pose(const cv::Mat &rvec_in, const cv::Mat &tvec_in,
		cv::Mat &rvec_out, cv::Mat &tvec_out) {
	cv::Mat R;
	cv::Rodrigues(rvec_in, R);
	cv::transpose(R, R);
	cv::Rodrigues(R, rvec_out);
	tvec_out = -R * tvec_in;
}


} // namespace


namespace openevo {


EVO::EVO(void) :
		detector(new cv::GridAdaptedFeatureDetector(
				new cv::FastFeatureDetector(1),
				0, // Set in EVO::updateKeyframe
				6,
				8)),
		target_keypts(350),
		min_keypts(50),
		near_clip(1.5),
		keyframe_thres(0.8),

		keyframe_initial_size(9999),
		key_r(cv::Mat::zeros(3, 1, CV_64F)),
		key_t(cv::Mat::zeros(3, 1, CV_64F)),
		cam_r(cv::Mat::zeros(3, 1, CV_64F)),
		cam_t(cv::Mat::zeros(3, 1, CV_64F)),
		prev_timestamp(0.0),
		imu_R(cv::Mat::eye(3, 3, CV_64F)),
		imu_bias(cv::Mat::zeros(3, 1, CV_64F)),
		imu_bias_gain(0.0), // XXX
		imu_prev_timestamp(0.0),
		vel(cv::Mat::zeros(3, 1, CV_64F)),
		rates(cv::Mat::zeros(3, 1, CV_64F)),
		estimate_valid(false)
{
	this->key_r.at<double>(0, 0) = -M_PI / 2.0;
	PROFILER_ENABLE;
}


EVO::~EVO(void) { }


bool EVO::getPose(cv::Mat &rvec, cv::Mat &tvec) {
	cv::composeRT(this->cam_r, this->cam_t, this->key_r, this->key_t,
			rvec, tvec);
	return this->estimate_valid;
}


bool EVO::getRates(cv::Mat &vel, cv::Mat &rates) {
	this->vel.copyTo(vel);
	this->rates.copyTo(rates);
	return this->estimate_valid;
}


void EVO::updateIMU(const cv::Mat &rates, double timestamp) {
	// Sanity checks
	if(timestamp < this->prev_timestamp) return; // Ignore IMU data before latest image
	if(this->imu_prev_timestamp == 0.0) this->imu_prev_timestamp = timestamp; // Late initialization
	// Update orientation prediction
	double dt = timestamp - this->imu_prev_timestamp;
	cv::Mat R;
	cv::Rodrigues((rates - this->imu_bias) * dt, R); // R_imu-1,imu
	cv::transpose(R,  R); // R_imu,imu-1
	this->imu_R = R * this->imu_R; // R_imu,imu-N
	this->imu_prev_timestamp = timestamp;
	// Update IMU bias
	// TODO find better method!
	cv::Mat error = rates - this->rates;
	this->imu_bias += this->imu_bias_gain * (error - this->imu_bias);
}


void EVO::updateImageDepth(
		const cv::Mat &image,
		const cv::Mat &depth,
		cv::InputArray intrinsic,
		double timestamp) {
	PROFILER_START(updateImageDepth);
	if(this->keyframe.size() > 0) {
		PROFILER_START(updatePose);

		// Calculate timestep
		if(this->prev_timestamp == 0.0) this->prev_timestamp = timestamp; // Late initialization
		double dt = timestamp - this->prev_timestamp;

		// Track keypoints
		std::vector<cv::Point2f> predicted_pts;
		this->predictKeypoints(intrinsic, predicted_pts, dt);
		this->trackKeypoints(image, predicted_pts);

		// Filter outliers
//		PROFILER_START(fcheck);
//		std::vector<uchar> is_inlier;
//		cv::findFundamentalMat(prev_pts, this->tracked_pts, is_inlier, CV_FM_LMEDS);
//		filter_vector(prev_pts, is_inlier); // Necessary?
//		filter_vector(this->tracked_pts, is_inlier);
//		filter_vector(this->keyframe, is_inlier);
//		PROFILER_END();

		// Compute pose and rates
		PROFILER_START(compute_pose);
		// Coarse initial estimate
		cv::Mat rvec; // Note: rvec, tvec are keyframe pose in camera frame.
		cv::Mat tvec;
		this->estimate_valid = this->tracked_pts.size() > this->min_keypts;
		if(this->estimate_valid) {
			std::vector<int> inlier_idxs;
			// Coarse estimation using P3P, find inliers
			cv::solvePnPRansac(this->keyframe, this->tracked_pts, intrinsic,
					std::vector<double>(), rvec, tvec, false, 100, 8.0, 100, inlier_idxs, CV_P3P);
			// Remove outliers
			select_vector(this->keyframe, inlier_idxs);
			select_vector(this->tracked_pts, inlier_idxs);
			// Verify enough inliers remain
			this->estimate_valid = this->tracked_pts.size() > this->min_keypts;
		}
		// Fine pose estimation
		if(this->estimate_valid) {
			cv::solvePnP(this->keyframe, this->tracked_pts, intrinsic,
							std::vector<double>(), rvec, tvec, true, CV_ITERATIVE);
			// Verify enough inliers remain
			this->estimate_valid = this->tracked_pts.size() > this->min_keypts;
		}

		// Update pose of camera in keyframe
		if(this->estimate_valid && dt > 0.0) {
			cv::Mat R, prev_cam_t, prev_cam_r;
			// Keep track of previous cam pose
			this->cam_t.copyTo(prev_cam_t);
			this->cam_r.copyTo(prev_cam_r);
			// Update current cam pose
			invert_pose(rvec, tvec, this->cam_r, this->cam_t);
			// Express previous cam frame in current cam frame
			// H_c,c-1 = H_c,k * H_k,c-1
			cv::Mat cam_r_inv, cam_t_inv, step_t, step_r;
			invert_pose(this->cam_r, this->cam_t, cam_r_inv, cam_t_inv);
			cv::composeRT(prev_cam_r, prev_cam_t, cam_r_inv, cam_t_inv,
					step_r, step_t);
			// Calculate rates
			this->vel = -step_t / dt;
			this->rates = -step_r / dt;
#ifdef DEBUG
			std::cerr << "vel = " << this->vel << std::endl;
			std::cerr << "rates = " << this->rates << std::endl;
#endif
		} else {
			// Lost track of features
			// Assume constant rates and update pose
#ifdef DEBUG
			std::cerr << "WARNING! Not enough features left: " << this->tracked_pts.size() << std::endl;
			std::cerr << "Propagating position using previous rates..." << std::endl;
#endif
			// H_k,c = H_k,c-1 * H_c-1,c
			cv::Mat step_t, step_r; // H_c,c-1
			step_t = -this->vel * dt;
			step_r = -this->rates * dt;
			invert_pose(step_r, step_t, step_r, step_t); // step_r, _t is now H_c-1,c
			cv::composeRT(step_r, step_t, this->cam_r, this->cam_t,
					this->cam_r, this->cam_t);
		}
		PROFILER_END();
#ifdef DEBUG
		std::cerr << "Remaining points = " << this->tracked_pts.size() << std::endl;
#endif

		// Keep track of previous data
		PROFILER_START(copy);
		image.copyTo(this->prev_img);
		this->prev_timestamp = timestamp;
		PROFILER_END();
		// If not new keyframe
		//	Predict features
		PROFILER_END(); // updatePose
		//	return; // Skip keyframe update
	}

	if(this->keyframe.size() < this->keyframe_thres * this->keyframe_initial_size) {
		this->updateKeyframe(image, depth, intrinsic);
	}

	// std::cerr << "Keypoints: " << this->keyframe.size() << std::endl;
	image.copyTo(this->prev_img); // XXX

	PROFILER_END(); // updateImageDepth

#ifdef DEBUG
	// Show tracked points
	cv::Mat debug;
	cv::cvtColor(image, debug, CV_GRAY2BGR);
	for(int i = 0; i < this->tracked_pts.size(); ++i) {
		int radius = 10.0 / this->keyframe[i].z;
		std::ostringstream ss;
		ss << std::fixed << std::setprecision(1) << this->keyframe[i].z;
		cv::circle(debug, this->tracked_pts[i], radius, cv::Scalar(0, 255, 0));
		cv::putText(debug, ss.str(),
				cv::Point(this->tracked_pts[i].x + 10, this->tracked_pts[i].y + 3),
				cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0));
	}
	cv::imshow("keypoints", debug);

	// Print cam pose in world
	cv::Mat camw_r, camw_t;
	this->getPose(camw_r, camw_t);
	std::cerr << "cam r: " << camw_r << std::endl;
	std::cerr << "cam t: " << camw_t << std::endl;
#endif

	LogProfiler();
}


/**
 * Predict next keypoint positions
 *
 * Predict keypoint positions using IMU data, or assuming constant angular
 * rates. No keypoints are removed during this step
 * @param intrinsic Camera intrinsic matrix
 * @param predicted_pts Output: contains predicted positions
 */
void EVO::predictKeypoints(
			cv::InputArray intrinsic,
			std::vector<cv::Point2f> &predicted_pts,
			double dt) {
	PROFILER_START(predict);
	cv::Mat R; // Should be R_c,c-1
	if(this->imu_prev_timestamp > this->prev_timestamp) {
		// Use available IMU data
		this->imu_R.copyTo(R);
#ifdef DEBUG
		std::cerr << "imu_R = " << this->imu_R << std::endl;
		std::cerr << "imu_bias = " << this->imu_bias << std::endl;
#endif
		// Zero IMU orientation relative to the new image
		this->imu_R = cv::Mat::eye(3, 3, CV_64F); // TODO move away from this function
	} else {
		// Assume constant angular rate
//		R = cv::Mat::eye(3, 3, CV_64F);
		cv::Rodrigues(dt * this->rates, R); // R_c-1,c
		cv::transpose(R, R); // R_c,c-1
	}
	// Apply rotation to tracked points
	cv::Mat intr, M;
	intr = intrinsic.getMat();
	M = intr * R * (intr.inv());
#ifdef DEBUG
	std::cerr << "R = " << R << std::endl;
	std::cerr << "M = " << M << std::endl;
	std::cerr << "M.type() = " << M.type() << std::endl;
#endif
	cv::perspectiveTransform(this->tracked_pts, predicted_pts, M);
	PROFILER_END();
}


/**
 * Track keypoints
 *
 * Tracks the this->tracked_pts keypoints from the previous image this->prev_img
 * to the new image 'image'. predicted_pts is used as an initial guess for the
 * new keypoint positions.
 * Keypoints that were lost during tracking are removed from the tracked_pts and
 * keyframe.
 *
 * @param image New input image
 * @param predicted_pts Predicted new keypoint positions
 */
void EVO::trackKeypoints(
			const cv::Mat &image,
			const std::vector<cv::Point2f> &predicted_pts) {
	PROFILER_START(track);
	std::vector<cv::Point2f> prev_pts(this->tracked_pts);
	this->tracked_pts = predicted_pts;
	std::vector<uchar> is_tracked;
	cv::Mat err;
	cv::calcOpticalFlowPyrLK(this->prev_img, image, prev_pts, this->tracked_pts,
			is_tracked, err, cv::Size(15, 15)); // See Kelly et al., 2008 for window size.
	// Remove keypoints that were lost during tracking
	filter_vector(prev_pts, is_tracked);
	filter_vector(this->tracked_pts, is_tracked);
	filter_vector(this->keyframe, is_tracked);
	PROFILER_END();
#ifdef DEBUG
	std::cerr << "Tracked points: " << this->tracked_pts.size() << std::endl;
#endif
}


void EVO::updateKeyframe(
		const cv::Mat &image,
		const cv::Mat &depth,
		cv::InputArray intrinsic) {
	PROFILER_START(updateKeyframe);
	// Convert input args
	assert(depth.type() == CV_32F);
	cv::Mat intr = intrinsic.getMat();
	assert(intr.type() == CV_64F);
	assert(intr.size() == cv::Size(3, 3));
	const double fx = intr.at<double>(0, 0);
	const double fy = intr.at<double>(1, 1);
	const double cx = intr.at<double>(0, 2);
	const double cy = intr.at<double>(1, 2);
	assert(fx != 0 && fy != 0 && cx != 0 && cy != 0);
#ifdef DEBUG
	std::cerr << "intr = " << intr << std::endl;
	std::cerr << "fx = " << fx << ", fy = " << fy << ", cx = " << cx << ", cy = " << cy << std::endl;
#endif
	// Remove remaining keypoints if no depth information is available anymore
	PROFILER_START(depth_mask);
	cv::Mat depth_mask(depth.rows, depth.cols, CV_8UC1);
	cv::Mat image_mask(image.rows, image.cols, CV_8UC1);
	depth_mask = (depth == depth); 				// Filter unknown (NaN)
	depth_mask &= (depth > this->near_clip);	// Filter too close (-Inf)
	depth_mask &= (depth < 1.0e6);				// Filter too far (+Inf)
	cv::resize(depth_mask, image_mask, image_mask.size(), 0, 0, cv::INTER_NEAREST);
	PROFILER_END();
	PROFILER_START(filter_keyframe);
	int ratio = image.cols / depth.cols; // Assume integer ratio between image and depth map size
	std::vector<bool> depth_available(this->tracked_pts.size());
	for(int i = 0; i < this->tracked_pts.size(); ++i) {
		int depth_x = this->tracked_pts[i].x / ratio;
		int depth_y = this->tracked_pts[i].y / ratio;
		depth_available[i] =
				depth_x > 0 && depth_x < depth_mask.cols &&
				depth_y > 0 && depth_y < depth_mask.rows &&
				depth_mask.at<uchar>(depth_y, depth_x);
	}
	filter_vector(this->keyframe, depth_available);
	filter_vector(this->tracked_pts, depth_available);
	PROFILER_END();

	// Detect new keypoints
	PROFILER_START(detect_keypoints);
	int num_new_kp = this->target_keypts - this->keyframe.size();
	if(num_new_kp <= 0) return;

	std::vector<cv::KeyPoint> new_kp;
	std::vector<cv::Point2f> new_pts;
	this->detector->set("maxTotalKeypoints", num_new_kp);
	this->detector->detect(image, new_kp, image_mask);
	cv::KeyPoint::convert(new_kp, new_pts);
	PROFILER_END();

	// Store 3D coords in keyframe
	PROFILER_START(update_keyframe);
	this->tracked_pts.insert(this->tracked_pts.end(), new_pts.begin(), new_pts.end());
	this->keyframe.resize(this->tracked_pts.size());
	for(int i = 0; i < this->tracked_pts.size(); ++i) {
		int depth_x = this->tracked_pts[i].x / ratio;
		int depth_y = this->tracked_pts[i].y / ratio;
		float z = depth.at<float>(depth_y, depth_x);
		if(!std::isfinite(z)) {
			std::cerr << "ERROR: depth " << z << "at (" << depth_x << ", "
					<< depth_y << ") is not finite!" << std::endl;
			continue;
		}
		this->keyframe[i].x = (this->tracked_pts[i].x - cx) / fx * z;
		this->keyframe[i].y = (this->tracked_pts[i].y - cy) / fy * z;
		this->keyframe[i].z = z;
	}
	this->keyframe_initial_size = this->keyframe.size();
	PROFILER_END();

	// Update keyframe pose in world, zero cam pose in keyframe
	PROFILER_START(update_pose);
	this->getPose(this->key_r, this->key_t);
	this->cam_r = cv::Mat::zeros(3, 1, CV_64F);
	this->cam_t = cv::Mat::zeros(3, 1, CV_64F);
	PROFILER_END();

	PROFILER_END(); // updateKeyframe
	assert(this->tracked_pts.size() == this->keyframe.size());
}


} // namespace openevo
