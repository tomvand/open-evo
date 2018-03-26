#include <opencv2/opencv.hpp>

#include <vector>

namespace openevo {

class EVO {
public:
	EVO(void);
	virtual ~EVO(void);

	void updateIMU(void);

	void updateImageDepth(const cv::Mat &image, const cv::Mat &depth);
//	void updateImagePair(void); // TODO
//	void updateImageDisparity(void); // TODO

private:
	bool initialized;
	cv::Ptr<cv::FeatureDetector> detector;
	std::vector<cv::Point2f> prev_pts;
	cv::Mat prev_img;
};

}
