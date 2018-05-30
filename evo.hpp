#include <opencv2/opencv.hpp>

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
	cv::Ptr<cv::FeatureDetector> detector;
};

}
