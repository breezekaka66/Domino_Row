#pragma once

#include <opencv2/core.hpp>
#include <array>
#include <optional>
#include <string>
#include <vector>

namespace cv { class VideoCapture; }

namespace domino {

using XY = cv::Point2d;
using Pose = std::vector<double>;
constexpr double pi = 3.14159265358979323846;
constexpr double positionTolerance = 0.002;
constexpr double angleTolerance = 8 * pi / 180;
inline const std::array<std::string, 3> colors{"red", "yellow", "blue"};

struct FlatPose {
    XY xy;
    double angle; // Long axis, measured counterclockwise from robot +X.
};
using Scene = std::array<FlatPose, 3>;

struct Move {
    int color;
    bool toBuffer;
    FlatPose from;
    FlatPose to;
};

struct Layout {
    double width, thickness, height, sizeTolerance;
    double toolHalfWidth, toolHalfThickness, clearance;
    Scene targets;
    FlatPose buffer;
    std::vector<cv::Point2f> workspace;
};

struct Settings {
    int camera, width, height;
    double minArea;
    std::array<std::vector<std::array<cv::Scalar, 2>>, 3> hsv;
    cv::Mat homography;
    std::vector<cv::Point2f> imageWorkspace;
    Layout layout;
    std::string ip;
    Pose reference, park;
    double safeZ, liftZ, placeZ;
    double referenceAngle, travelSpeed, contactSpeed, acceleration;
    int output, settleMs;
    bool openHigh;
};

struct Detection {
    std::array<std::vector<cv::RotatedRect>, 3> outlines;
    std::optional<Scene> scene;
    std::string issue;
};

double angleDelta(double angle, double reference);
bool poseMatches(const FlatPose& a, const FlatPose& b);
bool matches(const Scene& a, const Scene& b);
bool moveIsClear(const Scene& scene, int color, const FlatPose& dest, const Layout& layout);
std::vector<Move> plan(const Scene& initial, const Layout& layout);
Settings loadSettings(const std::string& file);
void setCalibration(Settings& cfg, const std::vector<cv::Point2f>& pixels,
                    const std::vector<cv::Point2f>& robot);
Detection detect(const cv::Mat& frame, const Settings& cfg);
cv::Mat annotate(const cv::Mat& frame, const Settings& cfg, const Detection& result);
void printPose(const Settings& cfg);
void execute(const Settings& cfg, cv::VideoCapture& camera, const Scene& initial,
             const std::vector<Move>& moves);
Scene stableScene(cv::VideoCapture& camera, const Settings& cfg);

} // namespace domino
