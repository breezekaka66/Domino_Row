#pragma once

// Project#3/week3/Student 的 C++/CLI 專案單檔版，使用 Release | x64。
// 將本檔放到 CppCLR_WinformsProjekt1，取代原 Form1.h 後重建。
// 現場設定可直接編輯畫面中的 JSON，或修改下方 defaultSettingsJson。
// 原專案的 OpenCV 3.4.3、ur_rtde 和 Boost 連結設定照常使用。

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef BOOST_ALL_NO_LIB
#define BOOST_ALL_NO_LIB
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif

#pragma managed(push, off)
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_io_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "mswsock.lib")

namespace domino_form {
namespace {

const char defaultSettingsJson[] = R"json(
{
  "camera": {"index": 0, "width": 640, "height": 480, "min_area_px": 80},
  "hsv": {
    "red": [[[0, 90, 60], [10, 255, 255]], [[170, 90, 60], [179, 255, 255]]],
    "yellow": [[[18, 90, 75], [38, 255, 255]]],
    "blue": [[[95, 85, 50], [132, 255, 255]]]
  },
  "calibration": {"image_px": [], "robot_xy_m": []},
  "domino": {"height_m": 0.050, "width_m": 0.025, "thickness_m": 0.009, "size_tolerance": 0.30},
  "row": {
    "order": ["red", "yellow", "blue"], "origin_xy_m": [0.31, -0.04],
    "direction_deg": 0.0, "spacing_m": 0.032, "buffer_xy_m": [0.38, 0.06]
  },
  "gripper": {"do": 0, "open_high": 1, "settle_ms": 1000, "half_along_width_m": 0.019, "half_along_thickness_m": 0.017, "clearance_m": 0.002},
  "robot": {
    "ip": "", "grasp_reference_pose": [0.0, 0.0, 0.25546, 2.182, -2.266, 0.02], "grasp_reference_long_axis_deg": 0.0,
    "park_pose": [0.09738, -0.20041, 0.31251, 2.225, -2.219, 0.032],
    "safe_z_m": 0.3125, "lift_z_m": 0.3125, "place_z_m": 0.2554,
    "travel_speed_m_s": 0.05, "contact_speed_m_s": 0.02, "acceleration_m_s2": 0.2
  }
}
)json";
// Shared geometry and settings.
using XY = cv::Point2d;
using Pose = std::vector<double>;
constexpr double pi = 3.14159265358979323846;
constexpr double positionTolerance = 0.002;
constexpr double angleTolerance = 8 * pi / 180;
const std::array<std::string, 3> colors{"red", "yellow", "blue"};

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
    Scene scene{};
    bool ready = false;
    std::string issue;
};


// Bounded placement planner.

double angleDelta(double angle, double reference);
bool poseMatches(const FlatPose& a, const FlatPose& b);

namespace {

struct Rectangle {
    XY center, along, across;
    double halfWidth, halfThickness;

    Rectangle(const FlatPose& pose, double width, double thickness)
        : center(pose.xy), along(std::cos(pose.angle), std::sin(pose.angle)),
          across(-along.y, along.x), halfWidth(width), halfThickness(thickness) {}

    double extent(const XY& axis) const {
        return halfWidth * std::abs(along.dot(axis)) + halfThickness * std::abs(across.dot(axis));
    }
};

Rectangle gripAt(const FlatPose& pose, const Layout& layout) {
    return {pose, layout.toolHalfWidth + layout.clearance, layout.toolHalfThickness + layout.clearance};
}

bool fits(const Rectangle& rect, const Layout& layout) {
    for (int side : {-1, 1}) {
        for (int end : {-1, 1}) {
            const XY corner = rect.center + side * rect.halfWidth * rect.along
                + end * rect.halfThickness * rect.across;
            if (cv::pointPolygonTest(layout.workspace, cv::Point2f(corner), false) < 0) return false;
        }
    }
    return true;
}

bool overlaps(const Rectangle& a, const Rectangle& b) {
    for (const XY& axis : {a.along, a.across, b.along, b.across}) {
        if (std::abs((b.center - a.center).dot(axis)) > a.extent(axis) + b.extent(axis)) return false;
    }
    return true;
}

} // namespace

bool moveIsClear(const Scene& scene, int color, const FlatPose& dest, const Layout& layout) {
    const auto pick = gripAt(scene[color], layout);
    const auto place = gripAt(dest, layout);
    if (!fits(pick, layout) || !fits(place, layout)) return false;
    for (int other = 0; other < static_cast<int>(scene.size()); ++other) {
        if (other == color) continue;
        const Rectangle block(scene[other], layout.width / 2, layout.thickness / 2);
        if (overlaps(pick, block) || overlaps(place, block)) return false;
    }
    return true;
}

double angleDelta(double angle, double reference) {
    return std::remainder(angle - reference, pi);
}

bool poseMatches(const FlatPose& a, const FlatPose& b) {
    return cv::norm(a.xy - b.xy) <= positionTolerance
        && std::abs(angleDelta(a.angle, b.angle)) <= angleTolerance;
}

bool matches(const Scene& a, const Scene& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (!poseMatches(a[i], b[i])) return false;
    }
    return true;
}

std::vector<Move> plan(const Scene& initial, const Layout& layout) {
    for (const FlatPose& target : layout.targets) {
        if (!fits(gripAt(target, layout), layout)) {
            throw std::runtime_error("A row target is outside the calibrated area for the gripper.");
        }
    }
    const bool bufferFits = fits(gripAt(layout.buffer, layout), layout);

    struct State {
        std::array<int, 3> slots{}; // 0: original, 1: own target, 2: buffer.
        Scene scene;
        std::vector<Move> moves;
    };
    std::queue<State> pending;
    std::set<std::array<int, 3>> seen;
    State start;
    start.scene = initial;
    pending.push(start);
    seen.insert(start.slots);

    while (!pending.empty()) {
        State current = std::move(pending.front());
        pending.pop();
        if (matches(current.scene, layout.targets)) return current.moves;

        for (int slot : {1, 2}) {
            if (slot == 2 && !bufferFits) continue;
            for (int color = 0; color < static_cast<int>(colors.size()); ++color) {
                const bool toBuffer = slot == 2;
                const FlatPose& dest = toBuffer ? layout.buffer : layout.targets[color];
                if (current.slots[color] == slot || poseMatches(current.scene[color], dest)
                    || !moveIsClear(current.scene, color, dest, layout)) continue;

                State next = current;
                next.slots[color] = slot;
                if (!seen.insert(next.slots).second) continue;
                next.moves.push_back({color, toBuffer, current.scene[color], dest});
                next.scene[color] = dest;
                pending.push(std::move(next));
            }
        }
    }
    throw std::runtime_error("No clear pick-and-place sequence. Check block separation, row spacing, and the buffer.");
}

// Camera calibration and block localization.
namespace {

double number(const cv::FileNode& node, const char* name) {
    if ((!node.isInt() && !node.isReal()) || !std::isfinite(double(node))) {
        throw std::runtime_error(std::string("Expected a number for ") + name);
    }
    return double(node);
}

int integer(const cv::FileNode& group, const char* key, int min, int max) {
    const auto node = group[key];
    const double value = number(node, key);
    if (!node.isInt() || value < min || value > max) {
        throw std::runtime_error(std::string("Invalid integer for ") + key);
    }
    return int(value);
}

std::vector<double> numbers(const cv::FileNode& node, size_t count, const char* name, bool optional = false) {
    if (!node.isSeq() || (node.size() != count && !(optional && node.size() == 0))) {
        throw std::runtime_error(std::string("Invalid list for ") + name);
    }
    std::vector<double> values;
    for (const auto& item : node) values.push_back(number(item, name));
    return values;
}

XY xy(const cv::FileNode& node, const char* name) {
    const auto values = numbers(node, 2, name);
    return {values[0], values[1]};
}

std::vector<cv::Point2f> points(const cv::FileNode& node, const char* name) {
    if (!node.isSeq()) throw std::runtime_error(std::string("Invalid list for ") + name);
    std::vector<cv::Point2f> result;
    for (const auto& item : node) result.emplace_back(xy(item, name));
    return result;
}

} // namespace

void setCalibration(Settings& cfg, const std::vector<cv::Point2f>& pixels,
                    const std::vector<cv::Point2f>& robot) {
    if (pixels.size() < 4 || pixels.size() != robot.size()) {
        throw std::runtime_error("Calibration requires at least four corresponding pixel and robot XY points.");
    }
    for (const auto& p : pixels) {
        if (!(p.x >= 0 && p.x < cfg.width && p.y >= 0 && p.y < cfg.height)) {
            throw std::runtime_error("A calibration pixel is outside the configured image.");
        }
    }
    cv::convexHull(pixels, cfg.imageWorkspace);
    cv::convexHull(robot, cfg.layout.workspace);
    cfg.homography = cv::findHomography(pixels, robot);
    if (cfg.homography.empty() || !cv::checkRange(cfg.homography)
        || cfg.imageWorkspace.size() < 4 || cfg.layout.workspace.size() < 4) {
        throw std::runtime_error("Calibration requires four well-separated corners around the work area.");
    }
    const cv::Mat& h = cfg.homography;
    const auto denominator = [&](cv::Point2f p) {
        return h.at<double>(2, 0) * p.x + h.at<double>(2, 1) * p.y + h.at<double>(2, 2);
    };
    const double first = denominator(cfg.imageWorkspace.front());
    for (const auto p : cfg.imageWorkspace) {
        if (std::abs(denominator(p)) < 1e-8 || denominator(p) * first <= 0) {
            throw std::runtime_error("Calibration has a singularity in the work area. Check point correspondences.");
        }
    }
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(pixels, projected, cfg.homography);
    for (size_t i = 0; i < robot.size(); ++i) {
        if (cv::norm(robot[i] - projected[i]) > positionTolerance) {
            throw std::runtime_error("A calibration point has more than 2 mm of mapping error.");
        }
    }
}

Settings loadSettings(const std::string& json) {
    const auto start = json.find_first_not_of(" \t\r\n");
    if (start == std::string::npos || json[start] != '{') {
        throw std::runtime_error("The settings must be a JSON object.");
    }
    cv::FileStorage storage(json.substr(start), cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    if (!storage.isOpened()) throw std::runtime_error("Cannot read the JSON settings.");
    Settings cfg{};
    const auto camera = storage["camera"];
    cfg.camera = integer(camera, "index", 0, 99);
    cfg.width = integer(camera, "width", 1, 8192);
    cfg.height = integer(camera, "height", 1, 8192);
    cfg.minArea = number(camera["min_area_px"], "min_area_px");
    if (cfg.minArea <= 0) throw std::runtime_error("min_area_px must be positive.");

    const auto hsv = storage["hsv"];
    for (size_t i = 0; i < colors.size(); ++i) {
        const auto ranges = hsv[colors[i]];
        if (!ranges.isSeq() || ranges.size() == 0) throw std::runtime_error("Missing HSV ranges for " + colors[i]);
        for (const auto& band : ranges) {
            if (!band.isSeq() || band.size() != 2) throw std::runtime_error("An HSV range needs lower and upper bounds.");
            const auto lo = numbers(band[0], 3, "HSV lower");
            const auto hi = numbers(band[1], 3, "HSV upper");
            for (size_t j = 0; j < 3; ++j) {
                if (lo[j] < 0 || hi[j] > (j == 0 ? 179 : 255) || lo[j] > hi[j]) {
                    throw std::runtime_error("Invalid HSV bounds for " + colors[i]);
                }
            }
            cfg.hsv[i].push_back({cv::Scalar(lo[0], lo[1], lo[2]), cv::Scalar(hi[0], hi[1], hi[2])});
        }
    }

    const auto calibration = storage["calibration"];
    const auto pixels = points(calibration["image_px"], "image_px");
    const auto robotPoints = points(calibration["robot_xy_m"], "robot_xy_m");
    if (!pixels.empty() || !robotPoints.empty()) setCalibration(cfg, pixels, robotPoints);

    const auto block = storage["domino"];
    Layout& layout = cfg.layout;
    layout.width = number(block["width_m"], "width_m");
    layout.thickness = number(block["thickness_m"], "thickness_m");
    layout.height = number(block["height_m"], "height_m");
    layout.sizeTolerance = number(block["size_tolerance"], "size_tolerance");
    if (!(layout.thickness > 0 && layout.width > layout.thickness && layout.height > layout.thickness
        && layout.sizeTolerance > 0 && layout.sizeTolerance < 0.5)) {
        throw std::runtime_error("Invalid domino dimensions or size tolerance.");
    }

    const auto row = storage["row"];
    const auto order = row["order"];
    if (!order.isSeq() || order.size() != colors.size()) throw std::runtime_error("Order needs red, yellow and blue.");
    const double angle = number(row["direction_deg"], "direction_deg") * (pi / 180);
    const double spacing = number(row["spacing_m"], "spacing_m");
    if (spacing <= layout.thickness || spacing >= layout.height) {
        throw std::runtime_error("Row center spacing must be greater than thickness and less than height.");
    }
    const XY origin = xy(row["origin_xy_m"], "origin_xy_m");
    const XY direction(std::cos(angle), std::sin(angle));
    std::array<bool, 3> used{};
    int slot = 0;
    for (const auto& color : order) {
        const auto index = std::find(colors.begin(), colors.end(), std::string(color)) - colors.begin();
        if (index == static_cast<ptrdiff_t>(colors.size()) || used[index]) {
            throw std::runtime_error("Order needs each of red, yellow and blue exactly once.");
        }
        used[index] = true;
        layout.targets[index] = {origin + slot++ * spacing * direction, angle + pi / 2};
    }
    layout.buffer = {xy(row["buffer_xy_m"], "buffer_xy_m"), angle + pi / 2};

    const auto grip = storage["gripper"];
    cfg.output = integer(grip, "do", 0, 7);
    cfg.openHigh = integer(grip, "open_high", 0, 1) == 1;
    cfg.settleMs = integer(grip, "settle_ms", 1, 10000);
    layout.toolHalfWidth = number(grip["half_along_width_m"], "half_along_width_m");
    layout.toolHalfThickness = number(grip["half_along_thickness_m"], "half_along_thickness_m");
    layout.clearance = number(grip["clearance_m"], "clearance_m");
    if (layout.toolHalfWidth < layout.width / 2 || layout.toolHalfThickness < layout.thickness / 2
        || layout.clearance < 0) throw std::runtime_error("Gripper envelope must cover the domino footprint.");

    const auto robot = storage["robot"];
    cfg.ip = std::string(robot["ip"]);
    cfg.reference = numbers(robot["grasp_reference_pose"], 6, "grasp_reference_pose", true);
    cfg.park = numbers(robot["park_pose"], 6, "park_pose", true);
    const auto motionZ = [&](const char* key, double fallback) {
        const auto value = robot[key];
        return value.isNone() ? fallback : number(value, key);
    };
    cfg.safeZ = motionZ("safe_z_m", cfg.park.empty() ? 0.0 : cfg.park[2]);
    cfg.liftZ = motionZ("lift_z_m", cfg.safeZ);
    cfg.placeZ = motionZ("place_z_m", cfg.reference.empty() ? 0.0 : cfg.reference[2]);
    cfg.referenceAngle = number(robot["grasp_reference_long_axis_deg"], "grasp_reference_long_axis_deg") * (pi / 180);
    cfg.travelSpeed = number(robot["travel_speed_m_s"], "travel_speed_m_s");
    cfg.contactSpeed = number(robot["contact_speed_m_s"], "contact_speed_m_s");
    cfg.acceleration = number(robot["acceleration_m_s2"], "acceleration_m_s2");
    if (!(cfg.contactSpeed > 0 && cfg.travelSpeed >= cfg.contactSpeed && cfg.acceleration > 0)) {
        throw std::runtime_error("Invalid robot speeds or acceleration.");
    }
    return cfg;
}

Detection detect(const cv::Mat& frame, const Settings& cfg) {
    Detection result;
    if (frame.empty() || frame.cols != cfg.width || frame.rows != cfg.height) {
        result.issue = "Expected an image of " + std::to_string(cfg.width) + "x" + std::to_string(cfg.height);
        return result;
    }
    const bool calibrated = !cfg.homography.empty();
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    Scene scene{};
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
    for (size_t i = 0; i < colors.size(); ++i) {
        cv::Mat mask = cv::Mat::zeros(hsv.size(), CV_8U);
        for (const auto& band : cfg.hsv[i]) {
            cv::Mat partial;
            cv::inRange(hsv, band[0], band[1], partial);
            cv::bitwise_or(mask, partial, mask);
        }
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < cfg.minArea) continue;
            const auto bounds = cv::boundingRect(contour);
            if (bounds.x == 0 || bounds.y == 0 || bounds.br().x == frame.cols || bounds.br().y == frame.rows) continue;
            const cv::RotatedRect outline = cv::minAreaRect(contour);
            if (calibrated) {
                if (cv::pointPolygonTest(cfg.imageWorkspace, outline.center, false) < 0) continue;
                const std::vector<cv::Point2f> polygon(contour.begin(), contour.end());
                std::vector<cv::Point2f> mapped;
                cv::perspectiveTransform(polygon, mapped, cfg.homography);
                const auto metric = cv::minAreaRect(mapped);
                const double wide = std::max(metric.size.width, metric.size.height);
                const double thin = std::min(metric.size.width, metric.size.height);
                const Layout& dims = cfg.layout;
                if (std::abs(wide - dims.width) > dims.width * dims.sizeTolerance
                    || std::abs(thin - dims.thickness) > dims.thickness * dims.sizeTolerance
                    || cv::contourArea(mapped) < 0.75 * wide * thin) continue;
                const double angle = (metric.angle + (metric.size.width < metric.size.height ? 90 : 0)) * (pi / 180);
                scene[i] = {metric.center, angleDelta(angle, 0)};
            }
            result.outlines[i].push_back(outline);
        }
        const size_t count = result.outlines[i].size();
        if (count != 1) {
            if (!result.issue.empty()) result.issue += "  ";
            result.issue += colors[i] + ": " + std::to_string(count);
        }
    }
    if (calibrated && result.issue.empty()) {
        result.scene = scene;
        result.ready = true;
    }
    if (!calibrated && result.issue.empty()) result.issue = "Pixel preview. Calibration is empty.";
    return result;
}

cv::Mat annotate(const cv::Mat& frame, const Settings& cfg, const Detection& result) {
    cv::Mat view = frame.clone();
    const std::array<cv::Scalar, 3> ink{cv::Scalar(65, 95, 255), cv::Scalar(0, 230, 255), cv::Scalar(255, 185, 80)};
    if (!cfg.homography.empty()) {
        const std::vector<cv::Point> workspace(cfg.imageWorkspace.begin(), cfg.imageWorkspace.end());
        cv::polylines(view, workspace, true, {220, 220, 220}, 1, cv::LINE_AA);
        std::vector<cv::Point2f> targets;
        for (const auto& pose : cfg.layout.targets) targets.emplace_back(pose.xy);
        targets.emplace_back(cfg.layout.buffer.xy);
        std::vector<cv::Point2f> pixels;
        cv::perspectiveTransform(targets, pixels, cfg.homography.inv());
        for (size_t i = 0; i < pixels.size(); ++i) {
            const cv::Point2f p = pixels[i];
            if (!(p.x >= 0 && p.x < view.cols && p.y >= 0 && p.y < view.rows)) continue;
            const cv::Scalar color = i < colors.size() ? ink[i] : cv::Scalar(220, 220, 220);
            cv::drawMarker(view, p, color, cv::MARKER_CROSS, 15, 2, cv::LINE_AA);
            cv::putText(view, i < colors.size() ? "target " + colors[i] : "buffer", p + cv::Point2f(8, 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv::LINE_AA);
        }
    }
    for (size_t i = 0; i < colors.size(); ++i) {
        for (const auto& box : result.outlines[i]) {
            cv::Point2f corners[4];
            box.points(corners);
            for (int j = 0; j < 4; ++j) cv::line(view, corners[j], corners[(j + 1) % 4], ink[i], 2, cv::LINE_AA);
            cv::circle(view, box.center, 3, ink[i], cv::FILLED, cv::LINE_AA);
            cv::putText(view, colors[i], box.center + cv::Point2f(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.5, ink[i], 1, cv::LINE_AA);
        }
    }
    cv::rectangle(view, {0, 0}, {view.cols, 26}, {25, 25, 25}, cv::FILLED);
    cv::putText(view, result.ready ? "Three upright footprints detected" : result.issue, {8, 18},
                cv::FONT_HERSHEY_SIMPLEX, 0.48, {255, 255, 255}, 1, cv::LINE_AA);
    return view;
}

void checkCancellation(const std::atomic_bool& cancelled) {
    if (cancelled.load()) throw std::runtime_error("Stopped by operator.");
}

Scene stableScene(cv::VideoCapture& camera, const Settings& cfg, const std::atomic_bool& cancelled) {
    Scene anchor{};
    int consecutive = 0;
    std::string issue = "block positions did not stabilize";
    for (int attempt = 0; attempt < 24; ++attempt) {
        checkCancellation(cancelled);
        cv::Mat frame;
        if (!camera.read(frame)) throw std::runtime_error("Cannot read camera frame.");
        const auto result = detect(frame, cfg);
        if (!result.ready) {
            issue = result.issue;
            consecutive = 0;
            continue;
        }
        issue = "block positions did not stabilize";
        if (consecutive == 0 || !matches(anchor, result.scene)) {
            anchor = result.scene;
            consecutive = 0;
        }
        if (++consecutive == 4) return result.scene;
    }
    throw std::runtime_error("Camera scene unavailable: " + issue);
}

// UR3 robot control.
namespace {

using ur_rtde::RTDEControlInterface;
using ur_rtde::RTDEIOInterface;
using ur_rtde::RTDEReceiveInterface;

cv::Matx33d rotation(const Pose& pose) {
    cv::Matx33d matrix;
    cv::Rodrigues(cv::Vec3d(pose[3], pose[4], pose[5]), matrix);
    return matrix;
}

bool atPose(const Pose& actual, const Pose& target) {
    if (actual.size() != 6 || !std::all_of(actual.begin(), actual.end(), [](double x) { return std::isfinite(x); })) return false;
    const cv::Vec3d delta(actual[0] - target[0], actual[1] - target[1], actual[2] - target[2]);
    cv::Vec3d angularDelta;
    cv::Rodrigues(rotation(actual) * rotation(target).t(), angularDelta);
    return cv::norm(delta) <= 0.002 && cv::norm(angularDelta) <= 0.035;
}

Pose contactPose(const Settings& cfg, const FlatPose& block, double z) {
    const double angle = angleDelta(block.angle, cfg.referenceAngle);
    cv::Matx33d yaw;
    cv::Rodrigues(cv::Vec3d(0, 0, angle), yaw);
    cv::Vec3d rot;
    cv::Rodrigues(yaw * rotation(cfg.reference), rot);
    return {block.xy.x, block.xy.y, z, rot[0], rot[1], rot[2]};
}

std::array<Pose, 7> waypoints(const Settings& cfg, const FlatPose& from, const FlatPose& to) {
    const Pose pick = contactPose(cfg, from, cfg.reference[2]);
    const Pose place = contactPose(cfg, to, cfg.placeZ);
    Pose abovePick = pick;
    Pose lift = pick;
    Pose abovePlace = place;
    abovePick[2] = abovePlace[2] = cfg.safeZ;
    lift[2] = cfg.liftZ;
    return {abovePick, pick, lift, abovePlace, place, abovePlace, cfg.park};
}

void checkState(RTDEReceiveInterface& read, const std::atomic_bool& cancelled) {
    checkCancellation(cancelled);
    if (!read.isConnected()) throw std::runtime_error("Robot receive connection lost.");
    const int safety = read.getSafetyMode();
    if (read.getRobotMode() != 7 || (safety != 1 && safety != 2)) {
        throw std::runtime_error("Robot must be running in Normal or Reduced safety mode.");
    }
}

void checkWaypoints(RTDEControlInterface& control, const std::array<Pose, 7>& path) {
    for (const auto& pose : path) {
        if (!control.isPoseWithinSafetyLimits(pose)) throw std::runtime_error("A waypoint is out of UR reach or safety limits.");
    }
}

} // namespace

void printPose(const Settings& cfg, std::ostream& log) {
    if (cfg.ip.empty()) throw std::runtime_error("Set robot.ip before reading the TCP pose.");
    RTDEReceiveInterface read(cfg.ip);
    const Pose pose = read.getActualTCPPose();
    if (pose.size() != 6) throw std::runtime_error("No TCP pose received.");
    log << std::fixed << std::setprecision(6) << '[';
    for (size_t i = 0; i < pose.size(); ++i) log << (i == 0 ? "" : ", ") << pose[i];
    log << "]\n";
}

void execute(const Settings& cfg, cv::VideoCapture& camera, const Scene& initial,
             const std::vector<Move>& moves, const std::atomic_bool& cancelled, std::ostream& log) {
    if (moves.empty()) return;
    if (cfg.ip.empty() || cfg.reference.size() != 6 || cfg.park.size() != 6) {
        throw std::runtime_error("Set robot.ip and teach the grasp reference and park poses before starting.");
    }
    const double clearanceZ = std::max(cfg.reference[2], cfg.placeZ) + cfg.layout.height;
    if (std::min({cfg.safeZ, cfg.liftZ, cfg.park[2]}) < clearanceZ) {
        throw std::runtime_error("Safe, lift, and park Z must be at least one domino height above grasp and place Z.");
    }

    RTDEReceiveInterface read(cfg.ip);
    checkState(read, cancelled);
    if (!atPose(read.getActualTCPPose(), cfg.park)) {
        throw std::runtime_error("Move the robot to the taught park pose before starting.");
    }
    RTDEControlInterface control(cfg.ip);
    RTDEIOInterface io(cfg.ip);
    try {
        for (const auto& move : moves) {
            checkCancellation(cancelled);
            checkWaypoints(control, waypoints(cfg, move.from, move.to));
        }
        const auto go = [&](const Pose& target, double speed) {
            checkState(read, cancelled);
            if (!control.moveL(target, speed, cfg.acceleration) || !atPose(read.getActualTCPPose(), target)) {
                throw std::runtime_error("moveL failed or TCP did not reach its target.");
            }
        };
        const auto grip = [&](bool open) {
            checkState(read, cancelled);
            if (!io.setStandardDigitalOut(static_cast<uint8_t>(cfg.output), open == cfg.openHigh)) {
                throw std::runtime_error("Could not set the gripper output.");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.settleMs));
        };

        Scene observed = stableScene(camera, cfg, cancelled);
        if (!matches(observed, initial)) throw std::runtime_error("The blocks moved since planning. Run again to replan.");
        for (const auto& move : moves) {
            if (!poseMatches(observed[move.color], move.from) || !moveIsClear(observed, move.color, move.to, cfg.layout)) {
                throw std::runtime_error("The observed pickup or destination no longer matches the plan.");
            }
            const auto path = waypoints(cfg, observed[move.color], move.to);
            checkWaypoints(control, path);
            log << "Moving " << colors[move.color] << (move.toBuffer ? " to buffer\n" : " to row\n") << std::flush;
            grip(true);
            go(path[0], cfg.travelSpeed);
            go(path[1], cfg.contactSpeed);
            grip(false);
            go(path[2], cfg.contactSpeed);
            go(path[3], cfg.travelSpeed);
            go(path[4], cfg.contactSpeed);
            grip(true);
            go(path[5], cfg.contactSpeed);
            go(path[6], cfg.travelSpeed);

            Scene expected = observed;
            expected[move.color] = move.to;
            observed = stableScene(camera, cfg, cancelled);
            if (!matches(observed, expected)) throw std::runtime_error("Camera could not verify the block placement.");
        }
        if (!matches(observed, cfg.layout.targets)) throw std::runtime_error("Camera could not verify the final row.");
    } catch (...) {
        try { control.stopScript(); } catch (...) {}
        throw;
    }
    control.stopScript();
}

// Output and synthetic scene test.
void printFlat(const FlatPose& pose, std::ostream& log) {
    log << '(' << pose.xy.x << ", " << pose.xy.y << ") m, " << pose.angle * (180 / pi) << " deg";
}

void printPlan(const Scene& initial, const std::vector<Move>& moves, std::ostream& log) {
    log << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < colors.size(); ++i) {
        log << colors[i] << " at ";
        printFlat(initial[i], log);
        log << '\n';
    }
    if (moves.empty()) log << "Row is already in place.\n";
    for (size_t i = 0; i < moves.size(); ++i) {
        const auto& move = moves[i];
        log << i + 1 << ". " << colors[move.color] << (move.toBuffer ? " to buffer " : " to row ");
        printFlat(move.to, log);
        log << '\n';
    }
}

Scene readScene(const cv::Mat& frame, const Settings& cfg) {
    const auto result = detect(frame, cfg);
    if (!result.ready) throw std::runtime_error("Camera scene unavailable: " + result.issue);
    return result.scene;
}

cv::Mat drawDemoFrame(const Settings& cfg, const Scene& scene) {
    cv::Mat frame(cfg.height, cfg.width, CV_8UC3, cv::Scalar(45, 45, 45));
    const std::array<cv::Scalar, 3> paint{cv::Scalar(0, 0, 230), cv::Scalar(0, 220, 220), cv::Scalar(230, 70, 20)};
    for (size_t i = 0; i < scene.size(); ++i) {
        const auto& block = scene[i];
        const cv::RotatedRect footprint(cv::Point2f(block.xy), {float(cfg.layout.width), float(cfg.layout.thickness)},
                                        float(block.angle * (180 / pi)));
        std::vector<cv::Point2f> corners(4);
        footprint.points(corners.data());
        std::vector<cv::Point2f> pixels;
        cv::perspectiveTransform(corners, pixels, cfg.homography.inv());
        const std::vector<cv::Point> polygon(pixels.begin(), pixels.end());
        cv::fillConvexPoly(frame, polygon, paint[i]);
    }
    return frame;
}

void demo(Settings cfg, std::ostream& log) {
    setCalibration(cfg, {{20, 20}, {620, 25}, {610, 450}, {25, 460}},
                   {{0.255f, -0.105f}, {0.495f, -0.105f}, {0.495f, 0.095f}, {0.255f, 0.095f}});
    const Scene shuffled{cfg.layout.targets[1], cfg.layout.targets[0], {{0.445, 0.03}, 0.49}};
    Scene scene = readScene(drawDemoFrame(cfg, shuffled), cfg);
    if (!matches(scene, shuffled)) throw std::runtime_error("Demo localization did not recover the original poses.");
    const auto moves = plan(scene, cfg.layout);
    printPlan(scene, moves, log);
    bool usedBuffer = false;
    for (const auto& move : moves) {
        if (!poseMatches(scene[move.color], move.from) || !moveIsClear(scene, move.color, move.to, cfg.layout)) {
            throw std::runtime_error("Demo could not replay the plan.");
        }
        scene[move.color] = move.to;
        usedBuffer = usedBuffer || move.toBuffer;
    }
    const Scene result = readScene(drawDemoFrame(cfg, scene), cfg);
    if (!usedBuffer || !matches(result, cfg.layout.targets) || !plan(result, cfg.layout).empty()) {
        throw std::runtime_error("Demo did not resolve the swapped colors.");
    }
    log << "Simulated row verified.\n";
}

enum class Action { Demo, Preview, ReadTcp, Plan, Run };

cv::VideoCapture openCamera(const Settings& cfg) {
    cv::VideoCapture camera;
    if (!camera.open(cfg.camera, cv::CAP_DSHOW)) {
        throw std::runtime_error("Cannot open camera " + std::to_string(cfg.camera));
    }
    camera.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
    return camera;
}

struct PreviewState {
    cv::Mat hsv;
    std::string label;
    std::ostream& log;

    explicit PreviewState(std::ostream& log) : log(log) {}
};

const char previewWindow[] = "Domino Row - Camera Preview";

void onPreviewClick(int event, int x, int y, int, void* data) {
    auto& state = *static_cast<PreviewState*>(data);
    if (event != cv::EVENT_LBUTTONDOWN || x < 0 || x >= state.hsv.cols || y < 0 || y >= state.hsv.rows) return;
    const auto pixel = state.hsv.at<cv::Vec3b>(y, x);
    std::ostringstream coords;
    coords << '[' << x << ", " << y << "]   HSV [" << int(pixel[0]) << ", " << int(pixel[1])
           << ", " << int(pixel[2]) << ']';
    state.label = coords.str();
    state.log << state.label << '\n';
}

void preview(const Settings& cfg, const std::atomic_bool& cancelled, std::ostream& log) {
    auto camera = openCamera(cfg);
    PreviewState state(log);
    cv::namedWindow(previewWindow, cv::WINDOW_AUTOSIZE);
    struct ClosePreview {
        ~ClosePreview() { try { cv::destroyWindow(previewWindow); } catch (...) {} }
    } closePreview;
    cv::setMouseCallback(previewWindow, onPreviewClick, &state);
    while (!cancelled.load()) {
        cv::Mat frame;
        if (!camera.read(frame) || frame.empty()) throw std::runtime_error("Cannot read camera frame.");
        cv::cvtColor(frame, state.hsv, cv::COLOR_BGR2HSV);
        auto image = annotate(frame, cfg, detect(frame, cfg));
        if (!state.label.empty()) {
            cv::rectangle(image, {0, image.rows - 25}, {image.cols, image.rows}, {25, 25, 25}, cv::FILLED);
            cv::putText(image, state.label, {8, image.rows - 8}, cv::FONT_HERSHEY_SIMPLEX,
                        0.48, {255, 255, 255}, 1, cv::LINE_AA);
        }
        cv::imshow(previewWindow, image);
        const int key = cv::waitKey(16);
        if (key == 'q' || key == 'Q' || key == 27 || cvGetWindowHandle(previewWindow) == nullptr) break;
    }
}

std::string perform(Action action, const std::string& json, const std::atomic_bool& cancelled) {
    std::ostringstream log;
    try {
        checkCancellation(cancelled);
        if (action == Action::Demo) {
            demo(loadSettings(defaultSettingsJson), log);
            return log.str();
        }

        const auto cfg = loadSettings(json);
        if (action == Action::ReadTcp) {
            printPose(cfg, log);
            return log.str();
        }
        if (action == Action::Preview) {
            preview(cfg, cancelled, log);
            return log.str();
        }
        if (cfg.homography.empty()) throw std::runtime_error("Fill the image and robot calibration points first.");

        auto camera = openCamera(cfg);
        const auto initial = stableScene(camera, cfg, cancelled);
        const auto moves = plan(initial, cfg.layout);
        printPlan(initial, moves, log);
        if (action == Action::Run && !moves.empty()) {
            execute(cfg, camera, initial, moves, cancelled, log);
            log << "Row verified by the camera.\n";
        }
    } catch (const std::exception& error) {
        log << "Error: " << error.what() << '\n';
    }
    return log.str();
}

} // namespace
} // namespace domino_form
#pragma managed(pop)

#include <msclr/marshal_cppstd.h>

namespace CppCLRWinformsProjekt {

using namespace System;
using namespace System::ComponentModel;
using namespace System::Drawing;
using namespace System::IO;
using namespace System::Windows::Forms;
using msclr::interop::marshal_as;

public ref class Form1 : public Form {
public:
    Form1() {
        InitializeComponent();
        cancellation = new std::atomic_bool(false);
    }

protected:
    ~Form1() {
        delete cancellation;
        cancellation = nullptr;
    }

    virtual void OnFormClosing(FormClosingEventArgs^ e) override {
        Form::OnFormClosing(e);
        if (!e->Cancel && worker->IsBusy) {
            e->Cancel = true;
            closeWhenFinished = true;
            RequestStop();
        }
    }

private:
    BackgroundWorker^ worker;
    FlowLayoutPanel^ actions;
    TextBox^ settingsBox;
    TextBox^ logBox;
    Label^ status;
    Button^ stopButton;
    std::atomic_bool* cancellation;
    bool closeWhenFinished = false;

    void InitializeComponent() {
        Text = L"Domino Row | UR3 三色骨牌";
        ClientSize = Drawing::Size(824, 680);
        MinimumSize = Drawing::Size(680, 540);
        StartPosition = FormStartPosition::CenterScreen;
        Font = gcnew Drawing::Font(L"Microsoft JhengHei UI", 10);

        worker = gcnew BackgroundWorker();
        worker->DoWork += gcnew DoWorkEventHandler(this, &Form1::RunWorker);
        worker->RunWorkerCompleted += gcnew RunWorkerCompletedEventHandler(this, &Form1::FinishWorker);

        auto layout = gcnew TableLayoutPanel();
        layout->ColumnCount = 1;
        layout->RowCount = 6;
        layout->Dock = DockStyle::Fill;
        layout->Padding = Windows::Forms::Padding(12);
        layout->RowStyles->Add(gcnew RowStyle(SizeType::AutoSize));
        layout->RowStyles->Add(gcnew RowStyle(SizeType::AutoSize));
        layout->RowStyles->Add(gcnew RowStyle(SizeType::Percent, 65));
        layout->RowStyles->Add(gcnew RowStyle(SizeType::AutoSize));
        layout->RowStyles->Add(gcnew RowStyle(SizeType::Percent, 35));
        layout->RowStyles->Add(gcnew RowStyle(SizeType::AutoSize));
        Controls->Add(layout);

        auto buttons = gcnew FlowLayoutPanel();
        buttons->AutoSize = true;
        buttons->Dock = DockStyle::Fill;
        layout->Controls->Add(buttons, 0, 0);
        actions = gcnew FlowLayoutPanel();
        actions->AutoSize = true;
        actions->Margin = Windows::Forms::Padding(0);
        buttons->Controls->Add(actions);
        AddAction(L"離線測試", domino_form::Action::Demo);
        AddAction(L"相機預覽", domino_form::Action::Preview);
        AddAction(L"讀取 TCP", domino_form::Action::ReadTcp);
        AddAction(L"試排", domino_form::Action::Plan);
        AddAction(L"開始排列", domino_form::Action::Run);

        stopButton = gcnew Button();
        stopButton->Text = L"停止";
        stopButton->AutoSize = true;
        stopButton->Enabled = false;
        stopButton->Click += gcnew EventHandler(this, &Form1::StopClicked);
        buttons->Controls->Add(stopButton);

        auto settingsHeader = gcnew FlowLayoutPanel();
        settingsHeader->AutoSize = true;
        settingsHeader->Dock = DockStyle::Fill;
        settingsHeader->Controls->Add(MakeLabel(L"現場設定（JSON）"));
        auto load = gcnew Button();
        load->Text = L"載入設定";
        load->AutoSize = true;
        load->Click += gcnew EventHandler(this, &Form1::LoadSettings);
        settingsHeader->Controls->Add(load);
        auto save = gcnew Button();
        save->Text = L"儲存設定";
        save->AutoSize = true;
        save->Click += gcnew EventHandler(this, &Form1::SaveSettings);
        settingsHeader->Controls->Add(save);
        layout->Controls->Add(settingsHeader, 0, 1);

        settingsBox = MakeTextBox();
        settingsBox->Text = (gcnew String(domino_form::defaultSettingsJson))->Trim()->Replace(L"\n", L"\r\n");
        layout->Controls->Add(settingsBox, 0, 2);

        layout->Controls->Add(MakeLabel(L"執行紀錄"), 0, 3);
        logBox = MakeTextBox();
        logBox->ReadOnly = true;
        logBox->Text = L"相機預覽可點選像素並讀取 HSV。按 Q、Esc 或「停止」結束預覽。";
        layout->Controls->Add(logBox, 0, 4);

        status = MakeLabel(L"就緒");
        layout->Controls->Add(status, 0, 5);
    }

    static Label^ MakeLabel(String^ text) {
        auto label = gcnew Label();
        label->AutoSize = true;
        label->Text = text;
        label->Margin = Windows::Forms::Padding(3, 6, 3, 4);
        return label;
    }

    static TextBox^ MakeTextBox() {
        auto box = gcnew TextBox();
        box->Multiline = true;
        box->ScrollBars = ScrollBars::Both;
        box->WordWrap = false;
        box->Font = gcnew Drawing::Font(L"Consolas", 10);
        box->Dock = DockStyle::Fill;
        return box;
    }

    void AddAction(String^ text, domino_form::Action action) {
        auto button = gcnew Button();
        button->Text = text;
        button->AutoSize = true;
        button->Tag = static_cast<int>(action);
        button->Click += gcnew EventHandler(this, &Form1::StartClicked);
        actions->Controls->Add(button);
    }

    void StartClicked(Object^ sender, EventArgs^) {
        if (worker->IsBusy) return;
        const auto action = static_cast<domino_form::Action>(safe_cast<int>(safe_cast<Button^>(sender)->Tag));
        cancellation->store(false);
        actions->Enabled = false;
        stopButton->Enabled = true;
        settingsBox->ReadOnly = true;
        status->Text = L"執行中";
        logBox->Clear();
        worker->RunWorkerAsync(gcnew array<Object^>{static_cast<int>(action), settingsBox->Text});
    }

    void RunWorker(Object^, DoWorkEventArgs^ e) {
        auto input = safe_cast<array<Object^>^>(e->Argument);
        const auto action = static_cast<domino_form::Action>(safe_cast<int>(input[0]));
        auto json = marshal_as<std::string>(safe_cast<String^>(input[1]));
        const auto message = domino_form::perform(action, json, *cancellation);
        e->Result = gcnew String(message.c_str());
    }

    void FinishWorker(Object^, RunWorkerCompletedEventArgs^ e) {
        String^ result = e->Error != nullptr ? e->Error->Message : safe_cast<String^>(e->Result);
        logBox->Text = result->Replace(L"\n", L"\r\n");
        logBox->SelectionStart = logBox->TextLength;
        logBox->ScrollToCaret();
        actions->Enabled = true;
        stopButton->Enabled = false;
        settingsBox->ReadOnly = false;
        status->Text = L"就緒";
        if (closeWhenFinished) Close();
    }

    void RequestStop() {
        cancellation->store(true);
        stopButton->Enabled = false;
        status->Text = L"等待目前步驟結束";
    }

    void StopClicked(Object^, EventArgs^) {
        RequestStop();
    }

    void LoadSettings(Object^, EventArgs^) {
        if (worker->IsBusy) return;
        auto dialog = gcnew OpenFileDialog();
        dialog->Filter = L"JSON (*.json)|*.json";
        try {
            if (dialog->ShowDialog(this) == Windows::Forms::DialogResult::OK)
                settingsBox->Text = File::ReadAllText(dialog->FileName);
        } catch (Exception^ e) {
            logBox->Text = e->Message;
        } finally {
            delete dialog;
        }
    }

    void SaveSettings(Object^, EventArgs^) {
        auto dialog = gcnew SaveFileDialog();
        dialog->Filter = L"JSON (*.json)|*.json";
        dialog->FileName = L"domino.local.json";
        try {
            if (dialog->ShowDialog(this) == Windows::Forms::DialogResult::OK)
                File::WriteAllText(dialog->FileName, settingsBox->Text);
        } catch (Exception^ e) {
            logBox->Text = e->Message;
        } finally {
            delete dialog;
        }
    }
};

} // namespace CppCLRWinformsProjekt
