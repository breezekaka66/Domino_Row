#include "domino.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace domino {
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

Settings loadSettings(const std::string& file) {
    cv::FileStorage storage(file, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!storage.isOpened()) throw std::runtime_error("Cannot read config: " + file);
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
    if (calibrated && result.issue.empty()) result.scene = scene;
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
    cv::putText(view, result.scene ? "Three upright footprints detected" : result.issue, {8, 18},
                cv::FONT_HERSHEY_SIMPLEX, 0.48, {255, 255, 255}, 1, cv::LINE_AA);
    return view;
}

Scene stableScene(cv::VideoCapture& camera, const Settings& cfg) {
    std::optional<Scene> anchor;
    int consecutive = 0;
    std::string issue = "block positions did not stabilize";
    for (int attempt = 0; attempt < 24; ++attempt) {
        cv::Mat frame;
        if (!camera.read(frame)) throw std::runtime_error("Cannot read camera frame.");
        const auto result = detect(frame, cfg);
        if (!result.scene) {
            issue = result.issue;
            anchor.reset();
            consecutive = 0;
            continue;
        }
        issue = "block positions did not stabilize";
        if (!anchor || !matches(*anchor, *result.scene)) {
            anchor = result.scene;
            consecutive = 0;
        }
        if (++consecutive == 4) return *result.scene;
    }
    throw std::runtime_error("Camera scene unavailable: " + issue);
}

} // namespace domino
