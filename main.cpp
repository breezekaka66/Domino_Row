#include "domino.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace domino {
namespace {

const char* windowName = "Domino Row";
const char* usage =
    "domino [--config FILE] [--preview | --plan | --run | --tcp | --demo] [--image FILE]\n"
    "  --preview   Show detections. Click for pixel coordinates and HSV. Press Q or Escape to exit.\n"
    "  --plan      Print a camera-based plan without connecting to the robot (default).\n"
    "  --run       Plan, move the UR3, and verify each placement with the camera.\n"
    "  --tcp       Read the robot's current TCP pose.\n"
    "  --demo      Simulate a row from a synthetic image, with no camera or robot.\n"
    "  --image     Use a saved image with --preview or --plan.\n";

cv::VideoCapture openCamera(const Settings& cfg) {
    cv::VideoCapture camera;
    if (!camera.open(cfg.camera, cv::CAP_DSHOW)) throw std::runtime_error("Cannot open camera " + std::to_string(cfg.camera));
    camera.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
    return camera;
}

void onClick(int event, int x, int y, int, void* userData) {
    const auto& hsv = *static_cast<cv::Mat*>(userData);
    if (event != cv::EVENT_LBUTTONDOWN || x < 0 || x >= hsv.cols || y < 0 || y >= hsv.rows) return;
    const auto pixel = hsv.at<cv::Vec3b>(y, x);
    std::cout << '[' << x << ", " << y << "]  HSV [" << int(pixel[0]) << ", " << int(pixel[1])
              << ", " << int(pixel[2]) << "]\n" << std::flush;
}

void preview(const Settings& cfg, cv::Mat frame) {
    cv::VideoCapture camera;
    const bool live = frame.empty();
    if (live) camera = openCamera(cfg);
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::Mat hsv;
    cv::setMouseCallback(windowName, onClick, &hsv);
    while (true) {
        if (live && !camera.read(frame)) throw std::runtime_error("Cannot read camera frame.");
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::imshow(windowName, annotate(frame, cfg, detect(frame, cfg)));
        const int key = cv::waitKey(live ? 15 : 100);
        if (key == 27 || key == 'q' || key == 'Q' || cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1) break;
    }
    cv::destroyAllWindows();
}

void printFlat(const FlatPose& pose) {
    std::cout << '(' << pose.xy.x << ", " << pose.xy.y << ") m, " << pose.angle * (180 / pi) << " deg";
}

void printPlan(const Scene& initial, const std::vector<Move>& moves) {
    std::cout << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < colors.size(); ++i) {
        std::cout << colors[i] << " at ";
        printFlat(initial[i]);
        std::cout << '\n';
    }
    if (moves.empty()) std::cout << "Row is already in place.\n";
    for (size_t i = 0; i < moves.size(); ++i) {
        const auto& move = moves[i];
        std::cout << i + 1 << ". " << colors[move.color] << (move.toBuffer ? " to buffer " : " to row ");
        printFlat(move.to);
        std::cout << '\n';
    }
}

Scene readScene(const cv::Mat& frame, const Settings& cfg) {
    const auto result = detect(frame, cfg);
    if (!result.scene) throw std::runtime_error("Camera scene unavailable: " + result.issue);
    return *result.scene;
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

void demo(Settings cfg) {
    setCalibration(cfg, {{20, 20}, {620, 25}, {610, 450}, {25, 460}},
                   {{0.255f, -0.105f}, {0.495f, -0.105f}, {0.495f, 0.095f}, {0.255f, 0.095f}});
    const Scene shuffled{cfg.layout.targets[1], cfg.layout.targets[0], {{0.445, 0.03}, 0.49}};
    Scene scene = readScene(drawDemoFrame(cfg, shuffled), cfg);
    if (!matches(scene, shuffled)) throw std::runtime_error("Demo localization did not recover the original poses.");
    const auto moves = plan(scene, cfg.layout);
    printPlan(scene, moves);
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
    std::cout << "Simulated row verified.\n";
}

} // namespace
} // namespace domino

int main(int argc, char* argv[]) {
    using namespace domino;
    try {
        std::string config;
        std::string image;
        std::string mode;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                std::cout << usage;
                return 0;
            }
            if (arg == "--config" || arg == "--image") {
                if (++i == argc || std::string(argv[i]).rfind("--", 0) == 0) {
                    throw std::runtime_error("Missing file after " + arg);
                }
                (arg == "--config" ? config : image) = argv[i];
                continue;
            }
            if (arg != "--preview" && arg != "--plan" && arg != "--run" && arg != "--tcp" && arg != "--demo") {
                throw std::runtime_error("Unknown option: " + arg);
            }
            if (!mode.empty()) throw std::runtime_error("Choose one mode. See --help.");
            mode = arg;
        }
        if (mode.empty()) mode = "--plan";
        if (!image.empty() && mode != "--preview" && mode != "--plan") {
            throw std::runtime_error("--image is available with --preview or --plan.");
        }
        if (config.empty()) {
            config = mode != "--demo" && std::filesystem::exists("domino.local.json")
                ? "domino.local.json" : "domino.example.json";
        }
        const Settings cfg = loadSettings(config);
        if (mode == "--demo") {
            demo(cfg);
            return 0;
        }
        if (mode == "--tcp") {
            printPose(cfg);
            return 0;
        }

        cv::Mat frame;
        if (!image.empty()) {
            frame = cv::imread(image, cv::IMREAD_COLOR);
            if (frame.empty()) throw std::runtime_error("Cannot open image: " + image);
        }
        if (mode == "--preview") {
            preview(cfg, frame);
            return 0;
        }
        if (cfg.homography.empty()) throw std::runtime_error("Calibrate the image-to-robot transform first. See README.md.");

        cv::VideoCapture camera;
        if (frame.empty()) camera = openCamera(cfg);
        const auto initial = frame.empty() ? stableScene(camera, cfg) : readScene(frame, cfg);
        const auto moves = plan(initial, cfg.layout);
        printPlan(initial, moves);
        if (mode == "--run" && !moves.empty()) {
            execute(cfg, camera, initial, moves);
            std::cout << "Row verified by the camera.\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
