#include "domino.hpp"

#include <opencv2/calib3d.hpp>
#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_io_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace domino {
namespace {

using ur_rtde::RTDEControlInterface;
using ur_rtde::RTDEIOInterface;
using ur_rtde::RTDEReceiveInterface;

volatile std::sig_atomic_t interrupted = 0;

void onInterrupt(int) { interrupted = 1; }

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

Pose contactPose(const Settings& cfg, const FlatPose& block) {
    const double angle = angleDelta(block.angle, cfg.referenceAngle);
    cv::Matx33d yaw;
    cv::Rodrigues(cv::Vec3d(0, 0, angle), yaw);
    cv::Vec3d rot;
    cv::Rodrigues(yaw * rotation(cfg.reference), rot);
    return {block.xy.x, block.xy.y, cfg.reference[2], rot[0], rot[1], rot[2]};
}

std::array<Pose, 7> waypoints(const Settings& cfg, const FlatPose& from, const FlatPose& to) {
    const Pose pick = contactPose(cfg, from);
    const Pose place = contactPose(cfg, to);
    Pose abovePick = pick;
    Pose abovePlace = place;
    abovePick[2] = abovePlace[2] = cfg.park[2];
    return {abovePick, pick, abovePick, abovePlace, place, abovePlace, cfg.park};
}

void checkState(RTDEReceiveInterface& read) {
    if (interrupted) throw std::runtime_error("Interrupted by operator.");
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

void printPose(const Settings& cfg) {
    if (cfg.ip.empty()) throw std::runtime_error("Set robot.ip before reading the TCP pose.");
    RTDEReceiveInterface read(cfg.ip);
    const Pose pose = read.getActualTCPPose();
    if (pose.size() != 6) throw std::runtime_error("No TCP pose received.");
    std::cout << std::fixed << std::setprecision(6) << '[';
    for (size_t i = 0; i < pose.size(); ++i) std::cout << (i == 0 ? "" : ", ") << pose[i];
    std::cout << "]\n";
}

void execute(const Settings& cfg, cv::VideoCapture& camera, const Scene& initial,
             const std::vector<Move>& moves) {
    if (moves.empty()) return;
    if (cfg.ip.empty() || cfg.reference.size() != 6 || cfg.park.size() != 6) {
        throw std::runtime_error("Set robot.ip and teach the grasp reference and park poses before --run.");
    }
    if (cfg.park[2] < cfg.reference[2] + cfg.layout.height) {
        throw std::runtime_error("Park/travel Z must be at least one domino height above the grasp Z.");
    }

    interrupted = 0;
    const auto oldHandler = std::signal(SIGINT, onInterrupt);
    struct RestoreSignal {
        decltype(oldHandler) handler;
        ~RestoreSignal() { if (handler != SIG_ERR) std::signal(SIGINT, handler); }
    } restore{oldHandler};

    RTDEReceiveInterface read(cfg.ip);
    checkState(read);
    if (!atPose(read.getActualTCPPose(), cfg.park)) {
        throw std::runtime_error("Move the robot to the taught park pose before starting.");
    }
    RTDEControlInterface control(cfg.ip);
    RTDEIOInterface io(cfg.ip);
    try {
        for (const auto& move : moves) checkWaypoints(control, waypoints(cfg, move.from, move.to));
        const auto go = [&](const Pose& target, double speed) {
            checkState(read);
            if (!control.moveL(target, speed, cfg.acceleration) || !atPose(read.getActualTCPPose(), target)) {
                throw std::runtime_error("moveL failed or TCP did not reach its target.");
            }
        };
        const auto grip = [&](bool open) {
            checkState(read);
            if (!io.setStandardDigitalOut(static_cast<uint8_t>(cfg.output), open == cfg.openHigh)) {
                throw std::runtime_error("Could not set the gripper output.");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.settleMs));
        };

        Scene observed = stableScene(camera, cfg);
        if (!matches(observed, initial)) throw std::runtime_error("The blocks moved since planning. Run again to replan.");
        for (const auto& move : moves) {
            if (!poseMatches(observed[move.color], move.from) || !moveIsClear(observed, move.color, move.to, cfg.layout)) {
                throw std::runtime_error("The observed pickup or destination no longer matches the plan.");
            }
            const auto path = waypoints(cfg, observed[move.color], move.to);
            checkWaypoints(control, path);
            std::cout << "Moving " << colors[move.color] << (move.toBuffer ? " to buffer\n" : " to row\n") << std::flush;
            go(path[0], cfg.travelSpeed);
            grip(true);
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
            observed = stableScene(camera, cfg);
            if (!matches(observed, expected)) throw std::runtime_error("Camera could not verify the block placement.");
        }
        if (!matches(observed, cfg.layout.targets)) throw std::runtime_error("Camera could not verify the final row.");
    } catch (...) {
        try { control.stopScript(); } catch (...) {}
        throw;
    }
    control.stopScript();
}

} // namespace domino
