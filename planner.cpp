#include "domino.hpp"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <queue>
#include <set>
#include <stdexcept>

namespace domino {
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

} // namespace domino
