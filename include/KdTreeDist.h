// Zach Youssef, 4/16/26
// Repurposing KD trees for initializing gaussian splat scales

// Original:
// Zach Youssef, 2/19/26
// Header defining a k-d tree structure for k-nearest neighbors classification

#pragma once

#include <math.h>
#include <limits>
#include <vector>
#include <stack>
#include <string>
#include <variant>
#include <unordered_map>
#include <unordered_set>

class KdTree {
private:
    struct Point {
        std::vector<float> value;
    };

    struct PartitionPlane {
        size_t axis;
        float median;
    };

    struct Node {
        std::variant<Point, PartitionPlane> value;
        size_t leftChild;
        size_t rightChild;

        bool isLeaf() {
            return std::holds_alternative<Point>(value);
        }
    };

public:
    KdTree(){}

    // Construct a K-D Tree from the given labeleld points
    // Assumes all point vectors are the same length 'K'
    KdTree(const std::vector<std::vector<float>>& points, size_t K): K(K) {
        // Reshape the data to a flat list of points & corresponding labels
        std::vector<Point> pointList;
        for (const auto& point : points) {
            pointList.push_back({point});
        }

        root_ = constructRecursive(pointList, 0);
    }

    // Returns the average distance to the nearest K points
    const std::vector<float> avgDistToKNearest(const std::vector<float>& point, size_t k) {
        std::vector<Point> nearest;
        std::vector<float> distances;
        size_t farthestNeighbor = 0;

        std::stack<size_t> path{};
        std::unordered_set<size_t> visited{};
        path.push(root_);
        Node node;
        while(!path.empty()) {
            size_t nodeIdx = path.top();
            node = nodes_.at(nodeIdx);
            visited.emplace(nodeIdx);

            if (node.isLeaf()) {
                // Update our list of neighbors if this point is closer than any of them
                updateKNearestSearch(std::get<Point>(node.value), point, nearest, distances, farthestNeighbor, k);
                path.pop();
            } else {
                const auto& partitionPlane = std::get<PartitionPlane>(node.value);
                float distToMedian = point[partitionPlane.axis] - partitionPlane.median;

                bool checkAltBranch = (nearest.size() < k) || ((distToMedian * distToMedian) <= distances[farthestNeighbor]);

                size_t next = distToMedian <= 0 ? node.leftChild : node.rightChild;
                size_t alt = distToMedian <= 0 ? node.rightChild : node.leftChild;
                if (!visited.contains(next)) {
                    // If we haven't visited the child on the 'correct' side of the plane, do so
                    path.push(next);
                } else if (checkAltBranch && !visited.contains(alt)) {
                    // If we haven't visited the child on the 'correct' side of the plane, do so
                    // If we have < k neighbors or the other side of the tree is within our search radius, 
                    // visit the other side of the plane
                    path.push(alt);
                } else {
                    // Recur back up the tree
                    path.pop();
                }
            }
        }

        // Compute average of distance to neighbors
        std::vector<float> avg;
        avg.resize(this->K, 0.f);
        for (const auto& neighbor: nearest) {
            for (int i = 0; i < this->K; ++i) {
                avg[i] = abs(point[i] - neighbor.value[i]);
            }
        }
        for (int i = 0; i < this->K; ++i) {
            avg[i] /= this->K;
        }

        return avg;
    }

private:
    // Helper for updating distance bookeeping for k nearest neighbors
    const void updateKNearestSearch(const Point& leaf, 
                              const std::vector<float>& point, 
                              std::vector<Point>& nearest, 
                              std::vector<float>& distances, 
                              size_t& farthestNeighbor,
                              size_t k) {
        float leafDist = ssd(point, leaf.value);
        // If we havent found k neighbors yet, then we always add this leaf
        if (nearest.size() < k) {
            nearest.push_back(leaf);
            distances.push_back(leafDist);
        } else {
            // Compare against our farthest neighbor
            if (leafDist < distances.at(farthestNeighbor)) {
                nearest[farthestNeighbor] = leaf;
                distances[farthestNeighbor] = leafDist;
            }
        }

        // Update which nearest neighbor is now the farthest
        float worstDistance = std::numeric_limits<float>::min();
        for (size_t i = 0; i < distances.size(); ++i) {
            if (distances.at(i) > worstDistance) {
                farthestNeighbor = i;
            }
        }
    }


    // Sum squared distance
    float ssd(const std::vector<float>& a, const std::vector<float>& b) {
        float sum;
        for (int i = 0; i < a.size(); ++i) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }

    // Praying my training data doesn't exceed the stack height because I find tree construction
    // much easier to write recursively
    size_t constructRecursive(const std::vector<Point>& pointList, size_t depth) {
        // Base case - we've reached a leaf
        if (pointList.size() == 1) {
            auto& point = pointList.at(0);
            Node leaf{};
            leaf.value = point;

            nodes_.push_back(leaf);
            return nodes_.size() - 1;
        }

        // Select axis plane based on depth
        size_t axis = depth % K;

        // Determine axis median and split points list
        std::vector<Point> leftPoints;
        std::vector<Point> rightPoints;
        float median = medianByAxis(pointList, axis, leftPoints, rightPoints);

        // Construct node & recur
        Node node{};
        node.value = PartitionPlane{axis, median};
        node.leftChild = constructRecursive(leftPoints, depth + 1);
        node.rightChild = constructRecursive(rightPoints, depth + 1);

        // Return node
        nodes_.push_back(node);
        return nodes_.size() - 1;
    }

    float medianByAxis(const std::vector<Point>& pointList, 
                       size_t axis, 
                       std::vector<Point>& leftPoints, 
                       std::vector<Point>& rightPoints) {

        // Construct a list of all values along this axis
        std::vector<float> axisValues;
        axisValues.reserve(pointList.size());
        for (const auto& point : pointList) {
            axisValues.push_back(point.value.at(axis));
        }

        float median = KdTree::median(axisValues);

        // Split our points based on the axis median
        leftPoints.reserve(pointList.size() / 2);
        rightPoints.reserve(pointList.size() / 2);
        for (const auto& point : pointList) {
            if (point.value[axis] <= median && leftPoints.size() < pointList.size() / 2) {
                leftPoints.push_back(point);
            } else {
                rightPoints.push_back(point);
            }
        }

        return median;
    }


public:
    // This meidan-of-medians implementation follows the pseudocode for the algorithm from
    // wikipedia https://en.wikipedia.org/wiki/Median_of_medians
    static float median(std::vector<float>& list) {
        if (list.size() % 2 == 1) {
            return nthSmallest(list, list.size() / 2);
        } else {
            return 0.5 * (nthSmallest(list, list.size() / 2 - 1) 
                        + nthSmallest(list, list.size() / 2));
        }
    }

private:
    static float nthSmallest(std::vector<float>& list, size_t n) {
        size_t index = select(list, 0, list.size() - 1, n);
        return list.at(index);
    }

    static size_t select(std::vector<float>& list, size_t left, size_t right, size_t n) {
        size_t pivotIndex;
        while (left != right) {
            pivotIndex = pivot(list, left, right);
            pivotIndex = partition(list, left, right, pivotIndex, n);
            if (pivotIndex == n) {
                return n;
            } else if (n < pivotIndex) {
                right = pivotIndex - 1;
            } else {
                left = pivotIndex + 1;
            }
        }
        return left;
    }

    static size_t pivot(std::vector<float>& list, size_t left, size_t right) {
        if (right - left < 5) {
            return partition5(list, left, right);
        }

        for (size_t i = left; i <= right; i += 5) {
            size_t subRight = i + 4;
            if (subRight > right) {
                subRight = right;
            }
            size_t median5 = partition5(list, i, subRight);
            swap(list, median5, left + floor((i - left) / 5));
        }

        size_t mid = floor((right - left) / 10) + left;
        return select(list, left, left + floor((right - left) / 5), mid);
    }

    static size_t partition(std::vector<float>& list, size_t left, size_t right, size_t pivotIndex, size_t n) {
        float pivotValue = list[pivotIndex];
        swap(list, pivotIndex, right);
        size_t storeIndex = left;
        for (size_t i = left; i < right; ++i) {
            if (list[i] < pivotValue) {
                swap(list, storeIndex++, i);
            }
        }
        size_t storeIndexEq = storeIndex;
        for(size_t i = storeIndex; i < right; ++i) {
            if (list[i] == pivotValue) {
                swap(list, storeIndexEq++, i);
            }
        }
        swap(list, right, storeIndexEq);
        if (n < storeIndex) {
            return storeIndex;
        } else if (n <= storeIndexEq) {
            return n;
        }
        return storeIndexEq;
    }

    static size_t partition5(std::vector<float>& list, size_t left, size_t right) {
        size_t i = left + 1;
        while (i < right) {
            size_t j = i;
            while (j > left && list[j - 1] > list[j]) {
                swap(list, j - 1, j);
                j -= 1;
            }
            i += 1;
        }
        return left + (right - left) / 2;
    }

    static void swap(std::vector<float>& list, size_t i, size_t j) {
        float tmp = list[i];
        list[i] = list[j];
        list[j] = tmp;
    }


private:
    size_t K;
    size_t root_;
    std::vector<Node> nodes_;
};