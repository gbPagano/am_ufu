import math
import random

import matplotlib.pyplot as plt
import numpy as np


class KPoint:
    def __init__(
        self,
        interval_x: tuple[float, float],
        interval_y: tuple[float, float],
    ):
        x = random.uniform(interval_x[0], interval_x[1])
        y = random.uniform(interval_y[0], interval_y[1])
        self.k_center = [x, y]
        self.points = []
        self.old_points = []

    def update_center(self):
        if self.old_points == self.points:
            self.points = []
            return False

        if self.points:
            self.k_center = np.mean(self.points, axis=0)
            self.old_points = self.points.copy()

        self.points = []
        return True

    def __repr__(self):
        return f"{self.k_center}"


def main():
    with open("data.txt") as f:
        lines = f.readlines()
        points = [list(map(float, data.strip().split())) for data in lines]

    x, y = zip(*points)

    results = k_means(
        k=4,
        points=points,
        interval_x=(min(x), max(x)),
        interval_y=(min(y), max(y)),
    )

    plt.scatter(x, y, color="red")
    print(results)
    for item in results:
        x, y = item.k_center
        a, b = zip(*item.old_points)
        plt.scatter(a, b)
        plt.scatter(x, y)

    plt.show()


def k_means(
    k: int,
    points: list[list[float]],
    interval_x: tuple[float, float],
    interval_y: tuple[float, float],
):
    k_points = []

    for _ in range(k):
        k_point = KPoint(interval_x, interval_y)
        k_points.append(k_point)

    results = [True]
    while any(results):
        for point in points:
            distances = [
                (math.dist(point, k_point.k_center), idx)
                for (idx, k_point) in enumerate(k_points)
            ]
            idx = min(distances)[1]
            k_points[idx].points.append(point)

        results = [k_point.update_center() for k_point in k_points]

    return [k_point for k_point in k_points]


if __name__ == "__main__":
    main()
