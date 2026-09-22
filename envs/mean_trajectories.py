from math import isfinite


class StationaryMeans:
    def __init__(self, means):
        values = tuple(means)
        if not values:
            raise ValueError("means should not be empty")
        for item in values:
            if not isfinite(item) or not 0 <= item <= 1:
                raise ValueError("means should not contain invalid value")
        self.means = values

    def at(self, t):
        if isinstance(t, bool) or not isinstance(t, int):
            raise TypeError("t must be an integer")
        if t < 1:
            raise ValueError("t must larger than or equal to 1")
        return self.means


class PiecewiseConstantMeans:
    def __init__(self, starts, levels):
        self.starts = tuple(starts)
        self.levels = tuple([tuple(item) for item in levels])

        if not len(self.starts) >= 1 or not len(self.starts) == len(levels):
            raise ValueError(
                "starts should not be empty and should hold the same length as levels"
            )
        for i in self.starts:
            if isinstance(i, bool) or not isinstance(i, int):
                raise TypeError("starts should only contain integers")
        if not self.starts[0] == 1:
            raise ValueError("starts should start from 1")
        for i in range(1, len(self.starts)):
            if not self.starts[i] > self.starts[i - 1]:
                raise ValueError("starts should increase in order strictly")
        K = len(self.levels[0])
        if K == 0:
            raise ValueError("should exit one arm at least")
        for item in self.levels:
            if len(item) != K:
                raise ValueError("the num of arms should not change")
            for j in item:
                if not isfinite(j) or not 0 <= j <= 1:
                    raise ValueError("means should not contain invalid value")

    def at(self, t):
        if isinstance(t, bool) or not isinstance(t, int):
            raise TypeError("t must be an integer")
        if t < 1:
            raise ValueError("t must larger than or equal to 1")
        j = 0
        while True:
            if j == len(self.starts):
                break
            if self.starts[j] <= t:
                j += 1
                continue
            break
        return self.levels[j - 1]
