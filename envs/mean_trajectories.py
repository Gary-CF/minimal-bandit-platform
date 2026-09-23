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

class LinearDriftMeans:
    def __init__(self,start_means,end_means,start_t,end_t):
        self.start_means=tuple(start_means)
        self.end_means=tuple(end_means)
        self.start_t=start_t
        self.end_t=end_t

        if isinstance(self.start_t,bool) or not isinstance(self.start_t,int):
            raise TypeError("start_t must be an integer")
        if isinstance(self.end_t,bool) or not isinstance(self.end_t,int):
            raise TypeError("end_t must be an integer")
        if not 1<=start_t<end_t:
            raise ValueError("start_t and end_t should hold 1<=start_t<end_t")
        if not self.start_means or not self.end_means:
            raise ValueError("start_means and end_means must not be empty")
        if len(self.start_means)!=len(self.end_means):
            raise ValueError("start_means and end_means should have the same number of arms")
        for item in self.start_means:
            if not isfinite(item) or not 0<=item<=1:
                raise ValueError("mean in start_means should be finite number with [0,1]")
        for item in self.end_means:
            if not isfinite(item) or not 0<=item<=1:
                raise ValueError("mean in end_means should be finite number with [0,1]")
            
        
    def at(self,t):
        if isinstance(t, bool) or not isinstance(t, int):
            raise TypeError("t must be an integer")
        if t < 1:
            raise ValueError("t must larger than or equal to 1")

        if t<=self.start_t:
            return self.start_means
        elif t>=self.end_t:
            return self.end_means
        else:
            alpha=(t-self.start_t)/(self.end_t-self.start_t)
            return tuple([(1-alpha)*self.start_means[i]+alpha*self.end_means[i]
                    for i in range(len(self.start_means))])