from envs.mean_trajectories import StationaryMeans,PiecewiseConstantMeans
import pytest


def test_tuple_and_seperate():
    means = [0.2, 0.8]
    trajectory = StationaryMeans(means)

    # 不同轮次，返回相同的均值元组。
    assert trajectory.at(1) == (0.2, 0.8)
    assert trajectory.at(100) == (0.2, 0.8)

    # 修改外部列表，不应改变已经创建的轨迹。
    means[0] = 0.9
    assert trajectory.at(1) == (0.2, 0.8)

    print("平稳查询与外部修改隔离：通过")


def test_rejects_empty_means():
    with pytest.raises(ValueError):
        StationaryMeans([])


def test_rejects_infinite_values():
    with pytest.raises(ValueError):
        StationaryMeans([2.0, 0.3])
    with pytest.raises(ValueError):
        StationaryMeans([0.2, float("inf")])
    with pytest.raises(ValueError):
        StationaryMeans([0.2, float("nan")])

def test_rejects_zero_round_and_invalid_type():
    trajectory = StationaryMeans([0.2,0.8])

    with pytest.raises(ValueError):
        trajectory.at(0)
    with pytest.raises(TypeError):
        trajectory.at(1.0)
    with pytest.raises(TypeError):
        trajectory.at(True)

def test_pieceswise_correctness():
    starts=[1,4,7]
    levels=[[0.2,0.8],[0.9,0.1],[0.4,0.6]]
    trajectory = PiecewiseConstantMeans(starts,levels)
    assert trajectory.at(1)==(0.2,0.8)
    assert trajectory.at(3)==(0.2,0.8)
    assert trajectory.at(4)==(0.9,0.1)
    assert trajectory.at(6)==(0.9,0.1)
    assert trajectory.at(7)==(0.4,0.6)
    assert trajectory.at(100)==(0.4,0.6)

def test_piecewise_copies_starts():
    starts=[1,4,7]
    levels=[[0.2,0.8],[0.9,0.1],[0.4,0.6]]
    trajectory = PiecewiseConstantMeans(starts,levels)
    starts[1]=5
    assert trajectory.starts == (1,4,7)

def test_piecewise_copies_levels():
    starts=[1,4,7]
    levels=[[0.2,0.8],[0.9,0.1],[0.4,0.6]]
    trajectory = PiecewiseConstantMeans(starts,levels)
    levels[1][0] = 0.3
    assert trajectory.levels[1][0] == 0.9

# ---------- 整体结构检查 ----------

def test_piecewise_init_rejects_empty_starts():
    """至少需要一段，不能用两个空序列创建轨迹。"""
    with pytest.raises(ValueError):
        PiecewiseConstantMeans(starts=[], levels=[])


def test_piecewise_init_rejects_empty_levels():
    """有段起点，却没有任何对应均值。"""
    with pytest.raises(ValueError):
        PiecewiseConstantMeans(starts=[1], levels=[])


def test_piecewise_init_rejects_mismatched_segment_counts():
    """段起点与均值行数必须一致：少一行、多一行都不行。"""
    with pytest.raises(ValueError):
        PiecewiseConstantMeans(
            starts=[1, 4],
            levels=[[0.2, 0.8]],
        )

    with pytest.raises(ValueError):
        PiecewiseConstantMeans(
            starts=[1],
            levels=[[0.2, 0.8], [0.9, 0.1]],
        )


# ---------- 段起点检查 ----------

def test_piecewise_init_rejects_boolean_starts():
    """布尔值不能作为起点，即使 True == 1。"""
    for starts in ([True, 4], [1, True], [1, False]):
        with pytest.raises(TypeError):
            PiecewiseConstantMeans(
                starts=starts,
                levels=[[0.2, 0.8], [0.9, 0.1]],
            )


def test_piecewise_init_rejects_noninteger_starts():
    """不能把浮点数、字符串或 None 当作整数轮次。"""
    for invalid_start in (4.0, "4", None):
        with pytest.raises(TypeError):
            PiecewiseConstantMeans(
                starts=[1, invalid_start],
                levels=[[0.2, 0.8], [0.9, 0.1]],
            )


def test_piecewise_init_requires_first_start_to_be_one():
    """第一段必须从第1轮开始，不能更早或更晚。"""
    for first_start in (-1, 0, 2):
        with pytest.raises(ValueError):
            PiecewiseConstantMeans(
                starts=[first_start, 4],
                levels=[[0.2, 0.8], [0.9, 0.1]],
            )


def test_piecewise_init_rejects_duplicate_starts():
    """起点必须严格递增，不能出现重复起点。"""
    with pytest.raises(ValueError):
        PiecewiseConstantMeans(
            starts=[1, 4, 4],
            levels=[
                [0.2, 0.8],
                [0.9, 0.1],
                [0.4, 0.6],
            ],
        )


def test_piecewise_init_rejects_unsorted_starts():
    """后一个起点不能小于前一个起点。"""
    with pytest.raises(ValueError):
        PiecewiseConstantMeans(
            starts=[1, 7, 4],
            levels=[
                [0.2, 0.8],
                [0.9, 0.1],
                [0.4, 0.6],
            ],
        )


# ---------- 臂数与均值检查 ----------

def test_piecewise_init_rejects_zero_arms():
    """即使只有一段，也至少要有一个臂。"""
    with pytest.raises(ValueError):
        PiecewiseConstantMeans(
            starts=[1],
            levels=[[]],
        )


def test_piecewise_init_rejects_inconsistent_arm_counts():
    """后续段不能少一个臂、多一个臂或变成空行。"""
    for second_level in ([0.9], [0.9, 0.1, 0.3], []):
        with pytest.raises(ValueError):
            PiecewiseConstantMeans(
                starts=[1, 4],
                levels=[[0.2, 0.8], second_level],
            )


def test_piecewise_init_rejects_out_of_range_means():
    """负概率和大于1的概率都应拒绝，不能只检查第一段。"""
    for invalid_mean in (-0.1, 1.2):
        with pytest.raises(ValueError):
            PiecewiseConstantMeans(
                starts=[1, 4],
                levels=[[0.2, 0.8], [0.9, invalid_mean]],
            )


def test_piecewise_init_rejects_nonfinite_means():
    """NaN、正无穷、负无穷都不是合法概率。"""
    for invalid_mean in (
        float("nan"),
        float("inf"),
        float("-inf"),
    ):
        with pytest.raises(ValueError):
            PiecewiseConstantMeans(
                starts=[1, 4],
                levels=[[0.2, 0.8], [0.9, invalid_mean]],
            )


# ---------- 合法边界：不能误伤正常输入 ----------

def test_piecewise_init_accepts_single_segment_and_endpoints():
    """一段轨迹合法；概率0和1也合法。"""
    trajectory = PiecewiseConstantMeans(
        starts=[1],
        levels=[[0.0, 1.0]],
    )

    assert trajectory.at(1) == (0.0, 1.0)
    assert trajectory.at(100) == (0.0, 1.0)


def test_piecewise_init_accepts_tuple_inputs():
    """元组输入合法；不同臂的均值不要求加起来等于1。"""
    trajectory = PiecewiseConstantMeans(
        starts=(1, 4),
        levels=(
            (0.8, 0.9),
            (0.1, 0.2),
        ),
    )

    assert trajectory.at(1) == (0.8, 0.9)
    assert trajectory.at(3) == (0.8, 0.9)
    assert trajectory.at(4) == (0.1, 0.2)
    assert trajectory.at(100) == (0.1, 0.2)

def test_piecewise_init_checks_first_start_type_before_value():
    """第一项类型错误时，应先报 TypeError，而不是起点取值错误。"""
    for first_start in (False, 0.5, "1", None):
        with pytest.raises(TypeError):
            PiecewiseConstantMeans(
                starts=[first_start, 4],
                levels=[[0.2, 0.8], [0.9, 0.1]],
            )
