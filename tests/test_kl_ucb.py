import numpy as np
import pytest

from algorithms.kl_ucb import (
    KLUCB,
    binary_kl,
    kl_ucb_bound,
)


# ============================================================
# 1. binary_kl
# ============================================================

@pytest.mark.parametrize(
    "p",
    [
        0.0,
        0.1,
        0.5,
        0.9,
        1.0,
    ],
)
def test_binary_kl_same_distribution_is_zero(p):
    """
    Bernoulli 分布与自身的 KL divergence 应当为 0。
    """
    assert binary_kl(p, p) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "p,q",
    [
        (0.1, 0.2),
        (0.2, 0.8),
        (0.5, 0.7),
        (0.8, 0.9),
        (0.9, 0.4),
    ],
)
def test_binary_kl_is_nonnegative(p, q):
    """
    KL divergence 永远非负。
    """
    value = binary_kl(p, q)

    assert value >= 0.0
    assert np.isfinite(value)


def test_binary_kl_handles_p_zero():
    """
    p = 0 时：

        d(0, q) = -log(1-q)

    必须避免直接计算 0 * log(0)。
    """
    q = 0.3

    expected = -np.log(1.0 - q)

    assert binary_kl(0.0, q) == pytest.approx(expected)


def test_binary_kl_handles_p_one():
    """
    p = 1 时：

        d(1, q) = -log(q)
    """
    q = 0.7

    expected = -np.log(q)

    assert binary_kl(1.0, q) == pytest.approx(expected)


@pytest.mark.parametrize(
    "p,q",
    [
        (-0.1, 0.5),
        (1.1, 0.5),
        (0.5, -0.1),
        (0.5, 1.1),
    ],
)
def test_binary_kl_rejects_values_outside_unit_interval(p, q):
    """
    Bernoulli 参数必须位于 [0, 1]。
    """
    with pytest.raises(ValueError):
        binary_kl(p, q)


# ============================================================
# 2. kl_ucb_bound
# ============================================================

def test_kl_ucb_bound_stays_between_p_and_one():
    """
    KL-UCB 上界的搜索区间就是 [p, 1]。
    """
    p = 0.4
    budget = 0.1

    q = kl_ucb_bound(p, budget)

    assert p <= q <= 1.0


def test_kl_ucb_bound_satisfies_budget():
    """
    返回的 q 必须仍然位于 KL 可行域中：

        d(p, q) <= budget
    """
    p = 0.35
    budget = 0.2

    q = kl_ucb_bound(p, budget)

    assert binary_kl(p, q) <= budget + 1e-10


def test_kl_ucb_bound_is_close_to_boundary():
    """
    二分搜索的目标不是随便找一个合法 q，
    而是找到最大的合法 q。

    因此最终 KL divergence 应非常接近 budget。
    """
    p = 0.4
    budget = 0.1

    q = kl_ucb_bound(p, budget)

    assert binary_kl(p, q) == pytest.approx(
        budget,
        abs=1e-8,
    )


def test_kl_ucb_bound_does_not_decrease_when_budget_increases():
    """
    budget 越大，允许算法越乐观，因此 KL upper bound
    不能反而下降。
    """
    p = 0.4

    q_small = kl_ucb_bound(
        p,
        budget=0.01,
    )

    q_medium = kl_ucb_bound(
        p,
        budget=0.1,
    )

    q_large = kl_ucb_bound(
        p,
        budget=0.5,
    )

    assert q_small <= q_medium <= q_large


def test_kl_ucb_bound_zero_budget_returns_p():
    """
    如果完全不允许 KL 偏离：

        budget = 0

    最大合法 q 就应该是 p 本身。
    """
    p = 0.37

    q = kl_ucb_bound(
        p,
        budget=0.0,
    )

    assert q == pytest.approx(
        p,
        abs=1e-10,
    )


def test_kl_ucb_bound_rejects_negative_budget():
    with pytest.raises(ValueError):
        kl_ucb_bound(
            p=0.5,
            budget=-0.1,
        )


# ============================================================
# 3. KLUCB
# ============================================================

def test_klucb_rejects_invalid_num_arms():
    with pytest.raises(ValueError):
        KLUCB(num_arms=0)

    with pytest.raises(ValueError):
        KLUCB(num_arms=-1)

    with pytest.raises(ValueError):
        KLUCB(num_arms=True)


@pytest.mark.parametrize(
    "c",
    [
        -0.1,
        -1.0,
        np.inf,
        np.nan,
    ],
)
def test_klucb_rejects_invalid_c(c):
    with pytest.raises(ValueError):
        KLUCB(
            num_arms=3,
            c=c,
        )


def test_klucb_initializes_arms_in_order():
    """
    初始化阶段必须：

        0 -> 1 -> 2 -> ...

    注意必须 select 后 update，
    因为算法依赖 counts 判断已经完成多少步。
    """
    algorithm = KLUCB(
        num_arms=3,
    )

    actions = []

    for reward in [1.0, 0.0, 1.0]:
        action = algorithm.select_action()
        actions.append(action)
        algorithm.update(action, reward)

    assert actions == [0, 1, 2]


def test_klucb_update_tracks_statistics_correctly():
    """
    检查 count、reward sum 和经验均值，
    不依赖随机实验。
    """
    algorithm = KLUCB(
        num_arms=2,
    )

    algorithm.update(
        action=0,
        reward=1.0,
    )

    algorithm.update(
        action=0,
        reward=0.0,
    )

    algorithm.update(
        action=0,
        reward=1.0,
    )

    assert algorithm.counts[0] == 3
    assert algorithm.reward_sums[0] == pytest.approx(2.0)
    assert algorithm.estimated_mean[0] == pytest.approx(2.0 / 3.0)

    assert algorithm.counts[1] == 0
    assert algorithm.reward_sums[1] == pytest.approx(0.0)
    assert algorithm.estimated_mean[1] == pytest.approx(0.0)


def test_klucb_select_action_matches_manual_bounds():
    """
    手动构造一个已经初始化完毕的内部状态。

    然后独立计算每个 arm 的 KL-UCB，
    检查 select_action 是否真的选择最大 bound。
    """
    algorithm = KLUCB(
        num_arms=3,
        c=3.0,
    )

    algorithm.counts[:] = np.array(
        [10, 5, 20],
        dtype=int,
    )

    algorithm.estimated_mean[:] = np.array(
        [0.50, 0.45, 0.60],
        dtype=float,
    )

    algorithm.reward_sums[:] = (
        algorithm.counts
        * algorithm.estimated_mean
    )

    current_round = int(
        algorithm.counts.sum()
    ) + 1

    numerator = (
        np.log(current_round)
        + algorithm.c
        * np.log(
            max(
                np.log(current_round),
                1.0,
            )
        )
    )

    budgets = (
        numerator
        / algorithm.counts
    )

    expected_bounds = np.array(
        [
            kl_ucb_bound(
                algorithm.estimated_mean[arm],
                budgets[arm],
            )
            for arm in range(
                algorithm.num_arms
            )
        ]
    )

    expected_action = int(
        np.argmax(expected_bounds)
    )

    actual_action = (
        algorithm.select_action()
    )

    assert actual_action == expected_action


def test_klucb_update_rejects_invalid_action():
    algorithm = KLUCB(
        num_arms=3,
    )

    with pytest.raises(IndexError):
        algorithm.update(
            action=-1,
            reward=1.0,
        )

    with pytest.raises(IndexError):
        algorithm.update(
            action=3,
            reward=1.0,
        )