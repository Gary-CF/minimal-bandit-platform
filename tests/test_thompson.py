import numpy as np

from algorithms.thompson_sampling import ThompsonSampling

def test_thompson_initializes_uniform_beta_priors()->None:
    algorithm=ThompsonSampling(
        num_arms=3,
        rng=np.random.default_rng(0),
    )

    assert np.array_equal(
        algorithm.alpha,
        np.array([1.0,1.0,1.0])
    )

    assert np.array_equal(
        algorithm.beta,
        np.array([1.0,1.0,1.0])
    )

def test_thompson_failure_updates_beta_only()->None:
    algorithm=ThompsonSampling(
        num_arms=3,
        rng=np.random.default_rng(0),
    )

    algorithm.update(
        action=2,
        reward=0,
    )

    assert np.array_equal(
        algorithm.alpha,
        np.array([1.0,1.0,1.0])
    )

    assert np.array_equal(
        algorithm.beta,
        np.array([1.0,1.0,2.0])
    )

def test_thompson_posterior_matches_observed_outcomes() -> None:
    algorithm = ThompsonSampling(
        num_arms=3,
        rng=np.random.default_rng(0),
    )

    observations = [
        (0, 1),
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 0),
        (2, 1),
    ]

    for action, reward in observations:
        algorithm.update(
            action=action,
            reward=reward,
        )

    assert np.array_equal(
        algorithm.alpha,
        np.array([3.0, 1.0, 2.0]),
    )

    assert np.array_equal(
        algorithm.beta,
        np.array([2.0, 3.0, 1.0]),
    )

def test_thompson_posterior_invariants() -> None:
    algorithm = ThompsonSampling(
        num_arms=3,
        rng=np.random.default_rng(0),
    )

    observations = [
        (0, 1),
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 0),
        (2, 1),
    ]

    action_counts = np.zeros(3, dtype=int)
    success_counts = np.zeros(3, dtype=int)

    for action, reward in observations:
        algorithm.update(
            action=action,
            reward=reward,
        )

        action_counts[action] += 1
        success_counts[action] += reward

    failure_counts = action_counts - success_counts

    assert np.array_equal(
        algorithm.alpha - 1,
        success_counts,
    )

    assert np.array_equal(
        algorithm.beta - 1,
        failure_counts,
    )

    assert np.array_equal(
        algorithm.alpha + algorithm.beta - 2,
        action_counts,
    )

def test_thompson_same_seed_reproduces_actions() -> None:
    algorithm_one = ThompsonSampling(
        num_arms=3,
        rng=np.random.default_rng(42),
    )

    algorithm_two = ThompsonSampling(
        num_arms=3,
        rng=np.random.default_rng(42),
    )

    rewards = [1, 0, 1, 1, 0, 0, 1, 0]

    actions_one = []
    actions_two = []

    for reward in rewards:
        action_one = algorithm_one.select_action()
        action_two = algorithm_two.select_action()

        actions_one.append(action_one)
        actions_two.append(action_two)

        algorithm_one.update(action_one, reward)
        algorithm_two.update(action_two, reward)

    assert actions_one == actions_two
    assert np.array_equal(
        algorithm_one.alpha,
        algorithm_two.alpha,
    )
    assert np.array_equal(
        algorithm_one.beta,
        algorithm_two.beta,
    )