import numpy as np
import scipy.stats


def lagk_ac(sequence, k, variance=None, v=False):
    if k < 0:
        raise ValueError('k must be equal to or greater than zero.')

    if k == 0:
        lag, seq = sequence, sequence
    else:
        lag, seq = sequence[:-k], sequence[k:]
    m = np.mean(sequence, axis=0)

    nom = np.sum(np.dot(lag[..., 0] - m[..., 0], (seq[..., 0] - m[..., 0]).T))
    if not variance:
        denom = np.sum(np.dot(sequence[..., 0] - m[..., 0], (sequence[..., 0] - m[..., 0]).T))
    else:
        denom = variance

    if v:
        print(nom, denom)

    return np.divide(nom, denom)


def acceptance_rate_per_step(states):
    states = np.asarray(states)
    steps = 0
    accepted = 0
    acc_rate = []
    for i in range(1, len(states)):
        steps += 1
        if (np.isclose(states[i, 0], states[i-1, 0], rtol=1e-10, atol=1e-15) and
                np.isclose(states[i, 1], states[i-1, 1], rtol=1e-10, atol=1e-15)):
            accepted += 1
        acc_rate.append(accepted/steps)
    return acc_rate


def conditional_prob(from_state, to_state, distribution, tau=1):
    to_state = np.asarray(to_state)
    from_state = np.asarray(from_state)
    return np.exp((-4 * tau) ** (-1) *
                  np.linalg.norm(to_state - from_state - tau *
                                 distribution.grad_logpdf(from_state)) ** 2)


def log_conditional_prob(from_state, to_state, distribution, tau=1):
    to_state = np.asarray(to_state)
    from_state = np.asarray(from_state)
    return (-4 * tau) ** (-1) * np.linalg.norm(to_state - from_state - tau * distribution.grad_logpdf(from_state)) ** 2


def mala_step(state, target_distribution, tau=1, v=False, log=False):
    state = np.asarray(state)
    proposed_state = (state +
                      tau * target_distribution.grad_logpdf(state) +
                      np.sqrt(2 * tau) * scipy.stats.multivariate_normal([0, 0], np.eye(2)).rvs(1))

    if not log:
        alpha = (target_distribution.pdf(proposed_state) * conditional_prob(proposed_state, state, target_distribution) / (
                target_distribution.pdf(state) * conditional_prob(state, proposed_state, target_distribution)))

        acceptance_prob = min([1, alpha])
        if np.random.rand() < acceptance_prob:
            accepted = True
        else:
            accepted = False

        if v:
            print('state:', state,
                  'prop_state:', proposed_state,
                  'alpha:', alpha,
                  'pdf_prop:', target_distribution.pdf(proposed_state),
                  'cond_to_prop:', conditional_prob(proposed_state, state, target_distribution),
                  'pdf_state:', target_distribution.pdf(state),
                  'cond_to_state:', conditional_prob(state, proposed_state, target_distribution))

    else:
        alpha = target_distribution.logpdf(proposed_state) + \
                log_conditional_prob(proposed_state, state, target_distribution) - \
                target_distribution.logpdf(state) - \
                log_conditional_prob(state, proposed_state, target_distribution)

        acceptance_prob = min([np.log(1), alpha])
        if np.log(np.random.rand()) < acceptance_prob:
            accepted = True
        else:
            accepted = False

        if v:
            print('state:', state,
                  'prop_state:', proposed_state,
                  'alpha:', alpha,
                  'pdf_prop:', target_distribution.logpdf(proposed_state),
                  'cond_to_prop:', log_conditional_prob(proposed_state, state, target_distribution),
                  'pdf_state:', target_distribution.logpdf(state),
                  'cond_to_state:', log_conditional_prob(state, proposed_state, target_distribution))

    if accepted:
        return proposed_state
    else:
        return state
