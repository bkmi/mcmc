import numpy as np
import scipy.stats


# Acceptance rate at every step
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


def mala_step(state, distribution, tau=1, v=False, log=False):
    state = np.asarray(state)
    proposed_state = (state +
                      tau * distribution.grad_logpdf(state) +
                      np.sqrt(2 * tau) * scipy.stats.multivariate_normal([0, 0], np.eye(2)).rvs(1))

    if not log:
        alpha = (distribution.pdf(proposed_state) * conditional_prob(proposed_state, state, distribution) /
                 (distribution.pdf(state) * conditional_prob(state, proposed_state, distribution)))

        acceptance_prob = min([1, alpha])
        if np.random.rand() < acceptance_prob:
            accepted = True
        else:
            accepted = False

        if v:
            print('state:', state,
                  'prop_state:', proposed_state,
                  'alpha:', alpha,
                  'pdf_prop:', distribution.pdf(proposed_state),
                  'cond_to_prop:', conditional_prob(proposed_state, state, distribution),
                  'pdf_state:', distribution.pdf(state),
                  'cond_to_state:', conditional_prob(state, proposed_state, distribution))

    else:
        alpha = distribution.logpdf(proposed_state) + \
                log_conditional_prob(proposed_state, state, distribution) - \
                distribution.logpdf(state) - \
                log_conditional_prob(state, proposed_state, distribution)

        acceptance_prob = min([np.log(1), alpha])
        if np.log(np.random.rand()) < acceptance_prob:
            accepted = True
        else:
            accepted = False

        if v:
            print('state:', state,
                  'prop_state:', proposed_state,
                  'alpha:', alpha,
                  'pdf_prop:', distribution.logpdf(proposed_state),
                  'cond_to_prop:', log_conditional_prob(proposed_state, state, distribution),
                  'pdf_state:', distribution.logpdf(state),
                  'cond_to_state:', log_conditional_prob(state, proposed_state, distribution))

    if accepted:
        return proposed_state
    else:
        return state
