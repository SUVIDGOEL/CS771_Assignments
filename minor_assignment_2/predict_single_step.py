r"""
CS771 -- Minor Assignment 2
Single-step joint MAP inference for Gaussian generative classifiers.

Suvid Goel (231065)

The lecture code performs inference in two sequential steps:

    Step 1:  y_hat   = argmax_c  P[x^o | y=c] . P[y=c]
    Step 2:  x_m_hat = argmax_v  P[v | x^o, y=y_hat]

This module replaces that with the *joint* optimisation

    {y_hat, x_m_hat} = argmax_{c,v}  P[y=c, x^m=v | x^o]

Carrying out the inner maximisation over v analytically (the mode of a
Gaussian is its mean, with peak density (2.pi)^{-d_m/2} |Sigma_bar^c|^{-1/2})
shows the joint score differs from the two-step score by an extra term:

    score(c) = log P[x^o | y=c] + log P[y=c]  -  0.5 * log|Sigma_bar^c|
               \_________ two-step score _________/   \_ new penalty _/

where Sigma_bar^c is the Schur complement -- the covariance of the missing
pixels conditioned on the observed ones:

    Sigma_bar^c = Sigma^{mm,c} - Sigma^{mo,c} (Sigma^{oo,c})^{-1} Sigma^{om,c}

So the two methods are NOT equivalent. The penalty favours classes that
predict the missing region confidently and penalises classes whose
conditional distribution over the missing pixels is diffuse. They agree
only when |Sigma_bar^c| is identical for every class.

Conventions follow the lecture code: numpy.linalg.pinv is used for the
rank-deficient covariance sub-blocks (border pixels have zero variance
within a class), and slogdet's sign is checked before using the
log-determinant.
"""

import numpy as np
from numpy import linalg as lin

# predictClassScores and truncatePixels are the unmodified lecture helpers
# from lecture_code/generative_classification.ipynb; neither needed changing.


def predictSingleStep(X, model, C, mask):
    """
    Solve {y_hat, x_m_hat} = argmax_{c,v} P[y=c, x^m=v | x^o] in one pass.

    For each fixed c the optimal v is the conditional mean mu_bar^c, known
    analytically. We precompute mu_bar^c for ALL classes inside the scoring
    loop and select the one belonging to the winning class afterwards --
    mirroring the inner-then-outer maximisation of the derivation.

    Parameters
    ----------
    X     : (n, d) test matrix
    model : iterable of (mu, Sigma, c, p) per class
    C     : number of classes
    mask  : (d,) boolean array, True where a pixel is observed

    Returns
    -------
    yPred  : (n,) predicted labels
    xm_hat : (n, d_m) estimates of the missing pixels
    """
    obs, mis = mask, ~mask
    n, d_m = X.shape[0], mis.sum()

    classScores = np.zeros((n, C))
    all_xm = np.zeros((n, C, d_m))

    for mu, Sigma, c, p in model:
        Soo = Sigma[np.ix_(obs, obs)]
        Smo = Sigma[np.ix_(mis, obs)]
        Smm = Sigma[np.ix_(mis, mis)]

        # A^c computed once, reused for both the Schur complement and mu_bar^c
        A = Smo @ lin.pinv(Soo)

        # Schur complement = conditional covariance of x^m given x^o
        Sigma_bar = Smm - A @ Smo.T
        sign, logdet = lin.slogdet(Sigma_bar)
        logdet_c = logdet if sign > 0 else 0.0

        # Single-step score: two-step score minus 0.5 * log|Sigma_bar^c|
        classScores[:, c] = (predictClassScores(X, mu, Sigma, p, obs)
                             - 0.5 * logdet_c)

        # Precompute mu_bar^c for all test points simultaneously
        all_xm[:, c, :] = mu[mis] + (X[:, obs] - mu[obs]) @ A.T

    # Outer maximisation over c
    yPred = np.argmax(classScores, axis=1)

    # Pick the precomputed mu_bar^c belonging to the winning class
    xm_hat = all_xm[np.arange(n), yPred, :]

    return yPred, xm_hat


def reconstructFromXm(X, xm_hat, mask):
    """
    Assemble the full image from the observed pixels and x_m_hat.
    Pure bookkeeping -- identical for both the two-step and single-step methods.
    """
    XRecon = X.copy()
    XRecon[:, ~mask] = xm_hat
    return truncatePixels(XRecon)
