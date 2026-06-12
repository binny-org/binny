.. |logo| image:: ../../_static/assets/logo.png
   :alt: logo
   :width: 32px

|logo| Calibration from mocks
=============================

Binny also implements a second workflow in which the parent
distribution is not specified purely by hand. Instead, the parameters of
a Smail model can be **calibrated from a mock catalog** containing true
redshifts and apparent magnitudes.

This is important because in realistic survey-design or forecasting
studies one often wants the parent :math:`n(z)` to reflect an underlying
simulated galaxy population rather than an arbitrary analytic choice.

The calibration tools implemented in Binny perform three related tasks:

1. infer the Smail shape parameters :math:`\alpha` and :math:`\beta`
   from a representative mock sample,
2. calibrate how the Smail redshift scale :math:`z_0` changes with
   limiting magnitude,
3. calibrate how the effective galaxy surface density
   :math:`n_{\rm gal}` changes with limiting magnitude.

This is exposed through
:meth:`binny.NZTomography.calibrate_smail_from_mock`.

The idea is straightforward. Given a mock catalog with true galaxy
redshifts and magnitudes, one considers a sequence of magnitude cuts
representing surveys of different depth. For each magnitude limit, one
selects the galaxies that would be observed and fits a smooth analytic
description to the resulting redshift distribution. One also counts how
many galaxies remain, converting this to an effective surface density.

The output is therefore not only a fitted parent :math:`n(z)`, but also
a set of depth-scaling relations that describe how the population shifts
as the survey becomes deeper.


Why depth calibration matters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A deeper survey typically includes fainter galaxies. In practice, this
usually changes the sample in two ways:

- the galaxy population extends to higher redshift,
- the total galaxy surface density increases.

In the Binny calibration workflow, these two effects are captured by
fitting

- a relation :math:`z_0(m_{\rm lim})`, describing how the characteristic
  redshift scale varies with limiting magnitude,
- and a relation :math:`n_{\rm gal}(m_{\rm lim})`, describing how the
  effective number density varies with limiting magnitude.

This provides a compact analytic summary of the mock catalog that can be
reused in later forecasting calculations. Instead of storing or
reprocessing the full mock each time, one can work with fitted scaling
relations that preserve the main statistical trends relevant for survey
depth.

This is one of the main reasons Smail remains useful: it is simple
enough to calibrate robustly, yet flexible enough to capture the broad
survey-level evolution of the parent galaxy population.


What the calibration does not do
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The calibration tools are designed to provide a smooth phenomenological
summary of a mock galaxy sample. They are not intended to reproduce
every detailed feature of a simulation or data set.

In particular, a fitted Smail model should not be interpreted as a
complete physical model of galaxy evolution or selection. Rather, it is
a compact approximation to the overall redshift structure of the sample.

Similarly, the fitted depth relations are empirical summaries of how the
mock population changes with limiting magnitude. They are useful for
forecasting and controlled survey studies, but they do not replace the
full information content of a realistic mock catalog.

This distinction is important for the theory documentation: Binny
implements a practical interface for **analytic parent-distribution
modeling**, not a full end-to-end simulation framework.


Connection to tomography
------------------------

Once a parent :math:`n(z)` has been specified or calibrated, Binny uses
it as the starting point for tomographic bin construction.

The parent distribution itself is not yet a tomographic object. It
contains the full galaxy population before splitting it into bins. The
later tomography step introduces bin edges, selection rules, and
possibly photometric uncertainty models that transform the parent
population into a set of tomographic bin curves.

It is therefore helpful to keep the conceptual separation clear:

- the **parent** :math:`n(z)` describes the overall galaxy population,
- the **tomographic bins** describe how that population is partitioned.

Many diagnostics of the tomographic bins, such as overlap, leakage, or
cross-bin coupling, depend not only on the binning scheme itself but
also on the structure of the underlying parent distribution. A broader
or more skewed parent :math:`n(z)` can lead to qualitatively different
bin behavior than a narrow or sharply bounded one.


Summary
-------

Binny implements a registry-based framework for parent redshift
distributions because tomographic workflows need a flexible but
well-defined description of the underlying galaxy population.

The Smail model is the default survey-like choice because it provides a
smooth, interpretable, and widely used phenomenological description of
magnitude-limited samples. Other models are included because they are
useful for flexible alternatives, multimodal structure, asymmetric
profiles, and controlled toy tests.

In addition, Binny supports calibration of Smail-based parent
distributions from mock catalogs, including depth-scaling relations for
the characteristic redshift scale and the galaxy number density. This
allows survey-motivated parent populations to be constructed from mocks
without requiring the full catalog to be propagated through every later
step of the workflow.

For executable usage examples, see the example pages on parent
:math:`n(z)` models and calibration from mocks.
