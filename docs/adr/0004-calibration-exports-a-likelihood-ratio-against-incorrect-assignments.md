# Calibration exports a likelihood ratio against incorrect assignments

Calibration used to export only the correct density g1 (then called the calibration score), and the MRF divided each candidate's g1 by its Feature's best row (MRF ADR-0012). That is exact only if the incorrect density g0 is flat in CSI:FingerID score, and it is not. In the ground truth snapshot, correct assignments have a median score of −215 and incorrect ones −317. If the true molecule sits among a Feature's candidates like any other row, and rows are independent given the truth, each candidate's posterior weight is its prior times g1/g0, while ∏ g0 over the Feature cancels. The unknown molecule's weight is then a constant prior odds, not a multiple of the Feature's best row. So calibration now fits g0 as well, and the Feature-probability table carries both densities, as natural logs, in the columns `log_correct_density` (ln g1) and `log_incorrect_density` (ln g0). The MRF side adopts these column names.

## Considered Options

- **Export one `log_likelihood_ratio` column.** Rejected: it hides rows where both densities are in the far tail, and the MRF loader can compute the difference itself. Logs (`logpdf`) keep tail rows finite where raw densities would give 0/0.
- **Fit g0 on features whose truth is listed and features whose truth isn't, separately.** Not chosen: g0 is pooled over every incorrect assignment of the deduplicated ground truth Features. `fit-kde` prints both subpopulation fits per Ionization mode, so a difference shows up. Acting on a difference would be an MRF model change.
- **Fit g0 on all incorrect rows with `gaussian_kde`.** Rejected: about 4 M training rows evaluated at 1.34 M field rows is about 5e12 kernel evaluations. g0 is a `gaussian_kde` (Scott's rule, same family as g1) on a seeded subsample of at most 20,000 incorrect rows per Ionization mode. The seed and subsample size are recorded on the KDE model.
- **Grid/FFT KDE (e.g. KDEpy).** Rejected: it adds a dependency and makes the two densities behave differently at the edges of the grid.
- **Classifier density ratio (logistic regression, GBM).** Rejected: it gives ln LR directly and loses the two densities the export carries.

## Consequences

- The export no longer has a `calibration_score` column. The MRF loader breaks until it reads the two new columns.
- Old single-KDE pickles are rejected, not converted. Migration drops and recreates only the derived KDE-model and calibration tables, then refits and re-runs `annotate` on SIRIUS cache hits, so SIRIUS is not rerun.
- Assumptions that may weaken the ratio: the Top-k cut (ADR 0001) selects rows on score; similar structures get correlated scores; and g0 may differ between Features whose truth is listed and Features whose truth isn't.
