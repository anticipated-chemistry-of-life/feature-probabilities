# MSnLib replaces MassSpecGym as ground truth; calibration splits by Ionization mode, not instrument

MassSpecGym v1.5 (231,104 spectra) is positive-mode only, so the pipeline could not calibrate negative-mode field runs, and our field data includes 1,142 negative-mode mzMLs. MSnLib (`salutmoic/msnlib-ms2-parquet`, pinned revision) covers both Ionization modes. We keep its `ALL_ENERGIES` spectra (90,653) and drop two kinds: spectra that matched more than one library compound, and spectra whose adduct (`[M]+`, `[M-H2O]+`, `[M]-`) SIRIUS cannot detect. That leaves 88,677: 50,276 positive and 38,401 negative. Every MSnLib spectrum is Orbitrap (ID-X). The calibration KDEs are therefore split by Ionization mode only, `positive` and `negative`, with no per-instrument KDE and no pooled fallback. Instrument type still selects each SIRIUS run's instrument profile, for ground truth runs and field runs alike.

## Considered Options

- **Keep MassSpecGym for positive mode and add MSnLib for negative.** Rejected: the positive KDE would be fitted on a mix of Orbitrap and QTOF while the negative KDE is Orbitrap-only, so the two Ionization modes would be calibrated against different instrument populations.
- **Split by both Ionization mode and instrument type.** Not possible: there is no negative-mode QTOF ground truth.
- **Keep a pooled `__global__` KDE as a fallback.** Rejected: every field run declares its Ionization mode, and both modes always have a fitted KDE, so a fallback could only hide a missing or mislabelled mode.
- **Deduplicate spectra before SIRIUS on the dataset's own adduct label.** Rejected. It would save at most ~9% of SIRIUS time (80,383 unique first-block-InChIKey/adduct pairs among 88,677), but the fit-time deduplication keys on the adduct SIRIUS actually detected. Deduplicating earlier on the dataset label could keep the worse spectrum.

## Consequences

- QTOF field runs (e.g. the 930 Synapt XS mzMLs) run SIRIUS under the QTOF profile but are scored by Orbitrap-trained KDEs. This is accepted for now and is the reason to revisit if QTOF ground truth with both Ionization modes becomes available.
- The Ionization mode of a field run comes from the metadata table, not a CLI flag. A wrong value silently selects the wrong KDE.
- `collision_energy` is no longer written to ground truth MGF chunks: an `ALL_ENERGIES` spectrum merges several energies (e.g. `20.0,30.0,60.0`), which no single measured field spectrum carries.
- MassSpecGym ground truth runs and KDE pickles keyed by instrument are not migrated. The database is recreated and the ground truth regenerated, with `source_kind` renamed to `ground_truth_msnlib`.
- ADR 0001's cost figures (chunk `orbitrap_8`) describe MassSpecGym runs; its top-k decision is unchanged.
