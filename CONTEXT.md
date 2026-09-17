# Feature Probabilities

Turns an mzML file into a table of probabilities that a given molecule is the correct identity of a detected LC-MS feature, calibrated against MassSpecGym ground truth via SIRIUS/CSI:FingerID.

## Language

**Feature**:
A chromatographic peak with its MS2 spectrum, produced by SIRIUS's peak-picking step from an mzML run. The unit being annotated.
_Avoid_: peak, spectrum (ambiguous with a raw, unpicked scan)

**Structure candidate**:
A molecule SIRIUS/CSI:FingerID proposes as a possible identity for a feature, carrying a CSI:FingerID score. Returned by `get_structure_candidates` (PySirius `StructureCandidateFormula`) — this row does NOT itself carry ion mass.
_Avoid_: molecule (too generic on its own), hit

**Annotation**:
One feature–structure-candidate pair: a structure candidate's CSI:FingerID score joined to its feature's ion mass. Ion mass lives on the feature (PySirius `AlignedFeature.ionMass`), not on the structure-candidate row, so producing an Annotation requires joining the two by feature id — the row unit consumed by calibration.

**Ground truth set**:
MassSpecGym v1.5, the benchmark dataset of spectra with known true structures, used to fit the calibration score's KDE.

**Correct assignment**:
A structure candidate whose InChIKey first block (skeleton/connectivity layer only) matches the ground truth set's known true structure for that spectrum. Stereochemistry and protonation state are ignored, since MS/MS routinely can't resolve them.
_Avoid_: exact match, true positive (without qualifying "skeleton-level")

**Calibration score**:
The value a KDE fitted on the ground truth set's correct assignments returns for a feature–candidate pair's (ion mass, CSI:FingerID score). A density used to rank candidates within a feature, not a normalized posterior probability.
_Avoid_: probability, confidence (both imply a calibrated posterior, which this isn't)
