# Feature Probabilities

Turns an mzML file into a table of probabilities that a given molecule is the correct identity of a detected LC-MS feature, calibrated against MassSpecGym ground truth via SIRIUS/CSI:FingerID.

## Language

**Feature**:
One MS2 spectrum treated as an analytical unit by SIRIUS — either detected via SIRIUS's peak-picking from an mzML run, or supplied already-picked (e.g. one MassSpecGym spectrum, imported without peak-picking). The unit being annotated.
_Avoid_: peak, spectrum (ambiguous with a raw, unpicked scan)

**Structure candidate**:
A molecule SIRIUS/CSI:FingerID proposes as a possible identity for a feature, carrying a CSI:FingerID score. Returned by `get_structure_candidates` (PySirius `StructureCandidateFormula`) — this row does NOT itself carry ion mass. Per-feature: the same underlying Molecule can appear as a structure candidate on many features.
_Avoid_: molecule (see the distinct **Molecule** entry below — the two are not interchangeable), hit

**Molecule**:
The globally deduplicated, skeleton-level chemical identity a structure candidate resolves to: a first-block (14-character) InChIKey plus a canonical, stereochemistry-stripped 2D SMILES. Stored once regardless of how many features/structure candidates reference it. Both SIRIUS's candidate search and MassSpecGym's ground truth already operate at this granularity — neither ever exposes a longer, stereo-resolved key — so a Molecule never distinguishes stereoisomers of the same skeleton.
_Avoid_: structure candidate (that's the per-feature, scored occurrence; a Molecule is the shared, unscored identity)

**Annotation**:
One feature–structure-candidate pair: a structure candidate's CSI:FingerID score joined to its feature's ion mass. Ion mass lives on the feature (PySirius `AlignedFeature.ionMass`), not on the structure-candidate row, so producing an Annotation requires joining the two by feature id — the row unit consumed by calibration.

**Extract**:
The physical sample (a species plus, e.g., organ) that one or more SIRIUS runs are acquired from. Purely biological/sample provenance — acquisition-specific detail (ionization mode, instrument type) lives on the SIRIUS run, not here.

**SIRIUS run**:
One execution of SIRIUS against one input spectra file (an mzML or an MGF chunk) with one fixed parameter set, identified by the rerun-avoidance caching key. A **field run** processes a real mzML tied to an Extract; a **ground truth run** processes a MassSpecGym chunk and has no Extract. Reprocessing the same input file with different parameters produces a new, coexisting SIRIUS run — never overwrites a prior one.
_Avoid_: job (SIRIUS's own API term for the async task that executes a run — a run is the durable record, a job is how it got produced)

**Ground truth set**:
MassSpecGym v1.5, the benchmark dataset of spectra with known true structures, used to fit the calibration score's KDE.

**Correct assignment**:
A structure candidate whose InChIKey first block (skeleton/connectivity layer only) matches the ground truth set's known true structure for that spectrum. Stereochemistry and protonation state are ignored, since MS/MS routinely can't resolve them.
_Avoid_: exact match, true positive (without qualifying "skeleton-level")

**Calibration score**:
The value a KDE fitted on the ground truth set's correct assignments returns for a feature–candidate pair's (ion mass, CSI:FingerID score). A density used to rank candidates within a feature, not a normalized posterior probability.
_Avoid_: probability, confidence (both imply a calibrated posterior, which this isn't)

**Feature-probability table**:
`annotate`'s final exported table (one row per feature–structure-candidate pair with its calibration score) — the deliverable the whole pipeline produces for a batch of new mzML files.
