# EEG network reorganization

Baseline EEG in transitioners, non-transitioners and healthy controls.
Run the numbered scripts in order. Statistics and plots are separate.

## Setup

Install `requirements.txt` in a dedicated Python environment. Set
`EEG_PROJECT_ROOT` to your EEG project folder, or edit `basepath` in
`config.py`. By default, the repository root is used as the data project root.
Set `SCHAEFER_CENTROIDS` to the Schaefer 100-parcel coordinate CSV.

Run from the repository root so the scripts can import `config.py` and
`plotting_utils.py`:

```powershell
python -m pip install -r requirements.txt
$env:EEG_PROJECT_ROOT = 'D:/path/to/EEG_projects'
$env:SCHAEFER_CENTROIDS = 'D:/path/to/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
python -m Scripts.01_preprocessing
```

Use the same `python -m Scripts.<script_name>` command for the subsequent scripts,
without the `.py` extension. For example, `python -m Scripts.16_nbs_statistics`
runs NBS and `python -m Scripts.17_nbs_plots` plots the saved NBS results.
The repository contains `Scripts/`, `config.py`, `plotting_utils.py`, and
`requirements.txt`; the data directory below may be stored separately.

The scripts create their output subfolders as needed:

```text
EEG_projects/
  RAW/
    raw_data_ncbp/
    raw_data_ncbpmc/
  derivatives/
    PREPROCESS_CONN/
      preprocess_ncbp/
      preprocess_ncbphc/
    CONN_NETW/
      conn_ncbp/sub-*/
      conn_ncbphc/sub-*/
    GROUP_CONN_NETW/
      preprocessing_qc/
      SPECTRAL/
        PSD_HEATMAPS/
        FOOOF_BOXPLOTS/
        FOOOF_GLASS_BRAIN/
      whole_brain_topology/FIGURES/
      within_network_topology/FIGURES/
      within_network_connectivity/FIGURES/
      between_network_connectivity/FIGURES/
      pain_trajectory/
      nbs/
        <comparison>/nbs_t_<threshold>/
        <comparison>/figures_nbs_t_<threshold>/
```

Collection, statistics and plotting use this same structure; there are no
density-suffixed output folders. Raw EEG, demographic files, preprocessing
inputs and the atlas coordinate file must be supplied by the researcher.
Participant data and analysis tables must not be committed to GitHub.
`output_plots/` contains the selected SVG figure exports matching the current
plotting scripts, including NBS; obsolete plot types such as circos are omitted.
PNG duplicates and CSV tables are excluded from this published figure folder.
The demographic CSV must contain the verified six-month outcome classification
in `chroncity_class_6m` (0: non-transitioner, 1: transitioner). The public code
does not override outcomes for individual participant IDs. Pain rainclouds use
`pain` and `pain_6m` on the 0–10 scale from the patient demographic CSV.

Optional overrides: `EEG_RESULTS_DIR` changes the group-results root.
`EEG_PLOT_RESULTS_DIR` defaults to that same root; `EEG_FIGURES_DIR` defaults
to the plotting-input root. These overrides are not needed for the layout above.

## Scripts

| File | Purpose |
| --- | --- |
| 01_preprocessing.py | Individual preprocessing and QC collection |
| 02_preprocessing_statistics.py | Three-group QC statistics; no figures |
| 03_subject_features.py | Source PSD, FOOOF, connectivity, binary topology and collection |
| 04_psd_statistics.py | Age/sex-adjusted PSD cluster inference |
| 05_psd_plots.py | PSD group and difference heatmaps |
| 06_fooof_statistics.py | FOOOF GEE and parcel follow-up estimates |
| 07_fooof_plots.py | FOOOF boxplots and brain maps |
| 08_whole_brain_topology_statistics.py | Whole-brain clustering and efficiency GEE |
| 09_whole_brain_topology_plots.py | Whole-brain violins, across and within bands |
| 10_network_topology_statistics.py | Within-network clustering and efficiency GEE |
| 11_network_topology_plots.py | Six-network boxplots/rainclouds and significance-masked glass brains |
| 12_within_network_connectivity_statistics.py | Within-network mean-connectivity GEE |
| 13_within_network_plots.py | Boxplots, rainclouds, parcel heatmaps and connectomes |
| 14_between_network_connectivity_statistics.py | Between-network connectivity GEE |
| 15_between_network_plots.py | Boxplots, vertical adjusted pointplots and heatmaps |
| 16_nbs_statistics.py | NBS at thresholds 3.0, 3.5 and 4.0 |
| 17_nbs_plots.py | NBS heatmaps and connectomes |
| 18_pain_trajectory_plots.py | Paired baseline/six-month pain rainclouds by transition group |

`config.py` defines paths, labels, colors and analysis settings.
`plotting_utils.py` contains shared figure styling, not statistical models.

Script 03 creates spectral collection files ending in `_all_groups.csv`.
Scripts 04–07 use these names consistently. Script 06 supplies the parcel-level
adjusted CSVs required by script 07. Existing results can be plotted without
rerunning statistics. Script 13 accepts `--distributions-only` to skip parcel
heatmaps/connectomes; those require the original subject matrices.

## Analysis settings

- Whole-brain topology thresholds the original 100 × 100 matrix at 20%.
  Within-network topology first extracts each original network submatrix, then
  independently retains its strongest 20% of unique edges and binarizes them.
  The retained edge count is rounded down.
- Inclusion requires at least 50 epochs in `n_epochs_after_autoreject`; no participant
  is explicitly excluded by ID. Available demographics are still required.
- Within-network topology includes six Schaefer networks, excluding Limbic.
  Within-network connectivity and other network analyses retain all seven networks.
- Clustering and efficiency retain the existing binary BCT definitions and
  missing-value convention. Mean connectivity uses unthresholded weights.
- GEE uses Gaussian models, exchangeable working correlation, participant clusters,
  robust covariance, age and categorical sex. HC is the reference group.
- Planned contrasts are transitioners minus non-transitioners, transitioners minus HC,
  and non-transitioners minus HC. `passes_interaction_gate` requires both the contrast
  and corresponding interaction to pass their existing FDR corrections.
- PSD uses 1–45 Hz, HC3 linear models and restricted wild-bootstrap cluster inference.
  NBS uses unadjusted pooled-variance Student t statistics on original connectivity
  matrices, with component-level inference.
- NBS uses 5,000 label permutations at |t| > 3.0, 3.5 and 4.0. Components are
  defined by suprathreshold edges regardless of sign; mixed-sign components are
  possible. Familywise correction applies within each comparison/measure/band/
  threshold analysis, not jointly across analyses or to individual edges.

## Figures

Default Matplotlib fonts; SVG and 600-dpi PNG. Group colors are orange for
transitioners, green for non-transitioners and blue for healthy controls.
Box, violin and raincloud fills are translucent; scatter points retain the
same group colors at full opacity. Adjusted estimates and intervals are opaque.

Topology glass-brain maps color only networks passing the interaction gate,
with black outlines; all other effects are masked. Surface maps are not generated.
Topology and connectivity rainclouds show observed data and saved GEE-adjusted
means with 95% CIs. Topology rainclouds cover network/metric/band combinations with
at least one interaction-supported contrast, retaining all three groups in each.
Boxplots show all included networks, not only significant ones.
Within-network parcel heatmaps display all differences; connectomes retain
the strongest 20% of absolute differences for display only.
Between-network figures are boxplots, vertical adjusted pointplots and heatmaps.
Pain rainclouds show half-violins, boxplots and paired participant points in separate
transition-group panels (12 × 6 inches), without sample-size or mean-value labels.
No new statistical tests are performed by plotting scripts.
