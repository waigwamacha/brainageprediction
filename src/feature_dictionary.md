# ASEG ROIs

## Subcortical features and the stats header

## ASEG stats dictionary

### ASEG_STATS_DICT

- etiv: EstimatedTotalIntraCranialVo
- bseg: BrainSeg
- bsegNV: BrainSegNotVent
- bsegNVS: BrainSegNotVentSurf
- lGM: lhCortex
- rGM: rhCortex
- totCortGM: Cortex
- subCortGM: SubCortGray
- totGM: TotalGray
- supTent: SupraTentorial
- supTentNV: SupraTentorialNotVent
- supTentNVv: SupraTentorialNotVentVox
- lWM: lhCorticalWhiteMatter
- rWM: rhCorticalWhiteMatter
- totCortWM: CorticalWhiteMatter
- maskVol: Mask
- etivRatio: BrainSegVol-to-eTIV
- maskRatio: MaskVol-to-eTIV
- lHoles: lhSurfaceHoles
- rHoles: rhSurfaceHoles
- totHoles: SurfaceHole

### 21 features

aseg_stats_hdr =
  ["FS_InterCranial_Vol", "FS_BrainSeg_Vol",
  "FS_BrainSeg_Vol_No_Vent", "FS_BrainSeg_Vol_No_Vent_Surf",
  "FS_LCort_GM_Vol", "FS_RCort_GM_Vol", "FS_TotCort_GM_Vol",
  "FS_SubCort_GM_Vol", "FS_Total_GM_Vol", "FS_SupraTentorial_Vol",
  "FS_SupraTentorial_Vol_No_Vent",
  "FS_SupraTentorial_No_Vent_Voxel_Count",]

- drop due to missing across datasets = 18

  "FS_L_WM_Vol", "FS_R_WM_Vol", "FS_Tot_WM_Vol",
  "FS_Mask_Vol", "FS_BrainSegVol_eTIV_Ratio",
  "FS_MaskVol_eTIV_Ratio"

- drop due to QC already predicted with Qoala-T = 15

  "FS_LH_Defect_Holes", "FS_RH_Defect_Holes", "FS_Total_Defect_Holes"

### 45 features

- missing in HCP: FS_L_InfLatVent_Mean, FS_R_Cerebellum_Cort_Mean, FS_Non-WM_Hypointens_Mean

aseg_roi_names =
  ["FS_L_LatVent", "FS_L_InfLatVent", 
  "FS_L_Cerebellum_WM",
  "FS_L_Cerebellum_Cort", "FS_L_ThalamusProper", "FS_L_Caudate",
  "FS_L_Putamen", "FS_L_Pallidum", "FS_3rdVent", "FS_4thVent",
  "FS_BrainStem", "FS_L_Hippo", "FS_L_Amygdala", "FS_CSF",
  "FS_L_AccumbensArea", "FS_L_VentDC", "FS_L_Vessel", "FS_L_ChoroidPlexus",
  "FS_R_LatVent", "FS_R_InfLatVent", "FS_R_Cerebellum_WM",
  "FS_R_Cerebellum_Cort", ** not in HCP
  "FS_R_ThalamusProper", "FS_R_Caudate",
  "FS_R_Putamen", "FS_R_Pallidum", "FS_R_Hippo", "FS_R_Amygdala",
  "FS_R_AccumbensArea", "FS_R_VentDC", "FS_R_Vessel",
  "FS_R_ChoroidPlexus", "FS_5thVent",
  "FS_OpticChiasm", "FS_CC_Posterior", "FS_CC_MidPosterior",
  "FS_CC_Central", "FS_CC_MidAnterior", "FS_CC_Anterior"
  ]

- Notes
  - "FS_WM_Hypointens", "FS_L_WM_Hypointens", "FS_R_WM_Hypointens", 
  - "FS_Non-WM_Hypointens", "FS_L_Non-WM_Hypointens", "FS_R_Non-WM_Hypointens",
  - "FS_R_Cerebellum_Cort", ** not in HCP

#### Functions for calculating mean and volume of aseg ROIs

- aseg_roi_means <- paste0(aseg_roi_names, "_Mean")
- aseg_roi_vol <- paste0(aseg_roi_names, "_Vol")