import pandas as pd
import argparse
import sys

from prep_dataframe import read_dataframe

def clean_columns(df):

    df.columns = df.columns.str.lower().str.replace('-', '_')
    
    df.columns = ['fs_' + col if col != 'scan_age' else col for col in df.columns]
    df.rename(columns={'fs_project': 'project', 'fs_site': 'site'}, inplace=True)

    if 'fs_scan_id' in df.columns:
        df.rename(columns={'fs_scan_id': 'scan_id'}, inplace=True)

    df.columns = df.columns.str.replace('right', 'r')
    df.columns = df.columns.str.replace('rh_', 'r_')
    df.columns = df.columns.str.replace('lh_', 'l_')
    df.columns = df.columns.str.replace('volume', 'grayvol')
    df.columns = df.columns.str.replace('lh', 'l')
    df.columns = df.columns.str.replace('left', 'l')
    df.columns = df.columns.str.replace('thickness', 'thck')
    df.columns = df.columns.str.replace('white_matter', 'wm')

    df.columns = df.columns.str.replace('choroid_plexus', 'choroidplexus_vol')
    df.columns = df.columns.str.replace('amygdala', 'amygdala_vol')
    df.columns = df.columns.str.replace('accumbens_area', 'accumbensarea_vol')
    df.columns = df.columns.str.replace('accumbens_area', 'accumbensarea_vol')
    df.columns = df.columns.str.replace('hippocampus', 'hippo_vol')
    df.columns = df.columns.str.replace('maskvol', 'mask_vol')
    df.columns = df.columns.str.replace('vessel', 'vessel_vol')

    df.columns = df.columns.str.replace('brainseg', 'brainseg_')
    df.columns = df.columns.str.replace('notvent', '_no_vent')
    df.columns = df.columns.str.replace('ventsurf', 'vent_surf')
    df.columns = df.columns.str.replace('vol_to_etiv', 'vol_etiv_ratio')

    df.columns = df.columns.str.replace('rhcort', 'rcort')
    df.columns = df.columns.str.replace('cortexvol', 'cort_gm_vol')
    df.columns = df.columns.str.replace('subcortgrayvol', 'subcort_gm_vol')
    df.columns = df.columns.str.replace('totalgrayvol', 'total_gm_vol')

    df.columns = df.columns.str.replace('fs_4th_ventricle', 'fs_4thvent_vol')
    df.columns = df.columns.str.replace('fs_brain_stem', 'fs_brainstem_vol')
    df.columns = df.columns.str.replace('fs_optic_chiasm', 'fs_opticchiasm_vol')
    df.columns = df.columns.str.replace('fs_mask_vol_etiv_ratio', 'fs_maskvol_etiv_ratio')
    df.columns = df.columns.str.replace('thalamus_proper', 'thalamusproper_vol')

    df.columns = df.columns.str.replace('fs_cc_anterior', 'fs_cc_anterior_vol')
    df.columns = df.columns.str.replace('fs_cc_mid_anterior', 'fs_cc_midanterior_vol')
    df.columns = df.columns.str.replace('fs_cc_posterior', 'fs_cc_posterior_vol')
    df.columns = df.columns.str.replace('fs_supratentorialvol_no_ventvox', 'fs_supratentorial_no_vent_voxel_count')

    df.columns = df.columns.str.replace('fs_brainsegvol_no_vent', 'fs_brainseg_vol_no_vent')
    df.columns = df.columns.str.replace('fs_brainsegvol_no_vent_surf', 'fs_brainseg_vol_no_vent_surf')
    df.columns = df.columns.str.replace('fs_brainseg_vol_etiv_ratio', 'fs_brainsegvol_etiv_ratio')

    df.columns = df.columns.str.replace('putamen', 'putamen_vol')
    df.columns = df.columns.str.replace('_csf', '_csf_vol')
    df.columns = df.columns.str.replace('_inf_lat_vent', '_inflatvent_vol')
    df.columns = df.columns.str.replace('3rd_ventricle', '3rdvent_vol')
    df.columns = df.columns.str.replace('supratentorialvol', 'supratentorial_vol')
    df.columns = df.columns.str.replace('_ventvox', '_vent_voxel_count')

    df.columns = df.columns.str.replace('cerebellum_wm', 'cerebellum_wm_vol')
    df.columns = df.columns.str.replace('cerebellum_cortex', 'cerebellum_cort_vol')
    df.columns = df.columns.str.replace('pallidum', 'pallidum_vol')
    df.columns = df.columns.str.replace('caudate', 'caudate_vol')
    df.columns = df.columns.str.replace('lateral_ventricle', 'latvent_vol')
    df.columns = df.columns.str.replace('fs_cc_central', 'fs_cc_central_vol')
    df.columns = df.columns.str.replace('ventraldc', 'ventdc_vol')
    df.columns = df.columns.str.replace('fs_cc_mid_posterior', 'fs_cc_midposterior_vol')
    df.columns = df.columns.str.replace('fs_cort_gm_vol', 'fs_totcort_gm_vol')

    df.columns = df.columns.str.replace('fs_estimatedtotalintracranialvol', 'fs_intercranial_vol')

    mri = read_dataframe(df)
    
    return mri


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default=f"/data/raw",
        help="the location where the raw data is located."
    )
    args = parser.parse_args()

    print("dataframe columns cleaned successfully")