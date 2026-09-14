



def cal_flow(mfc_df):
    base_P = 14.696
    if mfc_df["CO - 20 sccm_Flow_sccm"] <= 0.1:
        mfc_df["conc"] = 1 * mfc_df["CO - 20 sccm_Flow_sccm"]/(mfc_df["CO - 20 sccm_Flow_sccm"]+mfc_df["N2 - 100 sccm_Flow_sccm"])*101325*(14.696)/14.696/8.314/(273.15+100)
    elif mfc_df["H2 - 200 sccm"] == 0.0:
        mfc_df["conc"] = 1 * mfc_df["H2 - 200 sccm_Flow_sccm"]/(mfc_df["H2 - 200 sccm_Flow_sccm"]+mfc_df["N2 - 100 sccm_Flow_sccm"])*101325*(14.696)/14.696/8.314/(273.15+100)
    return mfc_df