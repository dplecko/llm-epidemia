
import pandas as pd

def split_counts(df, cat_col, pct_cols, cnt_cols, A_col="sex", B_col="race"):
    out = []
    for _, grp in df.groupby(cat_col):
        for cnt_col in cnt_cols:
            total = grp[cnt_col].values[0]
            for pct_col in pct_cols:
                a_cat = pct_col
                b_cat = cnt_col
                pct = grp[pct_col].values[0]
                cnt = total * pct
                out.append({cat_col: grp[cat_col].values[0], A_col: a_cat, B_col: b_cat, 'weight': cnt})
    return pd.DataFrame(out)