
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

def bin_labels(breaks, unit="", exact=False, last_plus=False, include_bins=False):
    if exact:
        if last_plus:
            labels = [f"{b} {unit}" for b in breaks[:-1]]
            labels.append(f"{breaks[-1]}+ {unit}")
            bins = breaks + [float("inf")]
        else:
            labels = [f"{b} {unit}" for b in breaks]
            bins = breaks
    else:
        labels = [f"< {breaks[0]} {unit}"]
        for i in range(1, len(breaks)):
            labels.append(f"{breaks[i-1]+1}–{breaks[i]} {unit}")
        labels.append(f"{breaks[-1]+1}+ {unit}")
        bins = [float("-inf")] + breaks + [float("inf")]

    if include_bins:
        return bins, labels
    return labels

def discrete_col(df, col, breaks, unit="", exact=False, last_plus=False, right=True, new_col=None):
    bins, labels = bin_labels(breaks, unit=unit, exact=exact, last_plus=last_plus, include_bins=True)
    
    if new_col is None:
        new_col = f"{col}_group"
    
    df[new_col] = pd.cut(df[col], bins=bins, labels=labels, right=right, include_lowest=True)
    return df
