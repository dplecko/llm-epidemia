

for dts in ["acs", "brfss", "meps", "nhanes", "nsduh", "scf", "gss", "labor", "edu", "fbi_arrests"]:
    print(f"Dataset: {dts}", f"with {len(load_dataset(dts))} rows\n")