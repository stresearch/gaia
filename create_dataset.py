from gaia.data import NCDataConstructor


cam4 = "cam4-famip-30m-timestep"
spcam  = "spcamclbm-nx-16-20m-timestep"

if __name__=="__main__":
    NCDataConstructor.default_data(split="train",  workers  =32, prefix=cam4, train_years=3, save_location=".", cache = ".")
    # NCDataConstructor.default_data(split="train")