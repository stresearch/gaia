from gaia.data import NCDataConstructor


cam4 = "cam4-famip-30m-timestep"
spcam  = "spcamclbm-nx-16-20m-timestep"
workers = 64
cache = "cache"

if __name__=="__main__":
    # NCDataConstructor.default_data(split="train",  workers  =workers, prefix=cam4, train_years=3, save_location=".", cache = cache)
    # NCDataConstructor.default_data(split="test",  workers  =workers, prefix=cam4, train_years=3, save_location=".", cache = cache)
    # NCDataConstructor.default_data(split="train",  workers  =workers, prefix=spcam, train_years=2, save_location=".", cache = cache)
    NCDataConstructor.default_data(split="test",  workers  =workers, prefix=spcam, train_years=2, save_location=".", cache = cache)


    # NCDataConstructor.default_data(split="train")