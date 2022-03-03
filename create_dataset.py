from gaia.data import NCDataConstructor


if __name__=="__main__":
    NCDataConstructor.default_data(split="train")
    NCDataConstructor.default_data(split="test")