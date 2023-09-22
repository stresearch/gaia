from gaia.preprocess import DataConstructor
import glob
from gaia import get_logger


logger = get_logger(__name__)

cache = "/dev/shm/gaia"
save_location = "."
file_pattern = "/ssddg1/gaia/nc_data/*.nc"
files = sorted(glob.glob(file_pattern))

input = "Q,T,U,V,OMEGA,PSL,SOLIN,SHFLX,LHFLX,FSNS,FLNS,FSNT,FLNT,Z3".split(",")
output = "PRECT,PRECC,PTEQ,PTTEND".split(",")

name_of_dataset = "dummy" 

#we want 2 samples per day after keep two consecutive time steps
steps_per_day = 16
adjacent_time_steps = 2
target_samples_per_day = 2
subsample_factor = steps_per_day//adjacent_time_steps//target_samples_per_day
num_workers = 4

logger.info("Preprocessing files train/val")

DataConstructor.preprocess_local_files(
    split="train",
    dataset_name=name_of_dataset,
    files=files,
    save_location=save_location,
    train_years=2,
    subsample_factor=subsample_factor,
    cache=cache,
    workers=num_workers,
    time_steps=adjacent_time_steps
)


logger.info("Preprocessing files test")

DataConstructor.preprocess_local_files(
    split="test",
    dataset_name=name_of_dataset,
    files=files,
    save_location=save_location,
    train_years=2,
    subsample_factor=subsample_factor,
    cache=cache,
    workers=num_workers,
    time_steps=adjacent_time_steps
)
