import xarray as xr
import s3fs


def read_xarray_netcdf_from_s3(s3_file, aws_access_key_id, aws_secret_access_key):

    fs = s3fs.S3FileSystem(
        anon=False, key=aws_access_key_id, secret=aws_secret_access_key
    )
    with fs.open(s3_file, "rb") as f:
        ds = xr.open_dataset(f, engine="h5netcdf")

    return ds



def extract_variables(xarray_data, var_list):
    pass
