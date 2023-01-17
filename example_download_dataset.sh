# copy dataset from aws
PATH_TO_DATASET="temp_dataset"
# might have to specifcy profile
~/aws-cli/bin/aws s3 sync s3://ff350d3a-89fc-11ec-a398-ac1f6baca408/datasets/cam4_v5 $PATH_TO_DATASET --profile gaia
