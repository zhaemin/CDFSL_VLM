#!/bin/bash

# forces the user to set the DATA_PATH variable or fails
if [ ! -v DATA_PATH ]; then
    echo "DATA_PATH is unset. Please make sure to set it while calling the script."
    exit 0
fi

# store the initial directory
DIR=$(pwd)

# move to the data directory and download the splits from google drive
cd ${DATA_PATH}
TMP=$RANDOM
pip install -U gdown
gdown https://drive.google.com/drive/folders/1uZN-Mjqz9U3Kjn36lj8eUhvJskBllT6j -O $TMP --folder

# extract all the tars
shots="1 2 4 8 16"
for shot in $shots; do
    echo Extracting split for $shot shots...
    sleep 1
    tar -xvf $TMP/jsonl_${shot}shot.tar
done

# once everything is done, get rid of the downloaded folder
# and move the readme of the splits to the main root of the repo
mv $TMP/README.txt $DIR/INFO4SPLITS.txt
rm -rf $TMP

# go back to original directory
cd $DIR
echo Setup finished, please have a look at $DATA_PATH for the splits! 