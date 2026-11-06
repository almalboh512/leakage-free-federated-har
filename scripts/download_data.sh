#!/usr/bin/env bash
# Downloads UCI-HAR and UCF101 plus its official train/test lists into ./data (or the directory given as $1).
set -e
DATA=${1:-data}
mkdir -p "$DATA" && cd "$DATA"

if [ ! -d "UCI HAR Dataset" ]; then
  wget -q "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip" -O uci.zip \
    || wget -q "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip" -O uci.zip
  unzip -q -o uci.zip
  # the newer archive wraps the dataset in a second zip
  if [ ! -d "UCI HAR Dataset" ] && [ -f "UCI HAR Dataset.zip" ]; then unzip -q -o "UCI HAR Dataset.zip"; fi
fi

if [ ! -d "UCF-101" ]; then
  wget -q --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar -O UCF101.rar
  if command -v unrar > /dev/null; then unrar x -inul UCF101.rar
  elif command -v 7z > /dev/null; then 7z x -bso0 UCF101.rar
  else echo "install unrar or p7zip to extract UCF101.rar" >&2; exit 1; fi
fi

if [ ! -d "ucfTrainTestlist" ]; then
  wget -q --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip -O splits.zip
  unzip -q -o splits.zip
fi

ls "UCI HAR Dataset" | head -3
ls UCF-101 | head -3
ls ucfTrainTestlist
