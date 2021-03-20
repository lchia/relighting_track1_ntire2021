# NTIRE 2021 Depth Guided Relighting Challenge Track 1: One-to-one relighting

## Make Environments
```python
conda create -n relighting2021 python=3.6
pip install -r requirments.txt
```

## Run test

- [1] Download [data.zip](https://drive.google.com/file/d/1oFCsGBcs-a3yIAr6BiA2mcXTTl2D5xAB/view?usp=sharing) from Google Drive



- [2] Unzip `data.zip` and put `data` folder to the parent directory, like `../data`, because we made soft links of `ckpts`, `data`, `results`, `TMP` in the current directory (showed in `structure` below) from `../data`.

```python
ln -s ../data/ckpts .
ln -s ../data/results .
ln -s ../data/data .
ln -s ../data/TMP .
```

- [3] Run test script
```python
cd ./code
sh run_test.sh
```

- *TEST* results lies in folder: `../results`

and can download from [results.zip](https://drive.google.com/file/d/1Q2H95cTqtCKxi7L1SmA3NTTYjqP5tEgr/view?usp=sharing)

## Structure
--relighting2021
 |--README.md 
 |--requirements.txt
 |--ckpts: model checkpoints
 |--data: test data
 |--results: final results of track1
 |--TMP: temp directory
 |--code: codes
    |--run_test.sh: test script
    |--fuse.py: test fusion
    |--main.py
    |--others
