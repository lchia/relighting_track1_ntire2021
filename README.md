# NTIRE 2021 Depth Guided Relighting Challenge Track 1: One-to-one relighting

## make python-env
```python
conda create -n relighting2021 python=3.6
pip install -r requirments.txt
```

## run test

- [1] Put the test data of track1 into folder
`./relighting2021/data/track1`
*already there*

- [2] Run test script
```python
cd ./relighting2021/code
sh run_test.sh
```

- [3] Fine results in folder
`./relighting2021/results`



## structure
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
