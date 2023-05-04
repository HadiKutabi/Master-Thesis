# Master-Thesis

This repository contains all the python files used for my master thesis.


To run the different libraries, the following python versions were used: 

| AutoML Framework | Python Version |   
|------------------|----------------|
| AutoSklearn      | 3.7            |
| TPOT             | 3.7            | 
| DSWIZARD         | 3.8            |  
| AlphaD3M         |                |  

I used a separate venv for each library. The corresponding dependencies are to be found under /config.

----------------------------
## Datasets 

Datasets can be downloaded by running 

```
python utils/download_openml_datasets.py
```
the directory `datasets` should be then automatically created for saving each dataset. 


------------------------------
## Running AutoML Frameworks

In each directory starting with `_`, run can be directly executed and the results should be saved automatically. 

note: before running DSWIZARD, make sure to run `./_dswizard/get_meta_learning_base.sh`


-------------------------- 

## Configs

All config files for AutoML framework, seed, dependencies are in `config`