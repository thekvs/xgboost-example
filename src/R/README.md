CRAN seems to have older version of `xgboost` so you need to install it manually.

1. `git clone --recursive https://github.com/dmlc/xgboost`
1. `cd xgboost`
1. `git checkout release_0.90`
1. `R CMD INSTALL .`

Usage: `Rscript predict.R ../../data/models/1.xgb ../../data/input/data.csv /tmp/r.txt`
