## Compiling

1. `mkdir build`
1. `cd build`
1. `cmake ../cmake_superbuild/`
1. `cmake --build .`
1. `cmake --build predict` (to rebuild only final target)

## Usage

From the `build` folder: `./predict/predict --model ../../../data/models/1.xgb --data ../../../data/input/data.csv --result /tmp/c.txt`
