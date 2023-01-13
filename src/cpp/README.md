## Compiling

1. `cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-linux-release -DCMAKE_BUILD_TYPE=Release`
2. `cmake --build build --parallel $(nproc)`

## Usage

From the `build` folder: `./predict --model ../../../data/models/1.xgb --data ../../../data/input/data.csv --result /tmp/c.txt`
