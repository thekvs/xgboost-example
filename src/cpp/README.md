## Compiling

1. [Install vcpkg](https://vcpkg.io/en/getting-started.html).
2. `cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-linux-release -DCMAKE_BUILD_TYPE=Release`
3. `cmake --build build --parallel $(nproc)`

## Usage

From the `build` folder: `./predict --model ../../../data/models/1.xgb --data ../../../data/input/data.csv --result /tmp/c.txt`
