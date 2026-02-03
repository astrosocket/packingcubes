## [unreleased]

### 🐛 Bug Fixes

- Fix node search starting from root children instead of root ([595b4f7](https://github.com/astrosocket/packingcubes/commit/595b4f7ac82ac426259722f1eb149d9b738a30c0))
- Allow for running without numba ([eb1545a](https://github.com/astrosocket/packingcubes/commit/eb1545acdc8efac3efe476b2ba3b5ce88846a0ac))
- Up position precision ([122abdd](https://github.com/astrosocket/packingcubes/commit/122abdd6de1efe2bec32131369d7c044e3a0038b))
- Move and update set_bounding_box to Dataset ([1a7abc0](https://github.com/astrosocket/packingcubes/commit/1a7abc03a6cb0d4ce872e6a09aea07716ba8a6a4))
- Tighten valid box precision check ([7a48b0d](https://github.com/astrosocket/packingcubes/commit/7a48b0de2666a39213e183ce3f7012e7819da616))
- Remove extraneous @njit decorator on full_morton ([d87f3a1](https://github.com/astrosocket/packingcubes/commit/d87f3a170b2c933e992cd03c7245127e1ecaad25))

### 💼 Other

- Creating temporary current release ([09bade4](https://github.com/astrosocket/packingcubes/commit/09bade48ee678f2973e7fb999efda2792597e3c7))

### 🚜 Refactor

- Clean up functions that are not and will not be used ([5089ec0](https://github.com/astrosocket/packingcubes/commit/5089ec0548c375eb3852c4c4a9f9cc4c5b61fd19))
- Remove tqdm from octrees and clean up pyproject.toml ([d755a22](https://github.com/astrosocket/packingcubes/commit/d755a22cfa171048cdff62c7172f192ce287a332))

### 📚 Documentation

- Fix source argument description ([71d74f9](https://github.com/astrosocket/packingcubes/commit/71d74f994d782f00f767929395acdce0b99f5804))
- Improve argument description ([54cb523](https://github.com/astrosocket/packingcubes/commit/54cb523dd011b17000f3eec4f1a78ddf60df7fb3))

### ⚡ Performance

- Unroll some tight loops based on profiling ([dd82d5d](https://github.com/astrosocket/packingcubes/commit/dd82d5dd484855e299c9dcc2158094f5a5f46654))
- Merge Numba changes ([#4](https://github.com/astrosocket/packingcubes/issues/4)) ([ad03775](https://github.com/astrosocket/packingcubes/commit/ad037751cda3014158c3b53ba94fce1d2e99eea1))

### 🧪 Testing

- Add PythonOctree search tests ([456bafd](https://github.com/astrosocket/packingcubes/commit/456bafde2f5c9f1a65149603ebbdc79fcd6b95cc))
- Additional cleanup plus deadline removal ([192bfd7](https://github.com/astrosocket/packingcubes/commit/192bfd764381f721b1752422803ebc03ba7bc9d4))
- Update to match the new/updated BoundingBox and DataContainer ([559eebb](https://github.com/astrosocket/packingcubes/commit/559eebb3f72b55faf1dcacdb93dd4fbe45bea8ad))
- Update conftest to support Numba API ([e99d774](https://github.com/astrosocket/packingcubes/commit/e99d7743fa87f811d9e4e9cdab6a7bac75e9af65))
- Add possible missing warnings for invalid indices ([e31a67d](https://github.com/astrosocket/packingcubes/commit/e31a67d6d5d7242fd1217efed6e79945a0a1c59f))
- Update bounding_box tests for numba ([994069c](https://github.com/astrosocket/packingcubes/commit/994069cfb202213747577ec0a284c8031a64f3f3))
- Allow decimation factor as CLI argument ([dc37d49](https://github.com/astrosocket/packingcubes/commit/dc37d49bca18257e4137a8b5fb2f763901dd38c5))
- Specify particle type to use ([3046a6d](https://github.com/astrosocket/packingcubes/commit/3046a6d410b58c16621c2bc6ea3a0de77254fb06))
- Rearrange test output ([c30ad8c](https://github.com/astrosocket/packingcubes/commit/c30ad8ced6655d995ef1302244de931ee2e78f64))
- Add dataset memory usage, remove debug print statements ([136201d](https://github.com/astrosocket/packingcubes/commit/136201dd8663a5c134625bb8465bf1a89a8c3803))
- Improve timing tests ([5699341](https://github.com/astrosocket/packingcubes/commit/56993412732cd29c9f6df046a9c4a9bd2b408678))
- Add randomization to search timing ([920432e](https://github.com/astrosocket/packingcubes/commit/920432e5462078751660b268af530510afdc253e))
- Improve timing test output ([7874aa4](https://github.com/astrosocket/packingcubes/commit/7874aa47188138a16dcb5c6dad303500c91fce41))

### ⚙️ Miscellaneous Tasks

- Update pixi.lock ([4c32262](https://github.com/astrosocket/packingcubes/commit/4c32262a302beb35245d101be7898f9a5619a246))
- Remove auth-host/token as unnecessary ([a9afe65](https://github.com/astrosocket/packingcubes/commit/a9afe650f58ce29071cc663cd0cbdc843b78444c))
- Remove matrix for the time being ([8b1322a](https://github.com/astrosocket/packingcubes/commit/8b1322af578b4cc7ec7db399e8537394ba31f7ec))
- Add macos to test suite to see if it resolves runs-on issue ([c22884b](https://github.com/astrosocket/packingcubes/commit/c22884bcedadc620f29e92a8bd780af0cb7cc2da))
- Remove extra hyphen ([029b3f9](https://github.com/astrosocket/packingcubes/commit/029b3f9361ef1f01e56d472512a0d2d12cc6529b))
- Move name to correct spot ([9d1d777](https://github.com/astrosocket/packingcubes/commit/9d1d777f34c69e49545afff5e203dd6c752d7c25))
- Move dependabot.yml to correct folder ([4192aaa](https://github.com/astrosocket/packingcubes/commit/4192aaa4854049cee0ecca22be9a5459cf7ec845))
- Add missing --- ([10ea0ff](https://github.com/astrosocket/packingcubes/commit/10ea0ffd38b671a3266aab3d6a3423b3938c2943))
- Add quotes around fields ([4d89e73](https://github.com/astrosocket/packingcubes/commit/4d89e73a90f664745ef15739ed616adc7fd56aee))
- Add initial CI and dependabot workflows ([f9b6dac](https://github.com/astrosocket/packingcubes/commit/f9b6dacc21b0a05edeb309b1e71604f3472be226))
## [before_numba] - 2026-01-20

### 🚀 Features

- Add CHANGELOG.md based on git-cliff ([663325a](https://github.com/astrosocket/packingcubes/commit/663325ae95a9deb2a82cfafd572f18cf13ee736f))
- Add packingcubes configuration file to gitignore ([de7225f](https://github.com/astrosocket/packingcubes/commit/de7225f1e30d202fbd5f0f2bf9c06ddaa00682b3))
- Add constructor for PackedTree ([193bf9b](https://github.com/astrosocket/packingcubes/commit/193bf9b2c91498db5559e1810eb5a95558c5cd74))
- Move complicated node state update to own function ([ad517d8](https://github.com/astrosocket/packingcubes/commit/ad517d85efbbc0c47d7fa28e258bea77182a5404))
- Add _print_packed class method for debugging purposes ([65f2c9d](https://github.com/astrosocket/packingcubes/commit/65f2c9dc0c07c105a75c8bc3eab21bc8b0db2b83))
- Simplify partition_data/child_list processing ([fe88cd1](https://github.com/astrosocket/packingcubes/commit/fe88cd1db0f44075333b7ced9d59f4cc03063e7d))
- Add packingcubes configuration loading ([f5b8509](https://github.com/astrosocket/packingcubes/commit/f5b8509668a7245b37978d1147f56de7c6514fa3))
- Setup dynamic versioning and clean up pyproject ([343dda4](https://github.com/astrosocket/packingcubes/commit/343dda4452dd65ae6284e983d881c1492fdac8b1))
- Calculate field formats. Remove extra attributes ([61ad94b](https://github.com/astrosocket/packingcubes/commit/61ad94bbec9fac379001ab3b093e300a70cdd1ee))
- Add get_name(CurrentNode) ([5a73900](https://github.com/astrosocket/packingcubes/commit/5a739007facb78c2db7cd9b095a3783ca2452332))
- Switch to regular ints, fix off-by-one error ([0671e02](https://github.com/astrosocket/packingcubes/commit/0671e021615a85e2b0c2ce3095f6d158ae9414cd))
- Return start-stop list instead of actual indices ([67f448d](https://github.com/astrosocket/packingcubes/commit/67f448df158a3fad5048f755ec149af0ae313607))
- Change to str-based tag ([5c98570](https://github.com/astrosocket/packingcubes/commit/5c985703b87a617b66a804d470a50977f0a837e7))
- Switch to using BoundingBox.copy directly ([008a495](https://github.com/astrosocket/packingcubes/commit/008a495b1799b19b1e7318cd1789518ac8f4dfa6))
- Add initial PackedTree implementation ([3084711](https://github.com/astrosocket/packingcubes/commit/30847113ee6210c08fa3269f3b722803ea3f8b7d))
- Add packed tree documentation and to_packed method ([7a78cd3](https://github.com/astrosocket/packingcubes/commit/7a78cd39d2fa473912bf06d0550a519113a0111c))
- Convert OctreeNode/Octree to abstract/Protocol ([012c711](https://github.com/astrosocket/packingcubes/commit/012c71169188b7dc7ab58b05f93e8d7cf0f79a2a))
- Update tag-str conversion ([6bdb9a3](https://github.com/astrosocket/packingcubes/commit/6bdb9a3724fda802b06976d4b11dc068a3f60f30))
- Add copyable protocol to BoundingBox ([aa4df9e](https://github.com/astrosocket/packingcubes/commit/aa4df9e6c3b084d3436cc4c6b68a1d3ae5a8fc23))
- Additional PERF based linting fix but removing PERF in general ([4ebd465](https://github.com/astrosocket/packingcubes/commit/4ebd4657cd0aaec69d5d949193e4f9836b05e79a))

### 🐛 Bug Fixes

- BoxLikes must be validated ([e64853a](https://github.com/astrosocket/packingcubes/commit/e64853af47a2eb84cf68ab6f3094cdd5ad8b4f06))
- Skipping root node ([2d9d0fe](https://github.com/astrosocket/packingcubes/commit/2d9d0fe3215f3d0b6f8a56c919101016b9d8a087))
- Fix insufficient default containment test ([f647947](https://github.com/astrosocket/packingcubes/commit/f64794753cbbdaf4cb5e1dfd51923e223d7d755c))
- Fix circular import ([54e45f4](https://github.com/astrosocket/packingcubes/commit/54e45f449f6bccdf746f27895b98c6018c1a8a3b))
- Fix off-by-one error ([a84aac9](https://github.com/astrosocket/packingcubes/commit/a84aac92e896d40dbb1b7588913ecd6a239d5a11))
- Remove extra self ([d386530](https://github.com/astrosocket/packingcubes/commit/d3865309506e647cd1561e7ed6bcf854dd7c7700))
- Convert to 2d array ([e8fdd1a](https://github.com/astrosocket/packingcubes/commit/e8fdd1ab6730d480ecfe084c0fec900c49d3c9cf))
- Handle case where no node with tag exists ([a605527](https://github.com/astrosocket/packingcubes/commit/a605527264b6398d8c3609218765a1224d662ccc))
- Ignore OOB hi index if it doesn't matter ([5c5b052](https://github.com/astrosocket/packingcubes/commit/5c5b05240cec35411649854422ac3894d7ab648c))
- Was generating invalid data in basic_data_strategy ([3c21d43](https://github.com/astrosocket/packingcubes/commit/3c21d435f497246ebeff61b471156a614d72965f))
- Morton was returning float arrays when ints would be better ([5d9fda4](https://github.com/astrosocket/packingcubes/commit/5d9fda443adc98761f18fa31b49a755a283b368f))

### 💼 Other

- Exclude _version.py from test coverage ([0ec8492](https://github.com/astrosocket/packingcubes/commit/0ec84923bd11134e04ec9b82cfda54db60dba53c))
- Add missing typing, swap argument order ([434c829](https://github.com/astrosocket/packingcubes/commit/434c829593b97b8bee58fd103b7121db6b607f61))
- Fix typing, remove explicit copy module dependence ([2bb3524](https://github.com/astrosocket/packingcubes/commit/2bb35249845873dbe2879ef57aeaa56bd507d333))
- Fix typing issues in octree.py ([68122e9](https://github.com/astrosocket/packingcubes/commit/68122e95d59b39e24ccff9214584c923f4a4621d))
- Updated pixi.lock file ([7802ebe](https://github.com/astrosocket/packingcubes/commit/7802ebe5992f05f9ada00f7faba369319efa2610))
- Update octree(node) typing in tree_vis ([1ae0609](https://github.com/astrosocket/packingcubes/commit/1ae06097844cd24c1279f46259e59f5f01eb4409))
- Switch to memory-mapped bytes array for packed output ([1ae2332](https://github.com/astrosocket/packingcubes/commit/1ae2332bf01f75dc7028e2065a25014d86139012))
- Provide both vertex testing and projection testing ([0ab0cf7](https://github.com/astrosocket/packingcubes/commit/0ab0cf7104a8283b7e438c63db70037ce51c8b24))
- Switch to memoryview and array casting ([3e915e0](https://github.com/astrosocket/packingcubes/commit/3e915e02647d1fbeaf590eb40ca686746316c18e))
- Unroll get_box_vertices loops ([a6344e2](https://github.com/astrosocket/packingcubes/commit/a6344e2498bf54fe845fac97f718bb5cd09cf9a2))
- Fix typing for mypy, use filter ([7a2c70c](https://github.com/astrosocket/packingcubes/commit/7a2c70cc313d98b4ef96277aaf41e3f7f577b8c1))
- Fix pbar typing complaint ([251fa3e](https://github.com/astrosocket/packingcubes/commit/251fa3ee8d68798ababac908eb394b0f1c978f2f))
- Return box copy ([dcbf932](https://github.com/astrosocket/packingcubes/commit/dcbf932c929ba180b872b716f55e3b4f7e7f48d5))
- Scalene, pympler ([ca1d7cb](https://github.com/astrosocket/packingcubes/commit/ca1d7cb9d1906e6aa12505824b601d251fb7f8fa))
- Changes due to mypy ([a5eb6b3](https://github.com/astrosocket/packingcubes/commit/a5eb6b3c97939e0ddbb7fa8297adbc9dae4f2d6f))
- Changes from increased linting levels. ([b0bab0b](https://github.com/astrosocket/packingcubes/commit/b0bab0b616767810159afa04c51836be46aca71e))

### 📚 Documentation

- Added/updated PackedTree documentation ([6994bc5](https://github.com/astrosocket/packingcubes/commit/6994bc5eea76d09021bf663f4da17317431c5588))
- Fixes and updates. ENH: Minor internal API change ([32a742c](https://github.com/astrosocket/packingcubes/commit/32a742c493be45e7b073fa56443f03b76483d432))

### ⚡ Performance

- Remove unnecessary searches, unused variables ([ec67196](https://github.com/astrosocket/packingcubes/commit/ec67196e246ef81da96a9838a270a916b5dbfc6e))

### 🧪 Testing

- Skip empty files ([9bda7b7](https://github.com/astrosocket/packingcubes/commit/9bda7b72048eba6c964896a3f2b5b6f91f9b6337))
- Update bounding_box and octree tests to match new APIs ([095f920](https://github.com/astrosocket/packingcubes/commit/095f920eb1916d3bafd5a43f9a3fb485fcbb42c1))

### ⚙️ Miscellaneous Tasks

- Add basic .gitlint file ([5c3a2e2](https://github.com/astrosocket/packingcubes/commit/5c3a2e261d21f06afea804f4a81eabd72b8cfad4))
- Ignore profiler output ([7e302b3](https://github.com/astrosocket/packingcubes/commit/7e302b39745f7ca79257b20f2eb7e44b1cd00b99))
<!-- generated by git-cliff -->
