## [before_numba] - 2026-01-17

### 🚀 Features

- Add packingcubes configuration file to gitignore ([de7225f](de7225f1e30d202fbd5f0f2bf9c06ddaa00682b3))
- Add constructor for PackedTree ([193bf9b](193bf9b2c91498db5559e1810eb5a95558c5cd74))
- Move complicated node state update to own function ([ad517d8](ad517d85efbbc0c47d7fa28e258bea77182a5404))
- Add _print_packed class method for debugging purposes ([65f2c9d](65f2c9dc0c07c105a75c8bc3eab21bc8b0db2b83))
- Simplify partition_data/child_list processing ([fe88cd1](fe88cd1db0f44075333b7ced9d59f4cc03063e7d))
- Add packingcubes configuration loading ([f5b8509](f5b8509668a7245b37978d1147f56de7c6514fa3))
- Setup dynamic versioning and clean up pyproject ([343dda4](343dda4452dd65ae6284e983d881c1492fdac8b1))
- Calculate field formats. Remove extra attributes ([61ad94b](61ad94bbec9fac379001ab3b093e300a70cdd1ee))
- Add get_name(CurrentNode) ([5a73900](5a739007facb78c2db7cd9b095a3783ca2452332))
- Switch to regular ints, fix off-by-one error ([0671e02](0671e021615a85e2b0c2ce3095f6d158ae9414cd))
- Return start-stop list instead of actual indices ([67f448d](67f448df158a3fad5048f755ec149af0ae313607))
- Change to str-based tag ([5c98570](5c985703b87a617b66a804d470a50977f0a837e7))
- Switch to using BoundingBox.copy directly ([008a495](008a495b1799b19b1e7318cd1789518ac8f4dfa6))
- Add initial PackedTree implementation ([3084711](30847113ee6210c08fa3269f3b722803ea3f8b7d))
- Add packed tree documentation and to_packed method ([7a78cd3](7a78cd39d2fa473912bf06d0550a519113a0111c))
- Convert OctreeNode/Octree to abstract/Protocol ([012c711](012c71169188b7dc7ab58b05f93e8d7cf0f79a2a))
- Update tag-str conversion ([6bdb9a3](6bdb9a3724fda802b06976d4b11dc068a3f60f30))
- Add copyable protocol to BoundingBox ([aa4df9e](aa4df9e6c3b084d3436cc4c6b68a1d3ae5a8fc23))
- Additional PERF based linting fix but removing PERF in general ([4ebd465](4ebd4657cd0aaec69d5d949193e4f9836b05e79a))

### 🐛 Bug Fixes

- BoxLikes must be validated ([e64853a](e64853af47a2eb84cf68ab6f3094cdd5ad8b4f06))
- Skipping root node ([2d9d0fe](2d9d0fe3215f3d0b6f8a56c919101016b9d8a087))
- Fix insufficient default containment test ([f647947](f64794753cbbdaf4cb5e1dfd51923e223d7d755c))
- Fix circular import ([54e45f4](54e45f449f6bccdf746f27895b98c6018c1a8a3b))
- Fix off-by-one error ([a84aac9](a84aac92e896d40dbb1b7588913ecd6a239d5a11))
- Remove extra self ([d386530](d3865309506e647cd1561e7ed6bcf854dd7c7700))
- Convert to 2d array ([e8fdd1a](e8fdd1ab6730d480ecfe084c0fec900c49d3c9cf))
- Handle case where no node with tag exists ([a605527](a605527264b6398d8c3609218765a1224d662ccc))
- Ignore OOB hi index if it doesn't matter ([5c5b052](5c5b05240cec35411649854422ac3894d7ab648c))
- Was generating invalid data in basic_data_strategy ([3c21d43](3c21d435f497246ebeff61b471156a614d72965f))
- Morton was returning float arrays when ints would be better ([5d9fda4](5d9fda443adc98761f18fa31b49a755a283b368f))

### 💼 Other

- Exclude _version.py from test coverage ([0ec8492](0ec84923bd11134e04ec9b82cfda54db60dba53c))
- Add missing typing, swap argument order ([434c829](434c829593b97b8bee58fd103b7121db6b607f61))
- Fix typing, remove explicit copy module dependence ([2bb3524](2bb35249845873dbe2879ef57aeaa56bd507d333))
- Fix typing issues in octree.py ([68122e9](68122e95d59b39e24ccff9214584c923f4a4621d))
- Updated pixi.lock file ([7802ebe](7802ebe5992f05f9ada00f7faba369319efa2610))
- Update octree(node) typing in tree_vis ([1ae0609](1ae06097844cd24c1279f46259e59f5f01eb4409))
- Switch to memory-mapped bytes array for packed output ([1ae2332](1ae2332bf01f75dc7028e2065a25014d86139012))
- Provide both vertex testing and projection testing ([0ab0cf7](0ab0cf7104a8283b7e438c63db70037ce51c8b24))
- Switch to memoryview and array casting ([3e915e0](3e915e02647d1fbeaf590eb40ca686746316c18e))
- Unroll get_box_vertices loops ([a6344e2](a6344e2498bf54fe845fac97f718bb5cd09cf9a2))
- Fix typing for mypy, use filter ([7a2c70c](7a2c70cc313d98b4ef96277aaf41e3f7f577b8c1))
- Fix pbar typing complaint ([251fa3e](251fa3ee8d68798ababac908eb394b0f1c978f2f))
- Return box copy ([dcbf932](dcbf932c929ba180b872b716f55e3b4f7e7f48d5))
- Scalene, pympler ([ca1d7cb](ca1d7cb9d1906e6aa12505824b601d251fb7f8fa))
- Changes due to mypy ([a5eb6b3](a5eb6b3c97939e0ddbb7fa8297adbc9dae4f2d6f))
- Changes from increased linting levels. ([b0bab0b](b0bab0b616767810159afa04c51836be46aca71e))

### 📚 Documentation

- Added/updated PackedTree documentation ([6994bc5](6994bc5eea76d09021bf663f4da17317431c5588))
- Fixes and updates. ENH: Minor internal API change ([32a742c](32a742c493be45e7b073fa56443f03b76483d432))

### ⚡ Performance

- Remove unnecessary searches, unused variables ([ec67196](ec67196e246ef81da96a9838a270a916b5dbfc6e))

### 🧪 Testing

- Skip empty files ([9bda7b7](9bda7b72048eba6c964896a3f2b5b6f91f9b6337))
- Update bounding_box and octree tests to match new APIs ([095f920](095f920eb1916d3bafd5a43f9a3fb485fcbb42c1))

### ⚙️ Miscellaneous Tasks

- Add basic .gitlint file ([5c3a2e2](5c3a2e261d21f06afea804f4a81eabd72b8cfad4))
- Ignore profiler output ([7e302b3](7e302b39745f7ca79257b20f2eb7e44b1cd00b99))
<!-- generated by git-cliff -->
