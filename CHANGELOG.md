## [0.3.0] - 2026-04-07

### 🚀 Features

- Switch to custom heap implementation ([0cabc92](https://github.com/astrosocket/packingcubes/commit/0cabc92a97eebd31b3d0172122f3f50dbcdbb850))
- Update Cubes API for get_closest_particles ([8f8627c](https://github.com/astrosocket/packingcubes/commit/8f8627c73d3b7bbab53f92839558996a7b91f884))
- Add get_closest_particles for Cubes ([234c1d7](https://github.com/astrosocket/packingcubes/commit/234c1d7984b36caaaa824bac3d24cb7f4f62e31b))
- Add clip_to_box method ([261933b](https://github.com/astrosocket/packingcubes/commit/261933b1fe3ca34d7a720b512714154c8bdb1092))
- Change slice handling to use slice lists ([85e8aa2](https://github.com/astrosocket/packingcubes/commit/85e8aa213d4aa483028c97f0555d55ecb3585e17))
- Expose save_dataset if Dataset passed to KDtree ([dcb735f](https://github.com/astrosocket/packingcubes/commit/dcb735fec2b989feb1e0335362a76a27fbabe5d6))
- Updated default workers argument and warning ([79c1d49](https://github.com/astrosocket/packingcubes/commit/79c1d49fdc4b350f3841084a845de5b1444c5cc5))
- Switch to using ParticleCubes as the backing search object ([9368f3f](https://github.com/astrosocket/packingcubes/commit/9368f3ffec410e27ccdebb18cff4524d41e31165))
- Add stubs for closest particle queries and index lists for cubes ([de36637](https://github.com/astrosocket/packingcubes/commit/de366370eac8a72a0d3dceea6191ae9abcfb5fe8))
- Add methods to get particle_index_lists ([6442a16](https://github.com/astrosocket/packingcubes/commit/6442a16e20486323ba23a43ba6fd12810de99c49))
- Add method to save ParticleCubes to file ([e2298ab](https://github.com/astrosocket/packingcubes/commit/e2298ab7ce3abd732ada9cb2294b99ee952094d4))
- Add method to get single ParticleCubes instance ([8eecee6](https://github.com/astrosocket/packingcubes/commit/8eecee65f72b0a8c81d92393641ecac5a5ebaeb7))
- Add _get_indices_in_shape, replace in _sphere, _box ([7f79f2b](https://github.com/astrosocket/packingcubes/commit/7f79f2b4fbaf182f2c04554196010c3f3c734a1f))
- Allow single string particle types when making cubes ([7b7bb19](https://github.com/astrosocket/packingcubes/commit/7b7bb1944aa9de10c304b3acd5aefc0f9594ebbe))
- Add a pass to remove empty cubes ([c1e2c2b](https://github.com/astrosocket/packingcubes/commit/c1e2c2b3a4ebab75aa98012c89d170607caca164))
- Add PackedTree __len__ as number of particles ([34c2c87](https://github.com/astrosocket/packingcubes/commit/34c2c87d2cf9f865baea0728a716318f39556f83))

### 🐛 Bug Fixes

- Forgot to increase scale ([8c5bb15](https://github.com/astrosocket/packingcubes/commit/8c5bb150c2a6c9e98236b45392b2617cb8afdffc))
- Nodes were changing the tree box ([202cd07](https://github.com/astrosocket/packingcubes/commit/202cd07a5a2eab5e5ecebe426b9e3d749dd5523d))
- Handle passing None to particle_threshold ([3bd6629](https://github.com/astrosocket/packingcubes/commit/3bd6629bbf18689d61f0c8bb35bd7bbdf1b30b50))
- Some keys not populated yet in incremental saving ([b1ca1a5](https://github.com/astrosocket/packingcubes/commit/b1ca1a581b4ad70a2b6272a6b51675a6ed33e0d6))
- Add default value for number_balls ([22410d8](https://github.com/astrosocket/packingcubes/commit/22410d83eba5de2dd983135d8e10659b88c17f19))
- Bounding_box setup after position loading ([202e658](https://github.com/astrosocket/packingcubes/commit/202e6585b34d55ab3b41ea35588de34c18f9164c))
- Explicit type conversion, move objmode out of loop ([0f6fea0](https://github.com/astrosocket/packingcubes/commit/0f6fea057eaa6ed8d3643aabdeb755fc100bc991))
- ParticleCubes don't have particle_type, clean up warning ([cc3b6d8](https://github.com/astrosocket/packingcubes/commit/cc3b6d8bf434ff0db68994fd6c278ab9283bc198))
- Incorrect box sizes passed to tree construction ([520000f](https://github.com/astrosocket/packingcubes/commit/520000f14472ddc7b167d7c6c4d4b8a841111f90))
- More off-by-one errors ([1f74205](https://github.com/astrosocket/packingcubes/commit/1f742055978885617d03f2a3491f0cca4b045ad0))
- Actually sort in place ([0837807](https://github.com/astrosocket/packingcubes/commit/0837807541424b52c67725ac6625d80d5a125505))
- Leaf root nodes should still check partiality ([f08e3db](https://github.com/astrosocket/packingcubes/commit/f08e3dbd7899935fe61300d4be04748fac4dbc4d))
- Off-by-one error on node length ([c1af9e2](https://github.com/astrosocket/packingcubes/commit/c1af9e2432c89581f8182089c66648281ff1d407))
- Dumb off-by-one error ([870315f](https://github.com/astrosocket/packingcubes/commit/870315f78c65e43c5112a2833fbde35d3f3c36af))

### 💼 Other

- Fix another parallel list index casting ([ed3f038](https://github.com/astrosocket/packingcubes/commit/ed3f0386cbc3883ec7e2b257dcfabf00654b8f0e))

### 🚜 Refactor

- Move boxsize processing to separate function ([b32fd92](https://github.com/astrosocket/packingcubes/commit/b32fd92f54e8033579897b41b748e9026c913352))
- Make variable names more consistent ([5531354](https://github.com/astrosocket/packingcubes/commit/553135414765df435a84387862f060cd79d17930))
- Change Cubes class to MultiCubes, add factory functions ([01bc347](https://github.com/astrosocket/packingcubes/commit/01bc347f4644689f5d91f2d60873ad5a6740d8d8))
- Use self._get_indices_in_shape instead _get_indices_in_shape ([7cb35ba](https://github.com/astrosocket/packingcubes/commit/7cb35ba39329351bee1b3a42d6cfb3cf9484bccc))
- Remove _big_index_tuple type ([849cf90](https://github.com/astrosocket/packingcubes/commit/849cf9025b5b95b738eb08d11f6d875403c1bb3c))
- Change order of methods in file ([efe7069](https://github.com/astrosocket/packingcubes/commit/efe7069fce39337fcf56094ce253f3278f46d0e8))
- Change output from list to array ([cd7fd50](https://github.com/astrosocket/packingcubes/commit/cd7fd50ded41a942a47736754dba299e499b1df2))
- Remove unnecessary bounding_box ([6ac9a7c](https://github.com/astrosocket/packingcubes/commit/6ac9a7c780d18a936e57a66d4ba4a57ece8e0ab4))

### 📚 Documentation

- Add query performance timings ([0201e7d](https://github.com/astrosocket/packingcubes/commit/0201e7dca67bbe898c84486a8cdcae85da9f081f))
- Formatting, update timings, switch to creation, improve scale plot ([3c5f347](https://github.com/astrosocket/packingcubes/commit/3c5f347fab42a2ad27cf11d1db30d04a9f0fe9b4))
- Fix description of workers in qbp ([732a599](https://github.com/astrosocket/packingcubes/commit/732a5994aa0674e965d0ea70f602e1e5b05bd121))
- Clean up _get_indices_in_shape ([083fafb](https://github.com/astrosocket/packingcubes/commit/083fafb13529e48b92e930f01df51e0adb843896))
- Add missed changelog addition ([c19cdc4](https://github.com/astrosocket/packingcubes/commit/c19cdc4a09f669c41725ad0a32c724bc739dcaa7))

### ⚡ Performance

- Rework get_closest_particles to use min-heap ([54afcdd](https://github.com/astrosocket/packingcubes/commit/54afcddcb6937e649e5c74ec7061f3bc588e0dd5))
- Updated timings showing strong scaling ([1dcd768](https://github.com/astrosocket/packingcubes/commit/1dcd7686a8e66ffa5718bb8510c811eb22a5a2f8))
- Change query_ball_point defaults to performant values ([45b7521](https://github.com/astrosocket/packingcubes/commit/45b75214df4287a6e88fdcaba4ce2aa9bba45fc0))
- Subboxes should probably be safe ([0bb1aee](https://github.com/astrosocket/packingcubes/commit/0bb1aee130ae7038269fd673d330380b77b54e32))
- Skip extraneous function call ([723bf13](https://github.com/astrosocket/packingcubes/commit/723bf13103f88889135566669ff3d8aace32cac7))

### 🎨 Styling

- Supress additional enumeration warning ([efac873](https://github.com/astrosocket/packingcubes/commit/efac8735d6e026b29b0f41cbee98169a00b997fd))
- Suppress complaint about enumerating for loops ([fef8c84](https://github.com/astrosocket/packingcubes/commit/fef8c842ef2cdf321eb6c3112293a3d81ce32c2f))

### 🧪 Testing

- Update query test to use config k, return sorted ([7942f96](https://github.com/astrosocket/packingcubes/commit/7942f966ab3f7359156ebc94e88580cbfe6f1483))
- Add performance options for query timing ([e8831ee](https://github.com/astrosocket/packingcubes/commit/e8831ee6b17570be58212a69aec2b7c136248fca))
- Kdtree query tests are working again ([d4359e6](https://github.com/astrosocket/packingcubes/commit/d4359e603d294be4d954251c6bf1d6a4d6d489e0))
- Stop saving dataset changes, API updates, cube profile update ([eec90ae](https://github.com/astrosocket/packingcubes/commit/eec90ae616c7033fc00f3a2b21e14df49527b3b7))
- Switch to custom timer for search objects ([7a039b2](https://github.com/astrosocket/packingcubes/commit/7a039b287d54f7ee44c7b31ae539a67ffa9410f0))
- Update profiling notebook ([cfe673f](https://github.com/astrosocket/packingcubes/commit/cfe673ffb8636223781519b828c0a1544302e837))
- Update benchmarking data ([c5a47dd](https://github.com/astrosocket/packingcubes/commit/c5a47dd2c2ac77adbd8b635f0bea8323f1a9383f))
- Save output on every loop iteration for incremental results ([10571af](https://github.com/astrosocket/packingcubes/commit/10571aff35786399ba7a02b8f561c80a7be99904))
- Add extra debug information ([eee95a0](https://github.com/astrosocket/packingcubes/commit/eee95a0777908605eec6de387817d2aa0737526c))
- Use new slice lists to load chunks of dataset instead of striding ([a1be6db](https://github.com/astrosocket/packingcubes/commit/a1be6db72e50161608afd8a0f5946a7529fafb7e))
- Add test for just the numba search portion ([eb9f60e](https://github.com/astrosocket/packingcubes/commit/eb9f60e1b33c3676e99907d709420b2b96925b4b))
- Add custom scaling to tests ([e5c45d8](https://github.com/astrosocket/packingcubes/commit/e5c45d82929580f02ef67ec8f7cce6c607de9cb3))
## [0.2.3] - 2026-03-25

### 🚀 Features

- Adding loading factor ([29f05af](https://github.com/astrosocket/packingcubes/commit/29f05afd3b16cd9591a0f16d9b147e5b214be124))
- Allow multiple dfs/sbs and save to file ([c5f0e54](https://github.com/astrosocket/packingcubes/commit/c5f0e5406bccdd18adf4af4be68594ad94369dbd))
- Allow dataset as argument to KDTree ([944540d](https://github.com/astrosocket/packingcubes/commit/944540d2c61ad0242dd17325aed32a98423db9fa))
- Add auto thread based default cubes_per_side ([c24f39b](https://github.com/astrosocket/packingcubes/commit/c24f39b016fed12a73b270ff8049ef1793263d11))
- Expose save_dataset flag, make output optional ([bbbd234](https://github.com/astrosocket/packingcubes/commit/bbbd2346fcd0cc4584842fe3c81dd04df0d94f7b))

### 🐛 Bug Fixes

- Assuming search_ball_sizes even when none ([1e6a018](https://github.com/astrosocket/packingcubes/commit/1e6a01808acda955b7b7a0fe8769ed562ac22b60))

### 🚜 Refactor

- Remove empty sidecar file header creation ([668a914](https://github.com/astrosocket/packingcubes/commit/668a914980a6fb1b0a7f0ccef5af5c903c3f379f))

### ⚡ Performance

- Move cubes reordering into parallel loop ([857295b](https://github.com/astrosocket/packingcubes/commit/857295b25257727f9a199ce42eae9c06dc9a8447))
- Add unsafe BoundingSphere creation ([af01b30](https://github.com/astrosocket/packingcubes/commit/af01b303608eb3c2d5d5afa55a8dc114103139df))
- Combine search methods and use single shape creation ([75985e9](https://github.com/astrosocket/packingcubes/commit/75985e99e4d270f69e4dae052360e4d1812f7bb6))
- Switch to check_overlap ([68edfd6](https://github.com/astrosocket/packingcubes/commit/68edfd6845249aa80fc79965ad4f81c4a20f6dfb))
- Inline _cubes_position, remove extra calculation ([83b51e7](https://github.com/astrosocket/packingcubes/commit/83b51e7a50ca027d4360d4995938965bd33b668a))
- Use explicit bounds in box references ([cac5169](https://github.com/astrosocket/packingcubes/commit/cac51691fdf7f61e360e665c59eb15e11d13f8e5))

### 🧪 Testing

- Number of bugfixes ([fcd5af2](https://github.com/astrosocket/packingcubes/commit/fcd5af21b718367f6e5dbc6cc1fa53f1eebe7e6b))
- Refactor timing_test to use fewer globals, general improvements ([b1e66b8](https://github.com/astrosocket/packingcubes/commit/b1e66b888644bad32c8de10390876eecca20a12e))
- Handle error in timing gracefully ([c4df754](https://github.com/astrosocket/packingcubes/commit/c4df7544d08649f49e6e7d5499ab9f2341e4124c))
- Switch to faster query_ball_point options ([340d8a3](https://github.com/astrosocket/packingcubes/commit/340d8a3d63c907924638662ef8104a4f8c779cbb))
- Display number of threads if cubes testing ([d55daa8](https://github.com/astrosocket/packingcubes/commit/d55daa8d61462674ea6876ac27828cc013662dc8))
## [0.2.2.2] - 2026-03-20

### ⚙️ Miscellaneous Tasks

- Add twine to build group, specify markdown ([#13](https://github.com/astrosocket/packingcubes/issues/13)) ([ed349a5](https://github.com/astrosocket/packingcubes/commit/ed349a51c4fc36099aa09c33d856ed7a63d1ce9f))
## [0.2.2.1] - 2026-03-20

### ⚙️ Miscellaneous Tasks

- Add twine to build group, specify markdown ([a33903e](https://github.com/astrosocket/packingcubes/commit/a33903e9f4d228a35046cc7bcef8012aa8f52d2f))
## [0.2.2] - 2026-03-20

### 🚀 Features

- Enhancements derived from updating the README ([#12](https://github.com/astrosocket/packingcubes/issues/12)) ([256877c](https://github.com/astrosocket/packingcubes/commit/256877c79eb08fcc1dc6ee54d138c1ad67687a45))
- Flesh out the readme ([e757b45](https://github.com/astrosocket/packingcubes/commit/e757b4586b2c79fe11e2152f70019a567f367d01))
- Support passing a regular array to Cubes constructor ([c59097b](https://github.com/astrosocket/packingcubes/commit/c59097b8eb844dda5be5510125b7b4c487e7c54c))
- Allow loading cubes data from file ([d45b5d7](https://github.com/astrosocket/packingcubes/commit/d45b5d70ee30975a870f75ee6ee348e745fafa37))
- Allow optional particle type when getting indices ([17084be](https://github.com/astrosocket/packingcubes/commit/17084be15b8d69830f378ce69f4c01e2db05037f))
- Allow specifying a string or Path when saving cubes ([1674664](https://github.com/astrosocket/packingcubes/commit/1674664e8e41dfe4739b3af796160a8018767ef3))
- Add CLI option to specify where cubes should be saved ([823d4fe](https://github.com/astrosocket/packingcubes/commit/823d4fe6216e833a5c9b2b4c56aa1bd539302b5f))
- Support more general BoxLike for cube_box argument ([a4b8545](https://github.com/astrosocket/packingcubes/commit/a4b8545f248dbe67039dda145697fedb6291a9b6))
- Add flag to save datasets in make_cubes ([5e16e98](https://github.com/astrosocket/packingcubes/commit/5e16e987e47abca74e6a214022920372d14b124b))
- Add method to save out sorted positions and indices ([cdc9be2](https://github.com/astrosocket/packingcubes/commit/cdc9be26c6fc7d2cb3a6fa8e72cee5968a8939da))
- Allow 1 & 2D data for KDTrees ([01f2828](https://github.com/astrosocket/packingcubes/commit/01f2828a391baf27f212be04c23ff38fe9200ae3))
- Expose the InMemory class ([f870ccd](https://github.com/astrosocket/packingcubes/commit/f870ccd970323969efefe3b02c10a9c3ad9b0ab0))

### 🐛 Bug Fixes

- Remove old view of data when loading new positions ([b8be47c](https://github.com/astrosocket/packingcubes/commit/b8be47cbea55dd0bbeba98eb4478623b62b3ce52))

### 💼 Other

- Fix argument typing ([c913ffb](https://github.com/astrosocket/packingcubes/commit/c913ffb844ec98bcdc4ceb58737908a0b68a6377))
- Fix typing complaints from mypy ([4736c03](https://github.com/astrosocket/packingcubes/commit/4736c03b370bc58d29ddf3b1ffa904d285d417ab))

### 🚜 Refactor

- Remove yt timing ([243d15f](https://github.com/astrosocket/packingcubes/commit/243d15f4d5c3ddb6f3890fbe210d5f393ed6326c))
- Remove more Dataset-Cubes interdependence ([7778fb7](https://github.com/astrosocket/packingcubes/commit/7778fb770653868a747285f4f479d4aec7e80f05))
- Remove the cache file for HDF5Datasets ([d33987b](https://github.com/astrosocket/packingcubes/commit/d33987bebc74372def420a116025844aa2e6d3ab))
- Remove some missed debugging print statements ([24d5ebd](https://github.com/astrosocket/packingcubes/commit/24d5ebd6a51ee39743182f1766beb6eaa25b1b3b))

### 📚 Documentation

- Add docstring to _get_particle_indices_in_shape ([60b9fac](https://github.com/astrosocket/packingcubes/commit/60b9facd1b904817a8420283070dc8debf0b4db8))
- Add missing docstring to make_cubes ([59b4be6](https://github.com/astrosocket/packingcubes/commit/59b4be626fe514973d5806341575435216f4aed1))
- Add _data_container to attribute list ([c087267](https://github.com/astrosocket/packingcubes/commit/c0872671595523642590da3a860b891650b4bcf7))

### ⚡ Performance

- Skip loading particle data if not actually changing particle_type ([e4476d3](https://github.com/astrosocket/packingcubes/commit/e4476d376f3dde3eb4d6bec1f28987d8e1bff79b))

### 🧪 Testing

- Allow 1 and 2D arrays for KDTree ([02c53d8](https://github.com/astrosocket/packingcubes/commit/02c53d88d3a3a720de50999430fdaacf2a8f557e))
## [0.2.1] - 2026-03-19

### 🐛 Bug Fixes

- Fix typo ([547cd0a](https://github.com/astrosocket/packingcubes/commit/547cd0ad8fbe1390787ada6a2844d530bf3a6051))

### ⚙️ Miscellaneous Tasks

- Readd pypi publishing ([1d374ab](https://github.com/astrosocket/packingcubes/commit/1d374ab11ea3aad609c8002e4ec54a02b6f9885c))
- Move scm variable ([9cd97e5](https://github.com/astrosocket/packingcubes/commit/9cd97e5d0eef020d4dbf67b2f4b496bca23f8b35))
- Attempt testpypi publishing ([47e169c](https://github.com/astrosocket/packingcubes/commit/47e169c7ecbb85e4b618f6bcfd11963c0a215c34))
- Set fetch-depth, remove local version component ([b071880](https://github.com/astrosocket/packingcubes/commit/b071880f730e591f6be6cc48ac864ae09e00e3bb))
- Fix version export, add setuptools_scm ([fb4e24b](https://github.com/astrosocket/packingcubes/commit/fb4e24b156ed5e2b1db291819877c77fc6d9b7e9))
- Print version and turn of upload temporarily ([2993a11](https://github.com/astrosocket/packingcubes/commit/2993a11ebcd944df6b10b58ba48ae46d612309bc))
- Forgot to add pixi run ([ccf9e5d](https://github.com/astrosocket/packingcubes/commit/ccf9e5d7043885cf106df18e9c2b47a01edd9d84))
- Switch to pixi install in pypi pipeline ([8a9e35f](https://github.com/astrosocket/packingcubes/commit/8a9e35f541a41e57ec13eedc67fb864071fe1d51))
- Switch to runtime version checking ([3130d73](https://github.com/astrosocket/packingcubes/commit/3130d7320e808a91b409eebdb2a86ccb992c96d7))
- Add install step ([a458f0f](https://github.com/astrosocket/packingcubes/commit/a458f0fa0433d5790f406ca29202d867f030e5a2))
- Update workflow to remove the frozen status and update lockfile ([9a21b96](https://github.com/astrosocket/packingcubes/commit/9a21b966afa59327b9e7ba5dd14264996f449cda))
- Add verbosity to publishing to help resolve error ([062eb10](https://github.com/astrosocket/packingcubes/commit/062eb10805a33bca1de9f9c5e9983cdd6903619c))
## [0.2.0] - 2026-03-18

### 🚀 Features

- Expose preliminary count_neighbors ([eede96d](https://github.com/astrosocket/packingcubes/commit/eede96dfd2ab19ef6761a50b2b14733ecd83b5b8))
- Add remaining KDTree function stubs ([b418385](https://github.com/astrosocket/packingcubes/commit/b41838519b6eb9e4ca348df59ad072ecbac18d08))
- Tack particle numbers and test spheres are being correctly searched ([e4b38ee](https://github.com/astrosocket/packingcubes/commit/e4b38ee37f3a0f7ad5ec621875f0c90a86f53bde))
- Add query_ball_tree method and test ([4087034](https://github.com/astrosocket/packingcubes/commit/408703471e5eec4268fc824e36404417a1ca9112))
- Expose the dataset's shuffle list for usage outside the tree ([6b57305](https://github.com/astrosocket/packingcubes/commit/6b57305e915bc40db701ded8e9a04659c5cd9927))
- Add _get_pilis_tree methods ([b8c744b](https://github.com/astrosocket/packingcubes/commit/b8c744b882526e18cdced023c8026e5c76774a77))
- Add some useful methods to CurrentNode ([ad146ec](https://github.com/astrosocket/packingcubes/commit/ad146ec347d66dc8d69da0bc1454180961286fe0))
- Allow specifying a DataContainer instead of Dataset for PackedTree ([f668ef2](https://github.com/astrosocket/packingcubes/commit/f668ef230c2e10a65f35bbca0d7f2f08a70bc8e7))
- Add constant particle number search capability ([3da4433](https://github.com/astrosocket/packingcubes/commit/3da4433d8d729795d896f6dd123f1293e3d9a878))
- Add option to return lists instead of arrays ([1a8a6fe](https://github.com/astrosocket/packingcubes/commit/1a8a6fe2eb65e2c916c0c8ce4ec245d73c15e4d1))
- Export KDTree ([e6813be](https://github.com/astrosocket/packingcubes/commit/e6813be904c0d08cc9273c58438e724d1ad8d232))
- Add query method for KDTree API ([60344e0](https://github.com/astrosocket/packingcubes/commit/60344e04e4539da23b9f230245aee8b80aa1e9bb))
- Add get_closest_particles function as analog to KDTrees query ([d64849c](https://github.com/astrosocket/packingcubes/commit/d64849ccff80571112e172561da06639ec76fd3e))
- Allow strict query_ball_point and set as default ([364d2e2](https://github.com/astrosocket/packingcubes/commit/364d2e2b3c0c934b7cf8fb1bd1ef19a138d92d22))
- Add query ball point method to KDTree API ([4326a6c](https://github.com/astrosocket/packingcubes/commit/4326a6c94547e213672c56aac21cd9e285ef2ec2))
- Add first pass at some of the KDTree API ([96e2e9e](https://github.com/astrosocket/packingcubes/commit/96e2e9e6379afddcbb0910127903c7ad00c38ce5))
- Add benchmarking and profiling notebooks for better remote access ([6e7e102](https://github.com/astrosocket/packingcubes/commit/6e7e102fc27bb731d95c1baa1eeca422778a6014))
- Allow spcifying number of search balls ([db50290](https://github.com/astrosocket/packingcubes/commit/db50290fdff46535ef7b644e7ce628e017196b7a))
- Improve 'not a box' help ([f2c7cff](https://github.com/astrosocket/packingcubes/commit/f2c7cff69fc619210de44532489a46b63a239515))
- Add version information to timing_tests.py ([54a5d76](https://github.com/astrosocket/packingcubes/commit/54a5d76aa3d40c7b2057172f4c20f012b8ed6dd7))
- Add version information to main __init__.py ([c8d152a](https://github.com/astrosocket/packingcubes/commit/c8d152a2b4db91d06ac5c4e56c2661004fd35e59))
- Add timing test for querying with full index list return ([1e43f5b](https://github.com/astrosocket/packingcubes/commit/1e43f5b155748d4cb815a1a5e6c9daa74396dbf5))
- Add strict particle index list search ([3d6d98e](https://github.com/astrosocket/packingcubes/commit/3d6d98ef5735a460509963e61264d779c7fb7b28))
- Implement cubes saving and loading to arbitrary HDF5 dataset ([e38ddb1](https://github.com/astrosocket/packingcubes/commit/e38ddb176955d271e74a9eea12eb09442c3d7465))
- Add new plots, clean up Basic_Usage ([2531302](https://github.com/astrosocket/packingcubes/commit/25313025e53d67cf3a4ac6efe62b24bafb7621ab))
- Support MultiParticleDataset since no HDF5 features used ([2b19fa0](https://github.com/astrosocket/packingcubes/commit/2b19fa0be3430b76c67012141395b87d8a139cee))
- Add default arg values to make_cubes, return PackedTrees ([f1b6c0d](https://github.com/astrosocket/packingcubes/commit/f1b6c0dc68547a6ec9b42ce6c7ffe3ffbe041fa0))
- Enable PackedTree construction from NDArrays ([e228268](https://github.com/astrosocket/packingcubes/commit/e228268d25f243a7a46cddc44fdd6419bb510756))
- Add new abstract MultiParticleDataset and InMemory dataset ([4e6dca7](https://github.com/astrosocket/packingcubes/commit/4e6dca7ffd5c1fc528fb7cc9d6369ca25556ca6f))
- Add PackedTree metadata ([61147c9](https://github.com/astrosocket/packingcubes/commit/61147c9794035d3ba3b4ffd75e36c4561490d4a5))
- Add xxhash to the dependency list ([31e756b](https://github.com/astrosocket/packingcubes/commit/31e756b9102ee404abb618545e45db70862692e9))
- Remove unnecessary dataset field from PackedTree(Numba) ([651dbb8](https://github.com/astrosocket/packingcubes/commit/651dbb8aed27e036992d2077a809c0fe815f5600))
- Add from_packed stub for eventual round-trip testing ([ef43608](https://github.com/astrosocket/packingcubes/commit/ef4360811cdeef281979977514302aaaa68897a3))
- Add second pass at cubes implementation ([554efb2](https://github.com/astrosocket/packingcubes/commit/554efb26a7296661ecd1aaf4b7580e7ac478f8d6))
- Convert _construct_tree to jitted function ([8603c4f](https://github.com/astrosocket/packingcubes/commit/8603c4f8fb26d0fbdc6d7941b619d9142697d6e6))
- Add method for getting subportion of data ([57a2f0a](https://github.com/astrosocket/packingcubes/commit/57a2f0affbf8f5f2cac665b977a749dfc19d1332))
- Add method for imposing 'external' particle order ([49ea645](https://github.com/astrosocket/packingcubes/commit/49ea645e9eab82b48b015ba61c3328c9b52f1c8e))
- Add methods for accessing particle numbers ([b097f53](https://github.com/astrosocket/packingcubes/commit/b097f5345d99526840568939e4657c3e9933f399))
- Add packingcubes level logger explicitly ([cd4764b](https://github.com/astrosocket/packingcubes/commit/cd4764b7e7108f54f7c64a7840120b2e372db97a))

### 🐛 Bug Fixes

- Missing character in repository-url ([2e13f53](https://github.com/astrosocket/packingcubes/commit/2e13f53e9b2808e76dfaa4f194d1924a6007c8df))
- Invert typo in publishing workflow ([35d2815](https://github.com/astrosocket/packingcubes/commit/35d281532a14799e81ef1c6b6061a66d2c9bac81))
- Fix typo in publishing workflow ([8a99a56](https://github.com/astrosocket/packingcubes/commit/8a99a56c8639e214e2f9eb1683f51843293d63ad))
- Jitter=0 not being set correctly ([63fce23](https://github.com/astrosocket/packingcubes/commit/63fce233a11b4f58203c7b20bccf25ddad214f63))
- Used < instead of <= ([4c44186](https://github.com/astrosocket/packingcubes/commit/4c441863d32ff5d3ac05ca1ddfc7569bb700ab9e))
- Move per node behavior to subfunction ([1da6e20](https://github.com/astrosocket/packingcubes/commit/1da6e20a3d766a3beedf38862ac9e5b2c30778b6))
- Missing guard against NxMxOx... arrays ([6767bc7](https://github.com/astrosocket/packingcubes/commit/6767bc7377e00eb108802e641140fd7ff3c800b7))
- Missed case where obj overlapped with root leaf but not fully ([005955d](https://github.com/astrosocket/packingcubes/commit/005955dcaeb0f9d9ed282fd53c4e54402230f38f))
- Missed updating kadtree->scipy_kdtree ([9cf7ae1](https://github.com/astrosocket/packingcubes/commit/9cf7ae16eea470606744e6c92b28b8b924d660d2))
- Fix cubes calls in profiling ([fb309fc](https://github.com/astrosocket/packingcubes/commit/fb309fcb1bb6b007ff945be9a3daf0a4023a4385))
- Strict length returned is not allowed ([6d79d64](https://github.com/astrosocket/packingcubes/commit/6d79d64ea399bfee30294cd387abc5c1dc4605f7))
- Fix typo on output order ([8676b43](https://github.com/astrosocket/packingcubes/commit/8676b437ed2468eee0c3590d6516f502ef81e88f))
- Correct return_sorted logic ([97182fc](https://github.com/astrosocket/packingcubes/commit/97182fc0eb3f4aedc95994e08e6d43c5902b0a89))
- Fix off-by-1 error ([c32b41b](https://github.com/astrosocket/packingcubes/commit/c32b41b4531f87232a5fa31dd7546d95536519d9))
- Fix get_closest_particle_bugs ([653729a](https://github.com/astrosocket/packingcubes/commit/653729abb006baf4469416a2a96f0331932fb975))
- Fix closest_particles bugs ([ba56c4d](https://github.com/astrosocket/packingcubes/commit/ba56c4d2db4225eee19602f4d249fde54155c0cb))
- Fix kwargs calls in jitclass function ([a853d77](https://github.com/astrosocket/packingcubes/commit/a853d77cd9425a7191643a4095930360fb0bcdab))
- Fix incorrect output type for Sequence and off-by-one error ([16cf84c](https://github.com/astrosocket/packingcubes/commit/16cf84c9ade09196c2ea54c965d4164742059284))
- Fix incorrect len call ([cf0e480](https://github.com/astrosocket/packingcubes/commit/cf0e4805421aa198eeae64a441f5efad2c9863af))
- Check x is 3d properly ([a0ab478](https://github.com/astrosocket/packingcubes/commit/a0ab4787b0a1387a333eecb90024ff86c4dc89aa))
- Fix typing ([a165e79](https://github.com/astrosocket/packingcubes/commit/a165e7997a8cb982cfbbe8078483386caa5ab85e))
- Fix typing for p ([7ef0e16](https://github.com/astrosocket/packingcubes/commit/7ef0e16d17b1ae9094a1ff6d33570aa4c1bc6afa))
- Correct return type ([63723d4](https://github.com/astrosocket/packingcubes/commit/63723d4e83136daeb2b55238c0c772de8afc0806))
- Handle boxsize is none ([89107a3](https://github.com/astrosocket/packingcubes/commit/89107a347d5f52c86635a79d474e3a26cc4c7e5e))
- Fix error message generation ([3c9d3bd](https://github.com/astrosocket/packingcubes/commit/3c9d3bd678739d78e0d3b2c564d83e5985b4da4a))
- Include check for under/overflow, refactor floating point test ([bb32909](https://github.com/astrosocket/packingcubes/commit/bb32909b214ebc929b03f209d0bf38e155603159))
- Missed KDTree added to __all__ ([a391c5f](https://github.com/astrosocket/packingcubes/commit/a391c5f5658a69b522cfe3243d0561d508cbbf39))
- Remove KDTreeAPI import ([c61b4c6](https://github.com/astrosocket/packingcubes/commit/c61b4c6440aa3df06121b4631849cf4ef211b958))
- Never updated PackedTree constructor ([34c20a7](https://github.com/astrosocket/packingcubes/commit/34c20a7cdaf5dc33b8952fb6aa95674c9609d616))
- Remove duplicatedindices ([76d195f](https://github.com/astrosocket/packingcubes/commit/76d195f7d93b27c238b99bc19bd1fe14753eca03))
- Add warning about threading layer ([4cc9616](https://github.com/astrosocket/packingcubes/commit/4cc96163580f558edfade32e5ea0845b38af950f))
- Missed combined_list could be None ([d31c197](https://github.com/astrosocket/packingcubes/commit/d31c197bb62d4c132b9bc77b855865108c880896))
- Switch to InMemory so cubes timing respects decimation factor ([2814e65](https://github.com/astrosocket/packingcubes/commit/2814e65c27b6917678fe9999b8ac11312fd0f4dc))
- Forgot to actually use args ([003b73b](https://github.com/astrosocket/packingcubes/commit/003b73b35bb89b896d78e1977ecdb1469f368671))
- Fix getattr instead of get, missing kw, number not set ([38c7f28](https://github.com/astrosocket/packingcubes/commit/38c7f28879c5c29628f4a8b59798bf201018b476))
- Add missing particle_types argument ([9d426ac](https://github.com/astrosocket/packingcubes/commit/9d426acf07efe2cd2eeddc5ce12d00dee7dc8b4e))
- Explicit cast to intp to remove type warning ([65f940f](https://github.com/astrosocket/packingcubes/commit/65f940f8fed2cfe165a228f41d34344824f129b4))
- Missed typing annotation leading to typing warning ([fa3c05e](https://github.com/astrosocket/packingcubes/commit/fa3c05eb61455746b526566b137486f039d8299d))
- Only save cubes to HDF5 datasets ([f462faa](https://github.com/astrosocket/packingcubes/commit/f462faa718810a438385477e12a27a38ce8eb995))
- Enforce providing Nx3 arrays to InMemory ([cafe1c5](https://github.com/astrosocket/packingcubes/commit/cafe1c5bc451852d1b4eb7fdc03729121f243111))
- Fix InMemory not setting up index. Allow 1D single point array ([8aca494](https://github.com/astrosocket/packingcubes/commit/8aca4945e5be8160e88229f0d674863057c0936b))
- Fix typing so _get_particle_indices_in_shape actually works ([909922c](https://github.com/astrosocket/packingcubes/commit/909922c0fdcae4a7438fc508f1a09a1ef9f833d6))
- _process_box should accept Dataset ([bdc7013](https://github.com/astrosocket/packingcubes/commit/bdc7013363cfcb207666a8833e9467d89aeaf989))
- Ensure empty dataset produce valid node_end ([b88586a](https://github.com/astrosocket/packingcubes/commit/b88586a084614cc198ab4b4b221df3a534514120))
- Fix typing issues revealed by working logger and mypy ([7edb5d3](https://github.com/astrosocket/packingcubes/commit/7edb5d3f498abc6f435d22e5a5eaa02cc8c75548))
- Fix duplicate index setup ([d5baed5](https://github.com/astrosocket/packingcubes/commit/d5baed5c7bbc013caad28b17d341991144536a6b))
- Fix node search starting from root children instead of root ([595b4f7](https://github.com/astrosocket/packingcubes/commit/595b4f7ac82ac426259722f1eb149d9b738a30c0))
- Allow for running without numba ([eb1545a](https://github.com/astrosocket/packingcubes/commit/eb1545acdc8efac3efe476b2ba3b5ce88846a0ac))
- Up position precision ([122abdd](https://github.com/astrosocket/packingcubes/commit/122abdd6de1efe2bec32131369d7c044e3a0038b))
- Move and update set_bounding_box to Dataset ([1a7abc0](https://github.com/astrosocket/packingcubes/commit/1a7abc03a6cb0d4ce872e6a09aea07716ba8a6a4))
- Tighten valid box precision check ([7a48b0d](https://github.com/astrosocket/packingcubes/commit/7a48b0de2666a39213e183ce3f7012e7819da616))
- Remove extraneous @njit decorator on full_morton ([d87f3a1](https://github.com/astrosocket/packingcubes/commit/d87f3a170b2c933e992cd03c7245127e1ecaad25))

### 💼 Other

- Fix typing for project_point_on_box ([d464ac9](https://github.com/astrosocket/packingcubes/commit/d464ac90e50c2d1a94c26d25eed8c6ed87d99d60))
- Fix output typing when returning lengths ([126262d](https://github.com/astrosocket/packingcubes/commit/126262dc7529d6c2c779ec574ec840e74b2d5f13))
- Remove incorrect ruff suppression hints ([4435d16](https://github.com/astrosocket/packingcubes/commit/4435d16b59726120fb9d821b80e72842099e2cc0))
- Fix output typing ([a8fc803](https://github.com/astrosocket/packingcubes/commit/a8fc803a3542186a13294bbb9612b013068b6578))
- Add list of tuple index type ([ce99d2f](https://github.com/astrosocket/packingcubes/commit/ce99d2f64bdeab7b75b468069df3cadb81d039cf))
- Seperate variables for typing check ([ae9f667](https://github.com/astrosocket/packingcubes/commit/ae9f6671ecf5f6ce170b0e1fa344323d61d7c29d))
- Creating temporary current release ([09bade4](https://github.com/astrosocket/packingcubes/commit/09bade48ee678f2973e7fb999efda2792597e3c7))

### 🚜 Refactor

- Need to enable returning both sorted and shuffle indices ([8135b16](https://github.com/astrosocket/packingcubes/commit/8135b162f6b470456a007337c09fa59bc3046ea2))
- Combine strict and non-strict index list functionality ([819780e](https://github.com/astrosocket/packingcubes/commit/819780e514f5a47213c0f44d7bc7527dc487728f))
- Update reset_data to use the reorder method and add typing ([5e4f6cf](https://github.com/astrosocket/packingcubes/commit/5e4f6cf961b78f8f95c6d0c8aa72062e4726c539))
- Rename (scipy) kdtree to scipy in benchmarks ([6cea45e](https://github.com/astrosocket/packingcubes/commit/6cea45e3d712c1f36284dad093189a5a6a61af5b))
- Move x checking to separate function ([67321b9](https://github.com/astrosocket/packingcubes/commit/67321b93be1bdf5bc53533e0c9859ab16a9a91bf))
- Remove get_closest_particle functionality ([67ec3c6](https://github.com/astrosocket/packingcubes/commit/67ec3c65eb34da286215ae34dfe98cefbb01f2d8))
- Move data loading after precompile check ([86bfe31](https://github.com/astrosocket/packingcubes/commit/86bfe317f77a58fc7efd54eeadf2f91b24ce3285))
- Remove unnecessary objmode ([66ce2de](https://github.com/astrosocket/packingcubes/commit/66ce2dea1ca937c454c7a4cc17b6ade5cad44fa7))
- Remove code from packed_tree.py ([247f9a7](https://github.com/astrosocket/packingcubes/commit/247f9a7595e2e0e9abb025a7c3c8f33748380913))
- Split packed_tree into multiple files ([399ae8d](https://github.com/astrosocket/packingcubes/commit/399ae8ddd9c2c88f36b2126e68523b986b482c81))
- Remove translated dataset, it belongs in separate package ([a7c62f4](https://github.com/astrosocket/packingcubes/commit/a7c62f4b087ac59b266d1f07e77e46fe289879ed))
- Rearrange Cubes ([79f5dfe](https://github.com/astrosocket/packingcubes/commit/79f5dfefe57738e47d55222fda8c99ec5f21bf9b))
- Clean up functions that are not and will not be used ([5089ec0](https://github.com/astrosocket/packingcubes/commit/5089ec0548c375eb3852c4c4a9f9cc4c5b61fd19))
- Remove tqdm from octrees and clean up pyproject.toml ([d755a22](https://github.com/astrosocket/packingcubes/commit/d755a22cfa171048cdff62c7172f192ce287a332))

### 📚 Documentation

- Add missing argument definitions ([d8c898b](https://github.com/astrosocket/packingcubes/commit/d8c898b45000be5ec0860c4506472c830551a953))
- Add information on KDTreeAPI constructor and attributes ([ac3f979](https://github.com/astrosocket/packingcubes/commit/ac3f9797d4ac592cbd3e335aaeca690245309175))
- Add missing docstrings ([6b7fdf9](https://github.com/astrosocket/packingcubes/commit/6b7fdf962b5ae9ea0d64f8327892c54306cd1a1e))
- Update docstrings to reflect api change ([cde6097](https://github.com/astrosocket/packingcubes/commit/cde6097a7ebea476e9da094586223cb85ceae20c))
- Add missing/update docstrings ([aced60d](https://github.com/astrosocket/packingcubes/commit/aced60d4d2bd0cdd8f9ccfce82f32ccb4a96703f))
- Update packed tree specification ([d3e7693](https://github.com/astrosocket/packingcubes/commit/d3e7693cd004f3c027686f9037d287cc8185bda8))
- Correct tree data structure ([4c5316a](https://github.com/astrosocket/packingcubes/commit/4c5316af874c2e0b407a07df52080f490de62b36))
- Fix source argument description ([71d74f9](https://github.com/astrosocket/packingcubes/commit/71d74f994d782f00f767929395acdce0b99f5804))
- Improve argument description ([54cb523](https://github.com/astrosocket/packingcubes/commit/54cb523dd011b17000f3eec4f1a78ddf60df7fb3))

### ⚡ Performance

- Update timings, add brute to ignore list ([c3bec3d](https://github.com/astrosocket/packingcubes/commit/c3bec3df4dc5381b67f1a0fcf10e39c306b7e801))
- Change tag list to fixed length array ([082862b](https://github.com/astrosocket/packingcubes/commit/082862bfd051b982d59b5e3048f1b84a5f9f9f6d))
- Use per-node containment information with containment test ([0388aa1](https://github.com/astrosocket/packingcubes/commit/0388aa1008408798a944f6b7bfbfdb983003e473))
- Add check_box_overlap ([5e15b38](https://github.com/astrosocket/packingcubes/commit/5e15b3861be6927be8db058bd921471cb05db511))
- Improve project_point_on_box performance ([3d104cb](https://github.com/astrosocket/packingcubes/commit/3d104cba0ac724d10e7db4d51f13f6f6e48433f2))
- Switch to contains_point/contains_pointlist/count_inside ([904ccaa](https://github.com/astrosocket/packingcubes/commit/904ccaaacc78e29772a701876cfc0c4bedebe7d5))
- Remove astype from data_container call ([25cd26b](https://github.com/astrosocket/packingcubes/commit/25cd26b6dc8e83efd56c8a9a05b046a85e33264d))
- Cache data container property ([9c88f69](https://github.com/astrosocket/packingcubes/commit/9c88f696b3e2621aac70f21be35b40158a44529e))
- Add initial constant number results ([8db2c37](https://github.com/astrosocket/packingcubes/commit/8db2c37e6ff33d958517b9c094edd741e2dd641b))
- Update LB_L10 numbers and notes ([af3e8b6](https://github.com/astrosocket/packingcubes/commit/af3e8b6c18afa402659567d808875ed37732ce67))
- More LB_L10 number updates ([d1f41a7](https://github.com/astrosocket/packingcubes/commit/d1f41a7b8d12c875a05d1f0710316f658ce6bfe6))
- Add cubes creation and search placeholders to the benchmarks ([1bb452a](https://github.com/astrosocket/packingcubes/commit/1bb452a7709355e709f3bc7480778e10624fcdaf))
- Cache search_obj so it's not recomputed for every test ([5da6b1a](https://github.com/astrosocket/packingcubes/commit/5da6b1abf788020f0d2dffff4d24d15184584de1))
- Move BoundingVolume creation out of PackedTreeNumba ([9bb90fd](https://github.com/astrosocket/packingcubes/commit/9bb90fd554151fe14aa6805e86dd5d565c78124a))
- Extract containment testing and position from for loop ([06c129f](https://github.com/astrosocket/packingcubes/commit/06c129fdf45197816535c24a4a57948c6318a506))
- Remove property based calls from octree ([c26cb38](https://github.com/astrosocket/packingcubes/commit/c26cb38871bf82d717f297326d3b9535baa59018))
- Remove property based calls ([f93b08f](https://github.com/astrosocket/packingcubes/commit/f93b08feab5cc9c5ea7a85dd1b4f7d266b5210e5))
- Remove overloaded bitwise count implementation ([2a7616e](https://github.com/astrosocket/packingcubes/commit/2a7616e65504546dcb14508dcbeba7bf8a7c0cc3))
- Improve project_point_on_box ([f62f982](https://github.com/astrosocket/packingcubes/commit/f62f98263b301534e40f48415219c714f2c3db50))
- Unroll some tight loops based on profiling ([dd82d5d](https://github.com/astrosocket/packingcubes/commit/dd82d5dd484855e299c9dcc2158094f5a5f46654))
- Merge Numba changes ([#4](https://github.com/astrosocket/packingcubes/issues/4)) ([ad03775](https://github.com/astrosocket/packingcubes/commit/ad037751cda3014158c3b53ba94fce1d2e99eea1))

### 🎨 Styling

- Clean up some extra print statements ([5766d06](https://github.com/astrosocket/packingcubes/commit/5766d061c0e5754d69c922d41f9d6f468e3c5785))

### 🧪 Testing

- Update project_point_on_box tests ([135264b](https://github.com/astrosocket/packingcubes/commit/135264b6fc62da1bf8872a895ab64c4e9a31a9ce))
- Add query timing ([127aa71](https://github.com/astrosocket/packingcubes/commit/127aa71ffcfba0ecdbc3b292f816019bee3e5ef2))
- Add query to profiling ([7c767af](https://github.com/astrosocket/packingcubes/commit/7c767afe8d47332e581786ceca83a64dfef7ee3d))
- Change packli search to strict for more useful results ([b51dc9d](https://github.com/astrosocket/packingcubes/commit/b51dc9d2cff7a1cb5540f8bccfdffcd069c3050b))
- Ensure KDTree output matches scipy's ([aef7172](https://github.com/astrosocket/packingcubes/commit/aef7172d5eeceabd44c287d8476cdf3ab973b284))
- Switch to scipy KDTree for generating balls. General clean up ([d496398](https://github.com/astrosocket/packingcubes/commit/d49639886f28ac72e9d907704385ccf6e82b00ef))
- Add python profiling section ([c534bb3](https://github.com/astrosocket/packingcubes/commit/c534bb349b68ccf756d583f80b88d160aec14216))
- Change to profiling just the numba portion ([5c58715](https://github.com/astrosocket/packingcubes/commit/5c58715faefef4c818f3f154c36e1f88e863eab5))
- Add brute force timings ([2b50200](https://github.com/astrosocket/packingcubes/commit/2b502002a2e8446d73cd039756087c282537b089))
- Add constant number ball search timings. Update plotting ([60213db](https://github.com/astrosocket/packingcubes/commit/60213dbd06ef3518d926aec84cd1205a5ee66691))
- Enable gc in test, do initial timing for precompile ([4f1ec1c](https://github.com/astrosocket/packingcubes/commit/4f1ec1cb8ab5d47907fbc8af90f87e0b270b8ded))
- Add brute-force creation/search ([7e71fb0](https://github.com/astrosocket/packingcubes/commit/7e71fb03a0a73d7a748e627c7888b1ed421b883e))
- Allow specifying centers/radii ([55f47e4](https://github.com/astrosocket/packingcubes/commit/55f47e48a7b5af8a2180514e24c39519bb34ac02))
- Update packed qbp indices to use DataContainer ([8167e6c](https://github.com/astrosocket/packingcubes/commit/8167e6c3d53f8a71395c250fbe8a4d00ca3a9b51))
- Changed default index output type and remove unwrapped function ([930182c](https://github.com/astrosocket/packingcubes/commit/930182cc2c7dbc8c61929541e0d59e02deabca3a))
- Switch expected growth curve ([a477ae3](https://github.com/astrosocket/packingcubes/commit/a477ae3cc8dd897497daee2ed630a7fc2062de61))
- Packed list test should be nonstrict ([60c3c82](https://github.com/astrosocket/packingcubes/commit/60c3c8247ddeca1d15b27d39743dc7efb28e2a8d))
- Add PackedKDTree placeholder and remove PythonOctree from plotting ([cd190b7](https://github.com/astrosocket/packingcubes/commit/cd190b7ce2b41ab7bdb5f2d5a3141ff7f962decd))
- Remove packli from packed grouping ([45de046](https://github.com/astrosocket/packingcubes/commit/45de046d4e1a2eddb7a64ed06eb570eab487ad5f))
- Add PackedKDTree options ([64f901b](https://github.com/astrosocket/packingcubes/commit/64f901bb67c583dcf4f9b31fda64ee35f037b520))
- Add KDTree profiling ([8e69220](https://github.com/astrosocket/packingcubes/commit/8e6922070f143c035f3b63de005fb31eb9d9bd85))
- Fix typo ([acf7d97](https://github.com/astrosocket/packingcubes/commit/acf7d97a7cdfb6ec0f9cd38204a257e979324cbd))
- Add second query ball point example ([d91ada7](https://github.com/astrosocket/packingcubes/commit/d91ada71cff2cec21286cee07a53761754664b41))
- Add first query ball point test based on example ([67e54a7](https://github.com/astrosocket/packingcubes/commit/67e54a744c5f6d174a0af0e8bc94d9de0603490d))
- Remove unwrapped fixture source ([1735272](https://github.com/astrosocket/packingcubes/commit/1735272d330a1f724ed6d18c3be0256baa9a71c8))
- Add first tests for KDTree API based on example ([708551e](https://github.com/astrosocket/packingcubes/commit/708551e6d55a16e63069a54258326910ee0184af))
- Update Illustris numbers, temporarily only plot Illustris+LB_L10 ([cf18e98](https://github.com/astrosocket/packingcubes/commit/cf18e98b895ecf746456ec9d3024bbdb181e1efe))
- Update LB_L10 numbers ([1652b57](https://github.com/astrosocket/packingcubes/commit/1652b57606662a3db03e377e7219e7b0a771646f))
- Update suite with qbp_indices ([add24eb](https://github.com/astrosocket/packingcubes/commit/add24eb4599421fbf3aa2c6f936f686705ce49cd))
- Add pack_list section to benchmarks ([7f7bf03](https://github.com/astrosocket/packingcubes/commit/7f7bf0330355f37fe87285a1cb75498d607aff08))
- Updated timing for L10 ([7ed6f41](https://github.com/astrosocket/packingcubes/commit/7ed6f413fad1e4577d683dee78ab59f493d94023))
- Rename kdtree tests to scipy tests ([80ea8ab](https://github.com/astrosocket/packingcubes/commit/80ea8ab5a6e652c97e2870d83e9a7f02d7555885))
- Check that FP errors are properly generated ([0b3e6a2](https://github.com/astrosocket/packingcubes/commit/0b3e6a2dc6b4dce421d24d2afe2f7be0a78a9a06))
- Add discovered check_valid failure mode ([2e798a5](https://github.com/astrosocket/packingcubes/commit/2e798a5d162e89c2025cff42f2dd1699642ee8d4))
- Add debug statement for data loading ([8d70268](https://github.com/astrosocket/packingcubes/commit/8d70268cb355fc206088ee22bb4700ebe36fc6db))
- Output only 3 decimals of particle number ([e11e03d](https://github.com/astrosocket/packingcubes/commit/e11e03dccecb138693bfcfe76acb36f1076e9260))
- Log number of particles used ([8841e63](https://github.com/astrosocket/packingcubes/commit/8841e63bcfb343df420708465e31b7d3b8afc8f4))
- Move timing statistics directly to timing_tests.py ([b41ea86](https://github.com/astrosocket/packingcubes/commit/b41ea86ff3356445596cc6f9d716b741d9049fd0))
- Use a single point InMemory dataset to do precompiling for speedup ([53716f4](https://github.com/astrosocket/packingcubes/commit/53716f48a61043b7d9ec33cdfc1fe9bd4a562419))
- Remove old comment ([31f2952](https://github.com/astrosocket/packingcubes/commit/31f29525c4fee6a7c042d7cc34102309f95554a5))
- Move cubes search to correct spot in benchmark order ([e61ea09](https://github.com/astrosocket/packingcubes/commit/e61ea09efb1deef6da61b775b37bce529ffdf919))
- Move cube creation to correct spot in benchmark order ([3f331ac](https://github.com/astrosocket/packingcubes/commit/3f331acbadcad76684b600e2e85d1190ff534077))
- Add cubes search benchmark and update cubes creation api ([8af8a72](https://github.com/astrosocket/packingcubes/commit/8af8a72a97d523e72a7b56593aed63964ecede58))
- Add basic benchmarking for cubing process ([06a60fc](https://github.com/astrosocket/packingcubes/commit/06a60fc1d60f9a94ba9d31179764a335cc11c9f2))
- Enforce single point in test, clarify docstring ([d226514](https://github.com/astrosocket/packingcubes/commit/d226514be23c54193a0d54a399e30339eef4b7eb))
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

- Add pypi publishing workflow and fix testpypi workflow ([4410603](https://github.com/astrosocket/packingcubes/commit/4410603ded84e0937f4673ba6b013001a3481e65))
- Set pydocstyle and fix license file name ([76da9fb](https://github.com/astrosocket/packingcubes/commit/76da9fbb3dfefa1231ab2b4ed54d7383294d053e))
- Fix extra space in dependabot timing ([#7](https://github.com/astrosocket/packingcubes/issues/7)) ([b2ac0f4](https://github.com/astrosocket/packingcubes/commit/b2ac0f4dbb7cf1bbc312e577f3786caaf6994942))
- Fix extra space in dependabot timing ([9102209](https://github.com/astrosocket/packingcubes/commit/9102209cd90bf7d9e8074427c6830fb8ec37d530))
- Switch to frozen install in CI ([1a974f0](https://github.com/astrosocket/packingcubes/commit/1a974f0e59585f506223875cef3bc2672eea7cb8))
- Update lockfile and jupyter environment ([933886d](https://github.com/astrosocket/packingcubes/commit/933886d166ab12224d3b495f664169549964a837))
- Update checkout version in workflows ([e9706ac](https://github.com/astrosocket/packingcubes/commit/e9706acdc86c5cea6ef262acc8703df1a69ff2a4))
- Update pixi version in CI ([8c5ffcf](https://github.com/astrosocket/packingcubes/commit/8c5ffcf4a4b7880b999d7b83bd58a4015f5d5d1d))
- Add project metadata and pypi release workflow ([ebb04b6](https://github.com/astrosocket/packingcubes/commit/ebb04b61edade29ef545b986b18be1197270e9fc))
- Add pykdtree for future benchmarking ([6787bdd](https://github.com/astrosocket/packingcubes/commit/6787bdd622c9e236b949645ab9dbbe0b75d2c9e5))
- Updating lock file ([ceab923](https://github.com/astrosocket/packingcubes/commit/ceab9234bbeb8485af1190508242b30b3b587b79))
- Remove old CurrentNode comment ([8c3562b](https://github.com/astrosocket/packingcubes/commit/8c3562b0cabf73245866ad6625b4b263d60e6243))
- Update jupytext versioning ([18f4bfb](https://github.com/astrosocket/packingcubes/commit/18f4bfbcb30946f07000484092e433775e8fa907))
- Update pixi lock file ([50bc557](https://github.com/astrosocket/packingcubes/commit/50bc557cf16d2b53bc8ea4b5ec741fa7ff0213f9))
- Update changelog processing and regenerate ([586c43e](https://github.com/astrosocket/packingcubes/commit/586c43e3d8570d5c00629c586c1c8e878f511402))
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
## [0.0.0] - 2025-10-29
<!-- generated by git-cliff -->
