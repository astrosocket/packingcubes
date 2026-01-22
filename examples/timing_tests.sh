
# echo "Data resetting"
# python -m timeit -s "import timing_tests;ds = timing_tests.load_data();" "ds = timing_tests.reset_data(ds)"

# echo "packingcubes.PythonOctree creation"
# python -m timeit -s "import timing_tests;ds = timing_tests.load_data();" "ds = timing_tests.reset_data(ds);timing_tests.python_octree_creation(ds)"

# echo "packingcubes.PythonOctree search"
# python -m timeit -s "import timing_tests;ds = timing_tests.load_data();tree = timing_tests.python_octree_creation(ds)" "timing_tests.python_octree_query_ball_point(tree)"

# echo "packingcubes.PackedOctree creation"
# python -m timeit -s "import timing_tests;ds = timing_tests.load_data();" "ds = timing_tests.reset_data(ds);timing_tests.packed_octree_creation(ds)"

# echo "packingcubes.PackedOctree search"
# python -m timeit -s "import timing_tests;ds = timing_tests.load_data();tree = timing_tests.packed_octree_creation(ds)" "timing_tests.packed_octree_query_ball_point(tree)"

# echo "scipy.spatial.kdtree creation"
# python -m timeit -s "import timing_tests;ds = timing_tests.load_data();" "ds = timing_tests.reset_data(ds);timing_tests.kdtree_creation(ds)"

# echo "scipy.spatial.kdtree search"
# python -m timeit -s "import timing_tests;ds = timing_tests.load_data();tree = timing_tests.kdtree_creation(ds)" "timing_tests.kdtree_query_ball_point(tree)"

echo "yt creation"
python -m timeit -s "import timing_tests;ytdata = timing_tests.yt_setup();" "sph = timing_tests.yt_creation(ytdata)"

echo "yt search"
python -m timeit -s "import timing_tests;ytdata = timing_tests.yt_setup();sph = timing_tests.yt_creation(ytdata)" "timing_tests.yt_search(sph)"