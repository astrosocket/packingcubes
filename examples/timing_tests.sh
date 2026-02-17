if [ -z "$1" ]
then
DECIMATION_FACTOR=1
else
DECIMATION_FACTOR=$1
fi
echo "Running with df=$DECIMATION_FACTOR"
echo "Timing description:"
python -c "import timing_tests;ds = timing_tests.load_data($DECIMATION_FACTOR);print(f'Testing {len(ds):.1e} particles')"

echo "Data resetting"
python -m timeit -s "import timing_tests;ds = timing_tests.load_data($DECIMATION_FACTOR);" "ds = timing_tests.reset_data(ds)"

echo "packingcubes.PythonOctree creation"
python -m timeit -s "import timing_tests;ds = timing_tests.load_data($DECIMATION_FACTOR);timing_tests.precompile()" "ds = timing_tests.reset_data(ds);timing_tests.python_octree_creation(ds)"

echo "packingcubes.PackedOctree creation"
python -m timeit -s "import timing_tests;ds = timing_tests.load_data($DECIMATION_FACTOR);timing_tests.precompile();" "ds = timing_tests.reset_data(ds);timing_tests.packed_octree_creation(ds)"

echo "scipy.spatial.kdtree creation"
python -m timeit -s "import timing_tests;ds = timing_tests.load_data($DECIMATION_FACTOR);" "ds = timing_tests.reset_data(ds);timing_tests.kdtree_creation(ds)"

# echo "yt creation"
# python -m timeit -s "import timing_tests;ytdata = timing_tests.yt_setup($DECIMATION_FACTOR);" "sph = timing_tests.yt_creation(ytdata)"

echo "packingcubes.PythonOctree search"
python -m timeit -s "import timing_tests;ds = timing_tests.load_data($DECIMATION_FACTOR);timing_tests.precompile();tree = timing_tests.python_octree_creation(ds)" "timing_tests.python_octree_query_ball_point(tree)"

echo "packingcubes.PackedOctree search"
python -m timeit -s "import timing_tests;ds = timing_tests.load_data($DECIMATION_FACTOR);timing_tests.precompile();tree = timing_tests.packed_octree_creation(ds)" "timing_tests.packed_octree_query_ball_point(tree)"

echo "scipy.spatial.kdtree search"
python -m timeit -s "import timing_tests;ds = timing_tests.load_data($DECIMATION_FACTOR);tree = timing_tests.kdtree_creation(ds)" "timing_tests.kdtree_query_ball_point(tree)"

# echo "yt search"
# python -m timeit -s "import timing_tests;ytdata = timing_tests.yt_setup($DECIMATION_FACTOR);sph = timing_tests.yt_creation(ytdata)" "timing_tests.yt_search(sph)"

echo "Tree sizes:"
python -c "import timing_tests;timing_tests.tree_sizes($DECIMATION_FACTOR)"

echo "Cubing"
python -m timeit -s "import timing_tests;setup = timing_tests.cubing_setup();timing_tests.cubing(setup);" "timing_tests.cubing(setup)"
