---
icon: lucide/terminal
---

# Command Line Interface

<pre><font color="#12488B"><b>usage: </b></font><font color="#A347BA"><b>packcubes</b></font> [<font color="#26A269">-h</font>] [<font color="#26A269">-v</font>] [<font color="#26A269">-n </font><font color="#A2734C">N</font>] [<font color="#26A269">-p </font><font color="#A2734C">PARTICLE_THRESHOLD</font>] [<font color="#26A269">-x </font><font color="#A2734C">X</font>] [<font color="#2AA1B3">-dx </font><font color="#A2734C">DX</font>]
                                                   [<font color="#26A269">-y </font><font color="#A2734C">Y</font>] [<font color="#2AA1B3">-dy </font><font color="#A2734C">DY</font>] [<font color="#26A269">-z </font><font color="#A2734C">Z</font>] [<font color="#2AA1B3">-dz </font><font color="#A2734C">DZ</font>] [<font color="#26A269">-c </font><font color="#A2734C">CONFIG</font>]
                                                   [<font color="#26A269">-t </font><font color="#A2734C">PARTICLE_TYPES [PARTICLE_TYPES ...]</font>]
                                                   [<font color="#2AA1B3">--force-overwrite</font>] [<font color="#2AA1B3">--no-saving-dataset</font>]
                                                   <font color="#26A269">snapshot</font> <font color="#26A269">[output]</font>

Run the packingcubes program on a snapshot file. Default is to use the bounding box provided by the
simulation, so if that&apos;s sufficient you do not need to provide x/y/z/dx/dy/dz

<font color="#12488B"><b>positional arguments:</b></font>
  <font color="#26A269"><b>snapshot</b></font>              Path to the snapshot file
  <font color="#26A269"><b>output</b></font>                Name of hdf5 file to save cubes information to. If not specified, cubes information
                        will be discarded!

<font color="#12488B"><b>options:</b></font>
  <font color="#26A269"><b>-h</b></font>, <font color="#2AA1B3"><b>--help</b></font>            show this help message and exit
  <font color="#26A269"><b>-v</b></font>, <font color="#2AA1B3"><b>--verbose</b></font>         increase output verbosity
  <font color="#26A269"><b>-n</b></font>, <font color="#2AA1B3"><b>--side-length</b></font> <font color="#A2734C"><b>N</b></font>   number of cells per side [3-32], default -1 means use the lowest number of cells such
                        that n**3 &gt; # of available threads
  <font color="#26A269"><b>-p</b></font>, <font color="#2AA1B3"><b>--particle-threshold</b></font> <font color="#A2734C"><b>PARTICLE_THRESHOLD</b></font>
                        the maximum number of particles per octree leaf. Default: 400
  <font color="#26A269"><b>-c</b></font>, <font color="#2AA1B3"><b>--config</b></font> <font color="#A2734C"><b>CONFIG</b></font>   Read in specified config file for arguments (CLI arguments will override)
  <font color="#26A269"><b>-t</b></font>, <font color="#2AA1B3"><b>--particle-types</b></font> <font color="#A2734C"><b>PARTICLE_TYPES [PARTICLE_TYPES ...]</b></font>
                        Names of particles to include (Can be integers or strings, 0 &lt;=&gt; PartType0)
  <font color="#2AA1B3"><b>--force-overwrite</b></font>     Flag to overwrite cubes data contained in OUTPUT
  <font color="#2AA1B3"><b>--no-saving-dataset</b></font>   Don&apos;t save sorted particle positions and shuffle lists Normally sorted particle
                        positions/shuffle lists are saved within a sidecar file to the snapshot. This flag
                        disables that behavior.

<font color="#12488B"><b>Box parameters:</b></font>
  Arguments to override parts of the default bounding box

  <font color="#26A269"><b>-x</b></font> <font color="#A2734C"><b>X</b></font>                  minimum bounding box x coordinate
  <font color="#2AA1B3"><b>-dx</b></font> <font color="#A2734C"><b>DX</b></font>                bounding box size in x direction
  <font color="#26A269"><b>-y</b></font> <font color="#A2734C"><b>Y</b></font>                  minimum bounding box y coordinate
  <font color="#2AA1B3"><b>-dy</b></font> <font color="#A2734C"><b>DY</b></font>                bounding box size in y direction
  <font color="#26A269"><b>-z</b></font> <font color="#A2734C"><b>Z</b></font>                  minimum bounding box z coordinate
  <font color="#2AA1B3"><b>-dz</b></font> <font color="#A2734C"><b>DZ</b></font>                bounding box size in z direction

If particle types are specified (using -t or --particle-types), the snapshot file should be specified with --
SNAPSHOT OUTPUT at the end. Additional arguments can be read from a file by specifying the file with
`@filename` anywhere among the argument string. Any arguments found in the file will overwrite previously
specified arguments and be overwritten by arguments specified later. Note that this is different behavior
from the -c/--config argument!</pre>