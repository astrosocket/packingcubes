# Packed Format Design Specification

Version 1.0.0

This file documents the format used to describe a PackedTree data structure.

A packed tree is composed of two parts: an initial header containing tree
metadata (e.g. tree checksum, bounding box, etc), and the actual tree structure
in memory. 

The basic unit of the packed format is the `field`, currently defined as a
`uint32`. Thus, the entire packed tree can be considered an array of `uint32`,
and internally is stored as a numpy array of such.


## Tree Metadata

| Field | Bytes   | Description                                                                                                                                            |
| ----- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0     | 0-3     | Size of header in bytes (=144).<br>Note this also sets endianness                                                                                      |
| 1     | 4       | Field size in bytes as an int8. Currently set to 4, representing a uint32.<br> A negative value would represent a _signed_ integer (i.e. -4 <=> int32) |
| 1     | 5-7     | Packed specification version as major - minor - patch<br>following semantic versioning (see [semver.org](semver.org)), stored as 3 uint8s              |
| 2-3   | 8-15    | Creation time as a UTC timestamp                                                                                                                       |
| 4-20  | 16-83   | Checksum method as a string: `'xxh3_64_intdigest'`                                                                                                     |
| 21-22 | 84-91   | Tree checksum computed via xxhash's xxh3_64_intdigest                                                                                                  |
| 23-24 | 92-99   | Tree bounding box x position                                                                                                                           |
| 25-26 | 100-107 | Tree bounding box y position                                                                                                                           |
| 27-28 | 108-115 | Tree bounding box z position                                                                                                                           |
| 29-30 | 116-123 | Tree bounding box dx                                                                                                                                   |
| 31-32 | 124-131 | Tree bounding box dy                                                                                                                                   |
| 33-34 | 132-139 | Tree bounding box dz                                                                                                                                   |
| 35    | 140-143 | Particle threshold                                                                                                                                     |

Timestamp is currently computed via

```python
# conversion to timestamp
t = np.float64(datetime.datetime.now(tz=datetime.UTC).timestamp())
# conversion from timestamp
datetime.datetime.fromtimestamp(t, tz=datetime.UTC)
```

For the checksum, use [xxhash's xxh3](https://pypi.org/project/xxhash/) (not part of the standard library):

```python
from xxhash import xxh3_64_intdigest 

checksum = xxh3_64_intdigest(packed_tree)
```
This creates a uint64 checksum.

## Tree Data Structure

Essentially we can think of the tree structure as the equivalent of `numpy`'s
structured array, except in an octree structure as opposed to the linear array
structure. Thus, the bytes are organized such that all information for a node
(*including it's children!*) are together in sequential order. 

The intent is to mimic the in-memory structure of a simple tree from a language
like C, where every child node pointer has been replaced by the actual bytes
of the child node.

Thus, the first few bytes correspond to data attached to the root node, the
next few bytes are the first child's data, followed by the first grandchild,
etc., in a pre-order depth-first traversal of the tree.


### Internal Node:

| skip_length | node_start | node_end | child_flag  | my_index | level | unused | C0  | C1  | C5  | C6  | C7  | parent_offset |
| ----------- | ---------- | -------- | ----------- | -------- | ----- | ------ | --- | --- | --- | --- | --- | ------------- |
| uint32      | uint32     | uint32   | uint8       | uint8    | uint8 | uint8  |     |     |     |     |     | uint32        |
| `5+sum([c.skip_length for c in children])` |     |    | CN is present if `child_flag & (2**N)=1` | Which child is this (0 if root) | What level node is this. root is 0 |        |     |     |     |     |     | `skip_length + sum([c.skip_length for c in siblings if c < self])` |
| field 0      | field 1     | field 2   | field 3      |          |       |        | *   | *   | *   | *   | *   | field 4        |

For the present case, `child_flag` would be 227 ($=2^0 + 2^1 + 2^5 + 2^6 + 2^7$). If all 5 nodes were leaves, then `skip_length`=30.

### Leaf Node:
Size = 5 fields = 20 bytes

| skip_length=20 | node_start | node_end | child_flag=0 | my_index | level | unused | parent_offset |
| -------------- | ---------- | -------- | ------------ | -------- | ----- | ------ | ------------- |
| uint32         | uint32     | uint32   | uint8        | uint8    | uint8 | uint8  | uint32        |

### Example:
![Example packed array implementation](example_packed.jpg)
Here, $x_{s}$, $x_{e}$, and $x_{m}$ are the node_start, node_end, and metadata (child_flag, my_index, level, and empty) fields for node $x$. The starred nodes in the tree are the last sibling for each group of children.


<script id="MathJax-script" src="https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
  window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    }
  };

  document$.subscribe(() => {
    MathJax.startup.output.clearCache()
    MathJax.typesetClear()
    MathJax.texReset()
    MathJax.typesetPromise()
  })
</script>
