# Tree Data Structure

## Internal Node:

| skip_length | node_start | node_end | child_flag  | my_index | level | unused | C0  | C1  | C5  | C6  | C7  | parent_offset |
| ----------- | ---------- | -------- | ----------- | -------- | ----- | ------ | --- | --- | --- | --- | --- | ------------- |
| uint32      | uint32     | uint32   | uint8       | uint8    | uint8 | uint8  |     |     |     |     |     | uint32        |
| `5+sum([c.skip_length for c in children])` |     |    | CN is present if `child_flag & (2**N)=1` | Which child is this (0 if root) | What level node is this. root is 0 |        |     |     |     |     |     | `skip_length + sum([c.skip_length for c in siblings if c < self])` |
| field 0      | field 1     | field 2   | field 3      |          |       |        | *   | *   | *   | *   | *   | field 4        |


## Leaf Node:
Size = 5 fields = 20 bytes

| skip_length=20 | node_start | node_end | child_flag=0 | my_index | level | unused | parent_offset |
| -------------- | ---------- | -------- | ------------ | -------- | ----- | ------ | ------------- |
| uint32         | uint32     | uint32   | uint8        | uint8    | uint8 | uint8  | uint32        |

## Example:
![Example packed array implementation](example_packed.jpg)
Here, $x_{s}$, $x_{e}$, and $x_{m}$ are the node_start, node_end, and metadata (child_flag, my_index, level, and empty) fields for node $x$. The starred nodes in the tree are the last sibling for each group of children.
