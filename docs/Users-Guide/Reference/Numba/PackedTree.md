---
icon: lucide/network
---

# Numba Packed Trees

::: packingcubes.packed_tree.packed_tree_numba
    options:
        show_root_heading: true
        members:
        - PackedTreeNumba
        - euclidean_distance_particle
        - euclidean_d2
        - _construct_tree
        - _construct_node_recursive


!!! info "Additional semi-private methods for PackedTreeNumba:"
::: packingcubes.packed_tree.packed_tree_numba.PackedTreeNumba._get_particle_indices_in_shape
    options:
        show_root_heading: true
::: packingcubes.packed_tree.packed_tree_numba.PackedTreeNumba._get_particle_index_list_in_shape
    options:
        show_root_heading: true

::: packingcubes.packed_tree.packed_node
    options:
        show_root_heading: true
        members: 
        - CurrentNode
        - PackedNodeNumba
        - get_name
        - get_children
        - is_leaf
        - is_root
        - expand_range
