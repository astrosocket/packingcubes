---
icon: lucide/box
---

--8<-- "README.md::16"

## Key Features

- **Separation between data and tree (meta)data**: Unlike competitors (e.g.
    SciPy's `spatial.KDTree` or `pykdtree`), `packingcubes` sorts your dataset
    using a chunked octree and doesn't associate any positional information
    with the nodes. This leads to a number of useful properties:

    - **Low-memory**: Each node contains only minimal information (20 bytes per
      node), meaning a `PackedTree` with the same number of nodes can be ~100x
      smaller than a `KDTree` on the same data
    - **Fast tree traversal**: Tree searches on big datasets are usually memory
      bound. Smaller trees can more easily fit in cache/memory -> much faster
      tree traversal
    - **Easy Serialization**: the tree information is trivially represented with
      collections of numpy arrays and can easily/quickly be saved to and loaded
      from storage.
    
- **Built for loading datasets**: `packingcubes` preferentially uses and
  returns index chunks/slices (i.e. `(start, stop)` <-> `start:stop`), leading
  to increased cache efficiency and significant speedups when working with
  large datasets.
- **Useability**: `packingcubes` is designed to work with datasets that contain
  information associated with the position information to make spatially
  searching other fields easy. E.g. finding a basic quantity like the mass of
  all particles in a sphere is a simple one-liner.
- **Parallelization**: The cubes in `packingcubes` are created and searched
  in parallel, with the cubes creation exhibiting strong scaling.
- **Pure Python**: There's no C/Rust/etc. extensions and we're available on
  PyPI -> we're extendable and portable!


## Site Navigation

<div class="grid cards" markdown>
    
- :lucide-rocket:{ .lg .middle } **Getting Started**
        
    ---
    
    Install `packingcubes` from [PyPI](https://pypi.org/project/packingcubes/) and start cubing :lucide-boxes:!

  
    [:octicons-arrow-right-24: Getting started](Users-Guide){ .md-button .md-button--primary }

- :lucide-book-open:{ .lg .middle } **User's Guide**

    ---

    Want to learn more?

    [:lucide-info: Tutorials](Users-Guide/Tutorials){ .md-button .md-button--primary }

    [:lucide-cooking-pot: Recipes](Users-Guide/Recipes){ .md-button .md-button--primary }

    [:lucide-library: API Reference](Users-Guide/Reference){ .md-button .md-button--primary }

    [:lucide-message-square-text: Explanations](Users-Guide/Explanations){ .md-button .md-button--primary }


- :lucide-square-code:{ .lg .middle } **Contributing**
        
    ---

    Interested in helping out?

    [:octicons-arrow-right-24: Information for developers](Developer-Guide){ .md-button .md-button--primary }

    [:octicons-mark-github-24: Source Code](https://www.github.com/astrosocket/packingcubes){ .md-button }    

  

- :lucide-book-search:{ .lg .middle } **Publishing?**

    ---

    Used `packingcubes` in your project?

    [:octicons-arrow-right-24: Citation Information](Citations){ .md-button .md-button--primary }

</div>
