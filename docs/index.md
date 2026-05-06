---
icon: lucide/box
---

--8<-- "README.md::16"

## Why use `packingcubes`?
Basically, you have a large dataset with position (or at least 3-dimensional)
information and you want information in it that's spatially clustered, as soon
as possible. Also, you want to do this more than once and not use too many
resources in the process. 

**Key Features**:

* **Low-memory**: Unlike competitors (e.g. SciPy's `spatial.KDTree` or 
  `pykdtree`), `packingcubes` sorts your dataset using a chunked octree, so
  each node contains only minimal information (20 bytes per node, meaning a
  `PackedTree` with the same number of nodes can be ~100x smaller than a
  `KDTree` on the same data)
* **Fast**: `packingcubes` preferentially uses and returns index chunks/slices
  (i.e. `(start, stop)` <-> `start:stop`), leading to increased cache
  efficiency and significant speedups when working with large datasets.
* **Useability**: `packingcubes` is designed to work with datasets that contain
  information associated with the position information to make spatially
  searching other fields easy. E.g. finding a basic quantity like the mass of
  all particles in a sphere is a simple one-liner.
* **Parallelization**: The cubes in `packingcubes` are created and searched
  in parallel, with the cubes creation exhibiting strong scaling.
* **Easy Serialization**: the tree information is trivially represented with
  collections of numpy arrays and can easily/quickly be saved to and loaded
  from storage.
* **Pure Python**: There's no C/Rust/etc. extensions and we're available on
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
