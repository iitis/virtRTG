# Algorithmic References and Acknowledgements

## Overview

The virtRTG plugin implements a number of well-established algorithms and
techniques commonly used in medical imaging, computer graphics, and scientific
computing. The implementations included in this project were developed
independently; however, the underlying methods are based on previously published
research and established technical conventions.

The following references are provided to acknowledge the original authors of
these methods and to facilitate further reading.

## Beer–Lambert Attenuation Model

The conversion between accumulated attenuation and image intensity is based on
the Beer–Lambert law, which describes exponential attenuation of radiation
passing through matter.

Relevant components:

* `XRayPhysicsModel`

References:

* Beer, A. (1852). *Bestimmung der Absorption des rothen Lichts in farbigen Flüssigkeiten*.
* Lambert, J. H. (1760). *Photometria*.

---

## Siddon Ray-Tracing Algorithm

The exact volumetric projection backend uses voxel-boundary traversal and
accumulates ray path lengths through individual voxels using principles
described by Siddon.

Relevant components:

* `VolumetricXRaySource`

Reference:

* Siddon, R. L. (1985). *Fast calculation of the exact radiological path for a three-dimensional CT array*. Medical Physics, 12(2), 252–255.

---

## Interpolated Ray Sampling

The sampled volumetric projection backend employs ray marching combined with
interpolation of volumetric data. This approach is conceptually related to the
family of interpolated projectors commonly used in computed tomography and
digital radiography simulations.

Relevant components:

* `XRayProjector`
* `VolumetricXRaySource`

---

## Möller–Trumbore Ray–Triangle Intersection

Mesh-based projection backends use barycentric ray–triangle intersection tests
following the method introduced by Möller and Trumbore.

Relevant components:

* Mesh projection and ray-intersection routines.

Reference:

* Möller, T., & Trumbore, B. (1997). *Fast, Minimum Storage Ray-Triangle Intersection*. Journal of Graphics Tools, 2(1), 21–28.

---

## Ray–Box Intersection (Slab Method)

Bounding-volume traversal uses the standard slab-based ray–axis-aligned bounding
box (AABB) intersection technique widely used in ray tracing and spatial search
structures.

Relevant components:

* BVH traversal and ray–box intersection routines.

Reference:

* Williams, A., Barrus, S., Morley, R., & Shirley, P. (2005). *An Efficient and Robust Ray–Box Intersection Algorithm*.

---

## Top-Left Rasterization Rule

Projected-triangle rasterization follows the standard top-left fill convention
commonly used in graphics APIs to ensure consistent coverage of shared edges and
to avoid double counting of pixels.

Relevant components:

* Triangle rasterization routines used by mesh projection backends.

---

## Additional Notes

The project also relies on a number of external open-source libraries,
including:

* NumPy
* SciPy
* OpenCV
* PyDICOM
* PyQt5
* PyOpenGL

Users redistributing the software should consult the corresponding licenses of
these dependencies.

## Disclaimer

References listed in this document acknowledge the scientific and technical
origins of the algorithms used by the project. Their inclusion does not imply
that source code from the referenced works or third-party software packages has
been copied into this repository.
