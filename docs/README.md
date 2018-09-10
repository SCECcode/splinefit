# BSpline surface fitting
In this document, we explain how to fit a BSpline surface to an initial surface
model that is defined by a point cloud and a triangulation. For this example, we
will focus on the *West Garlock fault*, which belongs to the easy
category. A view of the mesh for this fault geometry is shown in Figure 1.

![](figures/west_garlock_initial.png)
**Figure 1**: Initial mesh for the West Garlock fault.



## Outline of fitting procedure
The faults in the community fault model and the ones that have been selected for
testing out the automatic meshing procedure have a variety of different
features. Some faults are simple and can be described by a single rectangular
plane, other faults are draped like curvy carpets, and may also be defined by
multiple intersecting segments or faults with holes. Despite many of these
complexities it seems like a first initial assumption is to view each fault as a
collection of surfaces that can be mapped to a rectangular path in 2D space.
With this assumption in mind, we can formulate a simple strategy for performing
the fitting. This procedure outline below, will be the first thing to try and
evaluate. It is designed to be the simplest procedure that I can think of and 
that I believe while have some chance at succeeding. 

These are the key components to the surface fitting, 

1. Boundary segment detection
2. 2D Plane projection
3. BSpline boundary curve fitting
4. Surface parameterization
5. BSpline surface fitting

Below follows an overview of what each step does.

In the first step, the goal is to detect the nodes and edges that lie on the
boundary of the domain. The boundary should form a closed loop. Under the
assumption that the boundary can be mapped to the edges of a rectangle, it is
broken up into four segments, mapping to each side of the rectangular patch
(left, right, top, bottom). It is also quite possible that there are multiple
boundaries in the same mesh. These other boundaries could be internal ones if
there are holes in the surface, or if there are multiple surfaces. For now, we
will ignore these complications.

After that the boundary has been detected and segmented, it is projected onto
the best fitting 2D plane. This plane is simply found by performing principal
component analysis (PCA) and defining the plane using the eigenvectors
corresponding to the two maximum modulus eigenvalues.

## Boundary segment detection
Altough there probably are plenty of packages that can be used to detect the
boundary of some triangulation, I decided to implement my own solution. It
appears that many of the meshes contain multiple surfaces. For now, only one
surface is treated. Figure 2 shows the boundary detection method in action. Finding the additional surfaces in the
triangulation is simply of matter rerunning the algorithm on the mesh after haviing removed the first boundary detected,
and then repeating the process until no more boundaries are detected.


![](figures/PNRA-CRSF-USAV-Fontana_Seismicity_lineament-CFM1_boundary.png)
![](figures/GRFS-GRFZ-WEST-Garlock_fault-CFM5_boundary.png)
![](figures/WTRA-NCVS-VNTB-Southern_San_Cayetano_fault-steep-JHAP-CFM5_boundary.png)

**Figure 2** : Detection of boundary edges (and nodes) of the faults in the easy
list. The strange looking surface of some meshes is an artifact of the
renderer. The last example (San Cayetano fault) contains multiple surfaces. Only
the first detected surface will be treated for now.

The way the boundary detection works is by noting that an edge lies on the
boundary of the mesh if this edge only appears in a single triangle. Edges that
are shared by two triangles are interior edges. Once the boundary edges have
been detected, they are ordered by starting at some arbitrary boundary node and
then selecting the next node by looking for its nearest neighbor, and by
excluding itself or a previous node from the search. 

Once the boundary edges have been detected, they are split into four boundary
segments. The reason for wanting to split the boundary into separate segments is becuse BSpline curves assume
smoothness, but there is no continuity at the corner points. To identify the corner nodes, the angle between two neighboring edges
is measured at each node on the boundary. A node is marked as a corner point if it
is among the top four nodes that have the angles closest to being orthogonal. Unfortunately, this simple strategy does
not succesfully detect corner points in 3D. Due to variability in the data it is comppletely possible that non-corner
points are flagged as being corner points. Perhaps this difficulity is overcome by first projecting the boundary points
onto the best fitting plane, discussed next.

## 2D Plane Projection
After boundary detection, the boundary points are projected onto the best fitting plane. While there is a some global origin
defined for the entire point cloud, a new coordinate system is introduced which defines its origin as a point on the plane. This change of coordinates simplifies the construction of the plane.
This new origin is defined by taking the mean of the coordinates of the boundary points . In addition, each point on the
boundary is
renormalized using the standard deviation. The plane to use for the projection is computed by using principal component
analysis (PCA). The two eigenvectors that are paired with the two eigenvalues of largest and sencod largest eigenvalues are
selected as the basis vectors that span the plane. Naturally, the eigenvector corresponding to the eigenvalues of
smallest magnitude defines the normal to this plane. The reason why PCA has been chosen for selecting this plane is
because this method finds the plane that minimizes the orthogonal distance from the plane to the boundary points. Once the plane has been found, its basis vectors are used to project the boundary points onto the best fitting plane.

Figure 3 shows the basis vectors defining the plane, and the projection of the boundary onto the plane.

![](figures/PNRA-CRSF-USAV-Fontana_Seismicity_lineament-CFM1_projection.png) 
![](figures/GRFS-GRFZ-WEST-Garlock_fault-CFM5_projection.png)
![](figures/WTRA-NCVS-VNTB-Southern_San_Cayetano_fault-steep-JHAP-CFM5_projection.png)

**Figure 3:** Projection of boundary points (black) onto the best fitting plane (blue). The orange and green lines show
the basis vectors spanning the plane. The gap in the boundary is because it is not plotted as closed loop. This issue
will be addressed in the near future. 
