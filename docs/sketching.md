# Sketching in BasiCAM CNC

Sketches are flat drawings on a plane (XY, XZ, YZ) or on a face of an imported part; on a curved
surface they are projected onto it. A sketch becomes a toolpath target only when it is **fully
defined**: every point is fixed by dimensions and relations, or anchored to the origin or the model.

## Colours and status

- White: fully defined. Yellow (with `*` in the tree): under-defined, it can still move.
  Red: could not solve (conflicting dimensions or relations, a dangling model reference).
- The status line in the Relations panel says "Fully defined", "Under defined (n DOF, m free)" or
  "Could not solve (conflicting: ...)".

## Drawing

- Pick a tool (Line, Rectangle, Circle, Curve, Spline, Slot, Polygon, Centerline, Point). Click to
  start; the entity follows the mouse (rubber band) with its length and angle, size or radius
  shown next to the cursor. Click again, or press, drag and release, to finish.
- Lines chain: every click continues from the last end. Double-click, Esc or a right-click ends
  the chain; clicking the first point closes it. Press **A** for a tangent arc as the next segment.
- Type a value while drawing (for example `25`, `1in`, `1 1/2"`) and press Enter to make the
  line that long (or the circle that radius) with a dimension.
- Snapping is by screen distance (Preferences > Sketch snap radius): sketch points and centres
  first, then model vertices, midpoints, tangent points, model edges and sketch curves.
- Snapping adds relations automatically: on a sketch point coincident, on a midpoint midpoint, on a
  curve on-curve, on a **model vertex the point is defined there** (it follows the part when the
  part moves), on a model edge on-edge. A line within 3 degrees of horizontal or vertical gets
  that relation. Hold Ctrl to place a point without relations.
- Esc backs out one step at a time: the entity in progress, the tool, then the sketch.

## Relations

- The Relations panel lists every relation with its state (Satisfied, Conflicting, Dangling, Not
  solved). Select one to highlight its entities; Delete, Delete all, or Add a relation for the
  entities selected in the tree. A dangling model reference (its part was deleted) is fixed with
  **Re-pick** and a click on a vertex or edge.
- Glyphs in the view show each relation (View > Sketch relations).
- Drag an under-defined point to see how the sketch can move; fully defined geometry stays put.

## Smart Dimension (D)

- Pick one or two items, then click where the dimension goes and type its value. The type comes
  from the picks: a line gives its length, a circle its diameter, an arc its radius, two points a
  distance (horizontal, vertical or aligned by where you place it), a point and a line their
  distance, two lines their angle (or their distance when parallel), a point alone X / Y from the
  origin. Values accept units, fractions and arithmetic: `10+2.5`, `1/2in*2`, `(3 ft)/4`.
- Double-click a dimension to edit it; drag it to move the label. A dimension that would
  over-define the sketch can be made driven: it then only reports the size (shown in brackets).

## Modify

Trim (click a piece between intersections, or drag across several), Extend, Offset, Mirror about a
centerline, Fillet and Chamfer (the corner is kept as a construction point), Convert Entities (a
selected model face's outline, tied to the model), Move / Copy, and Fully Define, which adds the
horizontal / vertical relations and dimensions the sketch still needs.

## From sketch to toolpath

- Closed loops become regions (loops inside loops are islands), open chains are trace targets and
  points or circle centres are drill targets. **Check Sketch** must pass: no gaps over 0.01 mm, no
  overlaps, no zero-length or self-crossing entities (the message gives the location).
- Pocket clears a region around its islands; Profile runs Inside, Outside or On the line with
  the tool radius computed by BasiCAM; Trace follows a chain; Drill makes one hole per point.
- **Assign tool...** on a selected face or curved surface makes a fully defined toolpath sketch for
  the chosen tool: points 60% of the tool diameter apart, outside the surface, rounded away from
  the material, rough layers from the stock top, then 3-axis finishing.
