# VTacO_v2
### GeO structure
1. mano_pose -> hand_verts ($B \times 778 \times 3$), sensor_points (anchors ($B \times N_A \times 3$))  
   SDF -> obj_verts, obj_faces -> obj_normals
2. sensor points (anchors) ---(find)---> obj_verts_near with region ($B \times N_r \times N_A \times 3$)
3. Repl string: obj_verts_near with region ($B \times N_r \times N_A \times 3$) -> $E_{repl} = \frac{1}{2} k_{repl} * (e^{(v_o - v_h) \cdot n_o})^2$
4. Attr string (region sensor > 0, with mask ($B \times N_r$)): obj_verts_near with region ($B \times N_r \times N_A \times 3$) -> $E_{attr} = \frac{1}{2} k_{attr} * ||v_o - v_h||_2^2$, sensor forces regional sum ($B \times N_r$) -> $k_a$

### Data format

1. joint json: 
   - Joint pose: bodyTypeProperties > q (First 3: Trans) (Next 9: Rot)
   - Joint verts: verts > data > x
   - Joint verts contact force: verts > data > contact
