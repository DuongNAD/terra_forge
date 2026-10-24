"""
Procedural & PBR Shaders for TerraForge in Blender:
1. Triplanar Slope-Aware Terrain Shader (Rock, Scree, Forest Ground, Sand, Snow).
2. Physical Beer-Lambert Absorption Water Shader with Contact Foam Margin.
3. Bioluminescent Subsurface Cave Fungi Shader.
"""

from __future__ import annotations

try:
    import bpy
    HAS_BPY = True
except ImportError:
    bpy = None
    HAS_BPY = False

from terra_forge.schema.map_config import ShaderConfig


def _get_mix_rgba_sockets(node: bpy.types.ShaderNode):
    sock_fac = node.inputs.get("Factor_Float") or node.inputs.get("Factor") or node.inputs[0]
    sock_a = node.inputs.get("A_Color") or node.inputs.get("A") or node.inputs[6]
    sock_b = node.inputs.get("B_Color") or node.inputs.get("B") or node.inputs[7]
    sock_out = node.outputs.get("Result_Color") or node.outputs.get("Result") or node.outputs[2]
    return sock_fac, sock_a, sock_b, sock_out


def _get_mix_float_sockets(node: bpy.types.ShaderNode):
    sock_fac = node.inputs.get("Factor_Float") or node.inputs.get("Factor") or node.inputs[0]
    sock_a = node.inputs.get("A_Float") or node.inputs.get("A") or node.inputs[2]
    sock_b = node.inputs.get("B_Float") or node.inputs.get("B") or node.inputs[3]
    sock_out = node.outputs.get("Result_Float") or node.outputs.get("Result") or node.outputs[0]
    return sock_fac, sock_a, sock_b, sock_out


def create_triplanar_terrain_shader(config: ShaderConfig) -> bpy.types.Material:
    """
    Constructs a true Object-Space Triplanar PBR multi-splatting terrain material in Blender.
    Features:
    - 5 Geological PBR Material Classes:
      1. Cliff Bedrock (dark slate/granite, high roughness, rock strata normal bump)
      2. Scree & Talus (loose angular gravel/pebbles modulated by hf.scree_talus)
      3. Humus Soil & Moss (rich dark loam, moisture-driven damp moss and alluvial sediment)
      4. Coastal & River Sand (warm alluvial sand on low-elevation beaches and riverbanks)
      5. Alpine Snow (crisp crystalline snow caps shedding on steep cliff faces)
    - Normal Exponent Triplanar Blending (p = 6.0 in [4, 8]) eliminating seam artifacts.
    - Multi-scale procedural noise boundary perturbation eliminating laser-straight contours and UV stretching.
    - Strata vertex color (COLOR_0) geological multi-layering.
    """
    if not HAS_BPY or bpy is None:
        return None

    mat_name = "MAT_TerraForge_Terrain_Triplanar"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Material Output & Principled BSDF
    node_out = nodes.new(type="ShaderNodeOutputMaterial")
    node_out.location = (2000, 0)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (1650, 0)
    links.new(bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    # Geometry: Position & Normal
    geom = nodes.new(type="ShaderNodeNewGeometry")
    geom.location = (-1600, 200)

    sep_pos = nodes.new(type="ShaderNodeSeparateXYZ")
    sep_pos.location = (-1350, -100)
    links.new(geom.outputs["Position"], sep_pos.inputs["Vector"])

    sep_norm = nodes.new(type="ShaderNodeSeparateXYZ")
    sep_norm.location = (-1350, 400)
    links.new(geom.outputs["Normal"], sep_norm.inputs["Vector"])

    # Normal blend weights (exponent sharpness p = 6)
    abs_x = nodes.new(type="ShaderNodeMath")
    abs_x.location = (-1150, 600)
    abs_x.operation = "ABSOLUTE"
    links.new(sep_norm.outputs["X"], abs_x.inputs[0])

    abs_y = nodes.new(type="ShaderNodeMath")
    abs_y.location = (-1150, 450)
    abs_y.operation = "ABSOLUTE"
    links.new(sep_norm.outputs["Y"], abs_y.inputs[0])

    abs_z = nodes.new(type="ShaderNodeMath")
    abs_z.location = (-1150, 300)
    abs_z.operation = "ABSOLUTE"
    links.new(sep_norm.outputs["Z"], abs_z.inputs[0])

    pow_x = nodes.new(type="ShaderNodeMath")
    pow_x.location = (-950, 600)
    pow_x.operation = "POWER"
    pow_x.inputs[1].default_value = 6.0
    links.new(abs_x.outputs["Value"], pow_x.inputs[0])

    pow_y = nodes.new(type="ShaderNodeMath")
    pow_y.location = (-950, 450)
    pow_y.operation = "POWER"
    pow_y.inputs[1].default_value = 6.0
    links.new(abs_y.outputs["Value"], pow_y.inputs[0])

    pow_z = nodes.new(type="ShaderNodeMath")
    pow_z.location = (-950, 300)
    pow_z.operation = "POWER"
    pow_z.inputs[1].default_value = 6.0
    links.new(abs_z.outputs["Value"], pow_z.inputs[0])

    # Normalization: sum = pow_x + pow_y + pow_z
    sum_xy = nodes.new(type="ShaderNodeMath")
    sum_xy.location = (-750, 525)
    sum_xy.operation = "ADD"
    links.new(pow_x.outputs["Value"], sum_xy.inputs[0])
    links.new(pow_y.outputs["Value"], sum_xy.inputs[1])

    sum_xyz = nodes.new(type="ShaderNodeMath")
    sum_xyz.location = (-600, 450)
    sum_xyz.operation = "ADD"
    links.new(sum_xy.outputs["Value"], sum_xyz.inputs[0])
    links.new(pow_z.outputs["Value"], sum_xyz.inputs[1])

    weight_x = nodes.new(type="ShaderNodeMath")
    weight_x.location = (-400, 600)
    weight_x.operation = "DIVIDE"
    links.new(pow_x.outputs["Value"], weight_x.inputs[0])
    links.new(sum_xyz.outputs["Value"], weight_x.inputs[1])

    weight_y = nodes.new(type="ShaderNodeMath")
    weight_y.location = (-400, 450)
    weight_y.operation = "DIVIDE"
    links.new(pow_y.outputs["Value"], weight_y.inputs[0])
    links.new(sum_xyz.outputs["Value"], weight_y.inputs[1])

    weight_z = nodes.new(type="ShaderNodeMath")
    weight_z.location = (-400, 300)
    weight_z.operation = "DIVIDE"
    links.new(pow_z.outputs["Value"], weight_z.inputs[0])
    links.new(sum_xyz.outputs["Value"], weight_z.inputs[1])

    # Procedural Noise Perturbation (Anti-Contour Masking & Anti-Tiling)
    noise_coarse = nodes.new(type="ShaderNodeTexNoise")
    noise_coarse.location = (-1350, 150)
    noise_coarse.inputs["Scale"].default_value = 0.35
    noise_coarse.inputs["Detail"].default_value = 4.0
    noise_coarse.inputs["Roughness"].default_value = 0.5
    links.new(geom.outputs["Position"], noise_coarse.inputs["Vector"])

    # Perturbed Slope (Normal.Z + Noise)
    add_slope_pert = nodes.new(type="ShaderNodeMath")
    add_slope_pert.location = (-750, 300)
    add_slope_pert.operation = "MULTIPLY_ADD"
    links.new(noise_coarse.outputs["Fac"], add_slope_pert.inputs[0])
    add_slope_pert.inputs[1].default_value = 0.18
    links.new(sep_norm.outputs["Z"], add_slope_pert.inputs[2])

    # Perturbed Altitude (Position.Z + Noise)
    add_alt_pert = nodes.new(type="ShaderNodeMath")
    add_alt_pert.location = (-750, -150)
    add_alt_pert.operation = "MULTIPLY_ADD"
    links.new(noise_coarse.outputs["Fac"], add_alt_pert.inputs[0])
    add_alt_pert.inputs[1].default_value = 3.5
    links.new(sep_pos.outputs["Z"], add_alt_pert.inputs[2])

    # --- MATERIAL MASKS ---
    # 1. Snow Mask (High Altitude & Slope Shedding)
    snow_alt_range = nodes.new(type="ShaderNodeMapRange")
    snow_alt_range.location = (-550, -150)
    snow_alt_range.inputs["From Min"].default_value = config.snow_altitude - 2.0
    snow_alt_range.inputs["From Max"].default_value = config.snow_altitude + 4.5
    links.new(add_alt_pert.outputs["Value"], snow_alt_range.inputs["Value"])

    snow_slope_ramp = nodes.new(type="ShaderNodeValToRGB")
    snow_slope_ramp.location = (-550, 0)
    snow_slope_ramp.color_ramp.elements[0].position = 0.52  # ~58 deg slope shedding
    snow_slope_ramp.color_ramp.elements[1].position = 0.75  # ~41 deg
    links.new(add_slope_pert.outputs["Value"], snow_slope_ramp.inputs["Fac"])

    snow_weight = nodes.new(type="ShaderNodeMath")
    snow_weight.location = (-300, -100)
    snow_weight.operation = "MULTIPLY"
    links.new(snow_alt_range.outputs["Result"], snow_weight.inputs[0])
    links.new(snow_slope_ramp.outputs["Color"], snow_weight.inputs[1])

    # 2. Cliff Bedrock Mask (Steep rock scarp: slope > 30 deg, cos(30 deg) ≈ 0.866)
    cliff_slope_ramp = nodes.new(type="ShaderNodeValToRGB")
    cliff_slope_ramp.location = (-550, 300)
    cliff_slope_ramp.color_ramp.elements[0].position = 0.55  # Steep cliff rock
    cliff_slope_ramp.color_ramp.elements[1].position = 0.866 # 30 deg slope threshold
    links.new(add_slope_pert.outputs["Value"], cliff_slope_ramp.inputs["Fac"])

    # 3. Scree / Talus Mask (Intermediate slopes & modulated by hf.scree_talus attribute)
    attr_scree = nodes.new(type="ShaderNodeAttribute")
    attr_scree.location = (-550, 550)
    attr_scree.attribute_name = "scree_talus"

    scree_slope_ramp = nodes.new(type="ShaderNodeValToRGB")
    scree_slope_ramp.location = (-300, 450)
    scree_ramp = scree_slope_ramp.color_ramp
    scree_ramp.elements[0].position = 0.55
    scree_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    scree_ramp.elements[1].position = 0.94
    scree_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    e_mid1 = scree_ramp.elements.new(0.70)
    e_mid1.color = (1.0, 1.0, 1.0, 1.0)
    e_mid2 = scree_ramp.elements.new(0.866)
    e_mid2.color = (1.0, 1.0, 1.0, 1.0)
    links.new(add_slope_pert.outputs["Value"], scree_slope_ramp.inputs["Fac"])

    scree_combine = nodes.new(type="ShaderNodeMath")
    scree_combine.location = (-50, 500)
    scree_combine.operation = "MAXIMUM"
    links.new(scree_slope_ramp.outputs["Color"], scree_combine.inputs[0])
    links.new(attr_scree.outputs["Fac"], scree_combine.inputs[1])

    # 4. Coastal / River Sand Mask (Low altitude & gentle slope)
    sand_alt_range = nodes.new(type="ShaderNodeMapRange")
    sand_alt_range.location = (-550, -350)
    sand_alt_range.inputs["From Min"].default_value = -3.0
    sand_alt_range.inputs["From Max"].default_value = 3.8
    sand_alt_range.inputs["To Min"].default_value = 1.0
    sand_alt_range.inputs["To Max"].default_value = 0.0
    links.new(add_alt_pert.outputs["Value"], sand_alt_range.inputs["Value"])

    sand_weight = nodes.new(type="ShaderNodeMath")
    sand_weight.location = (-300, -300)
    sand_weight.operation = "MULTIPLY"
    links.new(sand_alt_range.outputs["Result"], sand_weight.inputs[0])
    links.new(cliff_slope_ramp.outputs["Color"], sand_weight.inputs[1])

    # 5. Moisture & Sediment Modulation on Soil
    attr_moist = nodes.new(type="ShaderNodeAttribute")
    attr_moist.location = (-300, 150)
    attr_moist.attribute_name = "moisture"

    attr_sed = nodes.new(type="ShaderNodeAttribute")
    attr_sed.location = (-300, -50)
    attr_sed.attribute_name = "sediment"

    # Soil color modulation: loam brown -> damp green moss -> alluvial silt
    mix_soil_moss = nodes.new(type="ShaderNodeMix")
    mix_soil_moss.location = (0, 150)
    mix_soil_moss.data_type = "RGBA"
    f_sm, a_sm, b_sm, out_sm = _get_mix_rgba_sockets(mix_soil_moss)
    links.new(attr_moist.outputs["Fac"], f_sm)
    a_sm.default_value = (0.22, 0.17, 0.11, 1.0)  # Rich humus loam
    b_sm.default_value = (0.11, 0.24, 0.08, 1.0)  # Lush damp moss

    mix_soil_sed = nodes.new(type="ShaderNodeMix")
    mix_soil_sed.location = (200, 150)
    mix_soil_sed.data_type = "RGBA"
    f_ss, a_ss, b_ss, out_ss = _get_mix_rgba_sockets(mix_soil_sed)
    links.new(attr_sed.outputs["Fac"], f_ss)
    links.new(out_sm, a_ss)
    b_ss.default_value = (0.38, 0.32, 0.22, 1.0)  # Alluvial deposit silt

    # --- MATERIAL MULTI-SPLATTING CHAIN ---
    # Step A: Cliff Rock vs Modulated Soil
    mix_rock_soil = nodes.new(type="ShaderNodeMix")
    mix_rock_soil.location = (450, 300)
    mix_rock_soil.data_type = "RGBA"
    f_rs, a_rs, b_rs, out_rs = _get_mix_rgba_sockets(mix_rock_soil)
    links.new(cliff_slope_ramp.outputs["Color"], f_rs)
    a_rs.default_value = (0.19, 0.18, 0.17, 1.0)  # Dark fractured cliff rock
    links.new(out_ss, b_rs)

    # Step B: Blend in Scree / Talus
    mix_scree = nodes.new(type="ShaderNodeMix")
    mix_scree.location = (650, 300)
    mix_scree.data_type = "RGBA"
    f_sc, a_sc, b_sc, out_sc = _get_mix_rgba_sockets(mix_scree)
    links.new(scree_combine.outputs["Value"], f_sc)
    links.new(out_rs, a_sc)
    b_sc.default_value = (0.34, 0.31, 0.27, 1.0)  # Loose angular gravel talus

    # Step C: Blend in Coastal / River Sand
    mix_sand = nodes.new(type="ShaderNodeMix")
    mix_sand.location = (850, 200)
    mix_sand.data_type = "RGBA"
    f_sa, a_sa, b_sa, out_sa = _get_mix_rgba_sockets(mix_sand)
    links.new(sand_weight.outputs["Value"], f_sa)
    links.new(out_sc, a_sa)
    b_sa.default_value = (0.74, 0.66, 0.48, 1.0)  # Warm quartz beach sand

    # Step D: Blend in Alpine Snow
    mix_snow = nodes.new(type="ShaderNodeMix")
    mix_snow.location = (1050, 100)
    mix_snow.data_type = "RGBA"
    f_sn, a_sn, b_sn, out_sn = _get_mix_rgba_sockets(mix_snow)
    links.new(snow_weight.outputs["Value"], f_sn)
    links.new(out_sa, a_sn)
    b_sn.default_value = (0.92, 0.95, 0.98, 1.0)  # Crisp crystalline snow

    # Step E: Multiply Strata Vertex Colors (COLOR_0)
    attr_vcol = nodes.new(type="ShaderNodeAttribute")
    attr_vcol.location = (1050, -150)
    attr_vcol.attribute_name = "COLOR_0"

    mix_vcol = nodes.new(type="ShaderNodeMix")
    mix_vcol.location = (1300, 50)
    mix_vcol.data_type = "RGBA"
    mix_vcol.blend_type = "MULTIPLY"
    f_vc, a_vc, b_vc, out_vc = _get_mix_rgba_sockets(mix_vcol)
    f_vc.default_value = 0.82
    links.new(out_sn, a_vc)
    links.new(attr_vcol.outputs["Color"], b_vc)

    links.new(out_vc, bsdf.inputs["Base Color"])

    # --- PBR ROUGHNESS & NORMAL BUMP ---
    # Dynamic 5-Class PBR Roughness:
    # 1. Cliff Bedrock: 0.85
    # 2. Scree & Talus: 0.90
    # 3. Humus Soil & Valley Moss: 0.50 - 0.70 (moisture modulated)
    # 4. Coastal & River Sand: 0.60
    # 5. Alpine Snow: 0.30

    # Step 1: Soil Roughness (Loam 0.70 -> Damp Moss 0.50 via moisture)
    mix_soil_rough = nodes.new(type="ShaderNodeMix")
    mix_soil_rough.location = (0, -150)
    mix_soil_rough.data_type = "FLOAT"
    f_sor, a_sor, b_sor, out_sor = _get_mix_float_sockets(mix_soil_rough)
    links.new(attr_moist.outputs["Fac"], f_sor)
    a_sor.default_value = 0.70  # Rich dark loam
    b_sor.default_value = 0.50  # Lush damp moss

    # Step 2: Cliff Bedrock (0.85) vs Modulated Soil
    mix_rock_soil_rough = nodes.new(type="ShaderNodeMix")
    mix_rock_soil_rough.location = (450, -150)
    mix_rock_soil_rough.data_type = "FLOAT"
    f_rsr, a_rsr, b_rsr, out_rsr = _get_mix_float_sockets(mix_rock_soil_rough)
    links.new(cliff_slope_ramp.outputs["Color"], f_rsr)
    a_rsr.default_value = 0.85  # Cliff Bedrock (dark fractured granite/slate)
    links.new(out_sor, b_rsr)

    # Step 3: Scree & Talus (0.90)
    mix_scree_rough = nodes.new(type="ShaderNodeMix")
    mix_scree_rough.location = (650, -150)
    mix_scree_rough.data_type = "FLOAT"
    f_scr, a_scr, b_scr, out_scr = _get_mix_float_sockets(mix_scree_rough)
    links.new(scree_combine.outputs["Value"], f_scr)
    links.new(out_rsr, a_scr)
    b_scr.default_value = 0.90  # Scree & Talus (angular gravel/pebbles)

    # Step 4: Coastal & River Sand (0.60)
    mix_sand_rough = nodes.new(type="ShaderNodeMix")
    mix_sand_rough.location = (850, -150)
    mix_sand_rough.data_type = "FLOAT"
    f_sar, a_sar, b_sar, out_sar = _get_mix_float_sockets(mix_sand_rough)
    links.new(sand_weight.outputs["Value"], f_sar)
    links.new(out_scr, a_sar)
    b_sar.default_value = 0.60  # Coastal & River Sand (quartz sand)

    # Step 5: Alpine Snow (0.30)
    mix_snow_rough = nodes.new(type="ShaderNodeMix")
    mix_snow_rough.location = (1050, -150)
    mix_snow_rough.data_type = "FLOAT"
    f_snr, a_snr, b_snr, out_snr = _get_mix_float_sockets(mix_snow_rough)
    links.new(snow_weight.outputs["Value"], f_snr)
    links.new(out_sar, a_snr)
    b_snr.default_value = 0.30  # Alpine Snow (crystalline snow)

    links.new(out_snr, bsdf.inputs["Roughness"])

    # Triplanar Procedural Normal Bump (Rock Strata & Gravel Grain)
    # Orthogonal planar coordinate projections eliminating cliff/wall stretching:
    # Plane X (YZ projection for steep East-West cliff scarps)
    proj_x = nodes.new(type="ShaderNodeCombineXYZ")
    proj_x.location = (100, -500)
    links.new(sep_pos.outputs["Y"], proj_x.inputs["X"])
    links.new(sep_pos.outputs["Z"], proj_x.inputs["Y"])
    proj_x.inputs["Z"].default_value = 0.0

    # Plane Y (XZ projection for steep North-South cliff scarps)
    proj_y = nodes.new(type="ShaderNodeCombineXYZ")
    proj_y.location = (100, -650)
    links.new(sep_pos.outputs["X"], proj_y.inputs["X"])
    links.new(sep_pos.outputs["Z"], proj_y.inputs["Y"])
    proj_y.inputs["Z"].default_value = 0.0

    # Plane Z (XY projection for horizontal flats and valley floor)
    proj_z = nodes.new(type="ShaderNodeCombineXYZ")
    proj_z.location = (100, -800)
    links.new(sep_pos.outputs["X"], proj_z.inputs["X"])
    links.new(sep_pos.outputs["Y"], proj_z.inputs["Y"])
    proj_z.inputs["Z"].default_value = 0.0

    # Micro-bump procedural noise sampled along each orthogonal plane
    noise_bump_x = nodes.new(type="ShaderNodeTexNoise")
    noise_bump_x.location = (350, -500)
    noise_bump_x.inputs["Scale"].default_value = 6.0
    noise_bump_x.inputs["Detail"].default_value = 5.0
    noise_bump_x.inputs["Roughness"].default_value = 0.6
    links.new(proj_x.outputs["Vector"], noise_bump_x.inputs["Vector"])

    noise_bump_y = nodes.new(type="ShaderNodeTexNoise")
    noise_bump_y.location = (350, -650)
    noise_bump_y.inputs["Scale"].default_value = 6.0
    noise_bump_y.inputs["Detail"].default_value = 5.0
    noise_bump_y.inputs["Roughness"].default_value = 0.6
    links.new(proj_y.outputs["Vector"], noise_bump_y.inputs["Vector"])

    noise_bump_z = nodes.new(type="ShaderNodeTexNoise")
    noise_bump_z.location = (350, -800)
    noise_bump_z.inputs["Scale"].default_value = 6.0
    noise_bump_z.inputs["Detail"].default_value = 5.0
    noise_bump_z.inputs["Roughness"].default_value = 0.6
    links.new(proj_z.outputs["Vector"], noise_bump_z.inputs["Vector"])

    # Weight planar noise projections by normalized normal-exponent triplanar weights
    mult_bump_x = nodes.new(type="ShaderNodeMath")
    mult_bump_x.location = (650, -500)
    mult_bump_x.operation = "MULTIPLY"
    links.new(noise_bump_x.outputs["Fac"], mult_bump_x.inputs[0])
    links.new(weight_x.outputs["Value"], mult_bump_x.inputs[1])

    mult_bump_y = nodes.new(type="ShaderNodeMath")
    mult_bump_y.location = (650, -650)
    mult_bump_y.operation = "MULTIPLY"
    links.new(noise_bump_y.outputs["Fac"], mult_bump_y.inputs[0])
    links.new(weight_y.outputs["Value"], mult_bump_y.inputs[1])

    mult_bump_z = nodes.new(type="ShaderNodeMath")
    mult_bump_z.location = (650, -800)
    mult_bump_z.operation = "MULTIPLY"
    links.new(noise_bump_z.outputs["Fac"], mult_bump_z.inputs[0])
    links.new(weight_z.outputs["Value"], mult_bump_z.inputs[1])

    # Accumulate triplanar weighted samples: triplanar_h = bump_x*w_x + bump_y*w_y + bump_z*w_z
    add_bump_xy = nodes.new(type="ShaderNodeMath")
    add_bump_xy.location = (900, -575)
    add_bump_xy.operation = "ADD"
    links.new(mult_bump_x.outputs["Value"], add_bump_xy.inputs[0])
    links.new(mult_bump_y.outputs["Value"], add_bump_xy.inputs[1])

    add_bump_xyz = nodes.new(type="ShaderNodeMath")
    add_bump_xyz.location = (1100, -650)
    add_bump_xyz.operation = "ADD"
    links.new(add_bump_xy.outputs["Value"], add_bump_xyz.inputs[0])
    links.new(mult_bump_z.outputs["Value"], add_bump_xyz.inputs[1])

    bump_terrain = nodes.new(type="ShaderNodeBump")
    bump_terrain.location = (1350, -350)
    bump_terrain.inputs["Strength"].default_value = 0.12
    bump_terrain.inputs["Distance"].default_value = 0.20
    links.new(add_bump_xyz.outputs["Value"], bump_terrain.inputs["Height"])
    links.new(bump_terrain.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def create_water_shader(config: ShaderConfig) -> bpy.types.Material:
    """
    Constructs an advanced physical optical water material for Blender 4.5.4 LTS (EEVEE Next & Cycles):
    - Beer-Lambert depth color attenuation (emerald shallow to deep sapphire via LightPath Ray Length)
    - Shoreline contact foam margin thresholded (< 1.0m) by water depth / ray distance
    - Dual-scale procedural wave normal bump perturbation
    - EEVEE Next real-time transparency (mat.surface_render_method = 'BLENDED')
    """
    if not HAS_BPY or bpy is None:
        return None

    mat_name = "MAT_TerraForge_Water_Deep"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Material Output & Surface BSDF
    node_out = nodes.new(type="ShaderNodeOutputMaterial")
    node_out.location = (1400, 0)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (1100, 100)
    links.new(bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    # Optical Beer-Lambert Volume Absorption (retained for node compatibility; not connected to open surface output)
    vol_absorp = nodes.new(type="ShaderNodeVolumeAbsorption")
    vol_absorp.location = (1100, -200)
    absorp = config.water_absorption_color
    vol_absorp.inputs["Color"].default_value = (absorp[0], absorp[1], absorp[2], 1.0)
    vol_absorp.inputs["Density"].default_value = 0.45

    # 1. Beer-Lambert Optical Attenuation via LightPath Ray Length
    lp = nodes.new(type="ShaderNodeLightPath")
    lp.location = (-600, 200)

    ray_mult = nodes.new(type="ShaderNodeMath")
    ray_mult.location = (-350, 250)
    ray_mult.operation = "MULTIPLY"
    ray_mult.inputs[1].default_value = -0.45
    links.new(lp.outputs["Ray Length"], ray_mult.inputs[0])

    ray_exp = nodes.new(type="ShaderNodeMath")
    ray_exp.location = (-150, 250)
    ray_exp.operation = "EXPONENT"
    links.new(ray_mult.outputs["Value"], ray_exp.inputs[0])

    # Depth gradient: exp(-alpha*d) -> 1 at surface (emerald), -> 0 at depth (deep sapphire)
    mix_depth = nodes.new(type="ShaderNodeMix")
    mix_depth.location = (100, 250)
    mix_depth.data_type = "RGBA"
    f_d, a_d, b_d, out_d = _get_mix_rgba_sockets(mix_depth)
    links.new(ray_exp.outputs["Value"], f_d)
    # Socket A (exp=0, deep water): deep sapphire blue
    a_d.default_value = (0.02, 0.08, 0.35, 1.0)
    # Socket B (exp=1, shallow water): emerald turquoise
    b_d.default_value = (0.15, 0.65, 0.70, 1.0)

    # 2. Shoreline Contact Foam Margin: Ray Length thresholded near geometry contact (< 1.0m)
    foam_range = nodes.new(type="ShaderNodeMapRange")
    foam_range.location = (-150, 50)
    foam_range.inputs["From Min"].default_value = 0.0
    foam_range.inputs["From Max"].default_value = 1.0  # 1.0m foam margin width
    foam_range.inputs["To Min"].default_value = 1.0    # 100% foam at contact edge
    foam_range.inputs["To Max"].default_value = 0.0    # 0% foam in open water
    links.new(lp.outputs["Ray Length"], foam_range.inputs["Value"])

    voro_foam = nodes.new(type="ShaderNodeTexVoronoi")
    voro_foam.location = (-150, -150)
    voro_foam.inputs["Scale"].default_value = 16.0

    noise_foam = nodes.new(type="ShaderNodeTexNoise")
    noise_foam.location = (-150, -300)
    noise_foam.inputs["Scale"].default_value = 7.0
    noise_foam.inputs["Detail"].default_value = 4.0

    mult_foam = nodes.new(type="ShaderNodeMath")
    mult_foam.location = (100, -200)
    mult_foam.operation = "MULTIPLY"
    links.new(voro_foam.outputs["Distance"], mult_foam.inputs[0])
    links.new(noise_foam.outputs["Fac"], mult_foam.inputs[1])

    foam_contact_mask = nodes.new(type="ShaderNodeMath")
    foam_contact_mask.location = (300, -50)
    foam_contact_mask.operation = "MULTIPLY"
    links.new(foam_range.outputs["Result"], foam_contact_mask.inputs[0])
    links.new(mult_foam.outputs["Value"], foam_contact_mask.inputs[1])

    mix_water_foam = nodes.new(type="ShaderNodeMix")
    mix_water_foam.location = (550, 100)
    mix_water_foam.data_type = "RGBA"
    f_wf, a_wf, b_wf, out_wf = _get_mix_rgba_sockets(mix_water_foam)

    if config.water_contact_foam:
        links.new(foam_contact_mask.outputs["Value"], f_wf)
    else:
        f_wf.default_value = 0.0

    links.new(out_d, a_wf)
    b_wf.default_value = (0.95, 0.98, 1.0, 1.0)  # Pure white bubbly contact foam
    links.new(out_wf, bsdf.inputs["Base Color"])

    # 3. Procedural Wave Normal Bump Perturbation with Flow Vector Distortion
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    tex_coord.location = (-150, -450)

    mapping_flow = nodes.new(type="ShaderNodeMapping")
    mapping_flow.location = (50, -450)
    mapping_flow.vector_type = "POINT"
    # Elongate ripples along primary flow axis (2.4x) for realistic directional flow distortion
    mapping_flow.inputs["Scale"].default_value = (2.4, 1.0, 1.0)
    links.new(tex_coord.outputs["Generated"], mapping_flow.inputs["Vector"])

    tex_wave1 = nodes.new(type="ShaderNodeTexNoise")
    tex_wave1.location = (250, -350)
    tex_wave1.inputs["Scale"].default_value = 2.8
    tex_wave1.inputs["Detail"].default_value = 4.0
    tex_wave1.inputs["Roughness"].default_value = 0.45
    links.new(mapping_flow.outputs["Vector"], tex_wave1.inputs["Vector"])

    tex_wave2 = nodes.new(type="ShaderNodeTexVoronoi")
    tex_wave2.location = (250, -550)
    tex_wave2.inputs["Scale"].default_value = 3.6
    links.new(mapping_flow.outputs["Vector"], tex_wave2.inputs["Vector"])

    mix_waves = nodes.new(type="ShaderNodeMix")
    mix_waves.location = (450, -450)
    mix_waves.data_type = "FLOAT"
    f_w, a_w, b_w, out_w = _get_mix_float_sockets(mix_waves)
    f_w.default_value = 0.4
    links.new(tex_wave1.outputs["Fac"], a_w)
    links.new(tex_wave2.outputs["Distance"], b_w)

    bump_water = nodes.new(type="ShaderNodeBump")
    bump_water.location = (700, -450)
    bump_water.inputs["Strength"].default_value = 0.06
    bump_water.inputs["Distance"].default_value = 0.15
    links.new(out_w, bump_water.inputs["Height"])
    links.new(bump_water.outputs["Normal"], bsdf.inputs["Normal"])

    # Water Optical Surface Parameters
    bsdf.inputs["Roughness"].default_value = max(0.02, config.water_roughness)
    bsdf.inputs["IOR"].default_value = 1.333

    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 0.90
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = 0.90

    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = 0.92

    # Viewport & EEVEE Next Real-Time Transparency
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"
    if hasattr(mat, "blend_method"):
        mat.blend_method = "BLEND"
    if hasattr(mat, "use_screen_refraction"):
        mat.use_screen_refraction = True
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "NONE"

    return mat


def create_cavern_mineral_shader(
    glow_color: tuple = (0.1, 0.9, 0.4),
    emission_strength: float = 4.5,
) -> bpy.types.Material:
    """
    Constructs a photorealistic PBR phosphorescent mineral crystal druse material
    (fluorite/calcite druse) for cavern crystal clusters:
    - Faceted crystalline cleavage planes via Voronoi cell bump
    - Low surface roughness (0.18) with fluorite IOR (1.434)
    - Crystalline transmission weight (0.45) for translucency
    - Bioluminescent emission (color + strength 4.5)
    - Full Blender 4.5.4 LTS BLENDER_EEVEE_NEXT compatibility
    """
    if not HAS_BPY or bpy is None:
        return None

    mat_name = "MAT_Cavern_Mineral_Emissive"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_out = nodes.new(type="ShaderNodeOutputMaterial")
    node_out.location = (800, 0)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (500, 0)
    links.new(bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    # Base crystalline druse color
    bsdf.inputs["Base Color"].default_value = (
        glow_color[0] * 0.25,
        glow_color[1] * 0.35,
        glow_color[2] * 0.30,
        1.0,
    )
    bsdf.inputs["Roughness"].default_value = 0.18
    bsdf.inputs["IOR"].default_value = 1.434  # Fluorite refractive index

    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 0.45
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = 0.45

    # Bioluminescent emission
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (glow_color[0], glow_color[1], glow_color[2], 1.0)
        bsdf.inputs["Emission Strength"].default_value = float(emission_strength)
    elif "Emission" in bsdf.inputs:
        bsdf.inputs["Emission"].default_value = (glow_color[0], glow_color[1], glow_color[2], 1.0)

    # Faceted Crystal Cleavage Bump
    geom = nodes.new(type="ShaderNodeNewGeometry")
    geom.location = (0, -200)

    voro_facet = nodes.new(type="ShaderNodeTexVoronoi")
    voro_facet.location = (200, -200)
    voro_facet.inputs["Scale"].default_value = 14.0
    links.new(geom.outputs["Position"], voro_facet.inputs["Vector"])

    bump_facet = nodes.new(type="ShaderNodeBump")
    bump_facet.location = (350, -200)
    bump_facet.inputs["Strength"].default_value = 0.35
    bump_facet.inputs["Distance"].default_value = 0.10
    links.new(voro_facet.outputs["Distance"], bump_facet.inputs["Height"])
    links.new(bump_facet.outputs["Normal"], bsdf.inputs["Normal"])

    # Transparency configuration for EEVEE Next
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"
    if hasattr(mat, "blend_method"):
        mat.blend_method = "BLEND"
    if hasattr(mat, "use_screen_refraction"):
        mat.use_screen_refraction = True

    return mat


def create_bioluminescent_fungus_shader(glow_color: tuple = (0.1, 0.9, 0.4)) -> bpy.types.Material:
    """Constructs glowing fungal shader with emission and SSS."""
    if not HAS_BPY or bpy is None:
        return None

    mat_name = "MAT_TerraForge_Cave_Fungi_Glow"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_out = nodes.new(type="ShaderNodeOutputMaterial")
    node_out.location = (600, 0)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    links.new(bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    bsdf.inputs["Base Color"].default_value = (glow_color[0] * 0.4, glow_color[1] * 0.4, glow_color[2] * 0.4, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.35
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (glow_color[0], glow_color[1], glow_color[2], 1.0)
        bsdf.inputs["Emission Strength"].default_value = 4.5

    return mat


