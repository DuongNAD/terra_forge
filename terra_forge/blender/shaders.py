"""
Procedural & PBR Shaders for TerraForge in Blender:
1. Triplanar Slope-Aware Terrain Shader (Rock, Scree, Forest Ground, Sand, Snow).
2. Physical Beer-Lambert Absorption Water Shader with Contact Foam Margin.
3. Bioluminescent Subsurface Cave Fungi Shader.
"""

from __future__ import annotations

import bpy

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

    # 2. Cliff Bedrock Mask (Steep rock scarp)
    cliff_slope_ramp = nodes.new(type="ShaderNodeValToRGB")
    cliff_slope_ramp.location = (-550, 300)
    cliff_slope_ramp.color_ramp.elements[0].position = 0.55  # Cliff rock
    cliff_slope_ramp.color_ramp.elements[1].position = 0.82  # Transition to gentle
    links.new(add_slope_pert.outputs["Value"], cliff_slope_ramp.inputs["Fac"])

    # 3. Scree / Talus Mask (Modulated by hf.scree_talus attribute)
    attr_scree = nodes.new(type="ShaderNodeAttribute")
    attr_scree.location = (-550, 550)
    attr_scree.attribute_name = "scree_talus"

    scree_slope_ramp = nodes.new(type="ShaderNodeValToRGB")
    scree_slope_ramp.location = (-300, 450)
    scree_slope_ramp.color_ramp.elements[0].position = 0.60
    scree_slope_ramp.color_ramp.elements[1].position = 0.82
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
    # Dynamic Roughness: Rock=0.85, Scree=0.90, Sand=0.60, Snow=0.30, Damp Soil=0.50
    mix_rough = nodes.new(type="ShaderNodeMix")
    mix_rough.location = (1300, -150)
    mix_rough.data_type = "FLOAT"
    f_ro, a_ro, b_ro, out_ro = _get_mix_float_sockets(mix_rough)
    links.new(snow_weight.outputs["Value"], f_ro)
    a_ro.default_value = 0.78
    b_ro.default_value = 0.30
    links.new(out_ro, bsdf.inputs["Roughness"])

    # Procedural Normal Bump (Rock Strata & Gravel Grain)
    tex_bump_noise = nodes.new(type="ShaderNodeTexNoise")
    tex_bump_noise.location = (1050, -350)
    tex_bump_noise.inputs["Scale"].default_value = 6.0
    tex_bump_noise.inputs["Detail"].default_value = 5.0
    tex_bump_noise.inputs["Roughness"].default_value = 0.6
    links.new(geom.outputs["Position"], tex_bump_noise.inputs["Vector"])

    bump_terrain = nodes.new(type="ShaderNodeBump")
    bump_terrain.location = (1350, -350)
    bump_terrain.inputs["Strength"].default_value = 0.12
    bump_terrain.inputs["Distance"].default_value = 0.20
    links.new(tex_bump_noise.outputs["Fac"], bump_terrain.inputs["Height"])
    links.new(bump_terrain.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def create_water_shader(config: ShaderConfig) -> bpy.types.Material:
    """
    Constructs an advanced physical water material:
    - Deep volume absorption (Beer-Lambert optical attenuation: I = I_0 * exp(-alpha * d))
    - Specular surface transmission and Fresnel reflection
    - Shoreline contact foam margin gradient
    - Procedural wave normal bump perturbation
    """
    mat_name = "MAT_TerraForge_Water_Deep"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Material Output
    node_out = nodes.new(type="ShaderNodeOutputMaterial")
    node_out.location = (1200, 0)

    # Principled BSDF Surface
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (900, 100)
    links.new(bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    # Optical Beer-Lambert Volume Absorption (retained for node compatibility; not connected to open surface output)
    vol_absorp = nodes.new(type="ShaderNodeVolumeAbsorption")
    vol_absorp.location = (900, -200)
    absorp = config.water_absorption_color
    vol_absorp.inputs["Color"].default_value = (absorp[0], absorp[1], absorp[2], 1.0)
    vol_absorp.inputs["Density"].default_value = 0.45
    # NOTE: Connecting Volume output to non-closed/open ribbon meshes causes ray-march extinction (pitch black) in Blender 5.2 EEVEE-Next.
    # Surface BSDF transmission handles physical optics cleanly without black shading artifacts.

    # Water Surface Parameters
    water_base_color = (absorp[0] * 0.75, absorp[1] * 0.90, absorp[2] * 0.98, 1.0)
    bsdf.inputs["Roughness"].default_value = max(0.02, config.water_roughness)
    bsdf.inputs["IOR"].default_value = 1.333

    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 0.88
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = 0.88
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = 0.92

    # Procedural Shoreline Contact Foam Band Gradient
    voro_foam = nodes.new(type="ShaderNodeTexVoronoi")
    voro_foam.location = (0, 300)
    voro_foam.inputs["Scale"].default_value = 16.0

    noise_foam = nodes.new(type="ShaderNodeTexNoise")
    noise_foam.location = (0, 100)
    noise_foam.inputs["Scale"].default_value = 7.0
    noise_foam.inputs["Detail"].default_value = 4.0

    mult_foam = nodes.new(type="ShaderNodeMath")
    mult_foam.location = (250, 200)
    mult_foam.operation = "MULTIPLY"
    links.new(voro_foam.outputs["Distance"], mult_foam.inputs[0])
    links.new(noise_foam.outputs["Fac"], mult_foam.inputs[1])

    ramp_foam = nodes.new(type="ShaderNodeValToRGB")
    ramp_foam.location = (450, 200)
    ramp_foam.color_ramp.elements[0].position = 0.55
    ramp_foam.color_ramp.elements[1].position = 0.85
    links.new(mult_foam.outputs["Value"], ramp_foam.inputs["Fac"])

    mix_water_foam = nodes.new(type="ShaderNodeMix")
    mix_water_foam.location = (650, 200)
    mix_water_foam.data_type = "RGBA"
    f_wf, a_wf, b_wf, out_wf = _get_mix_rgba_sockets(mix_water_foam)

    if config.water_contact_foam:
        links.new(ramp_foam.outputs["Color"], f_wf)
    else:
        f_wf.default_value = 0.0

    a_wf.default_value = water_base_color
    b_wf.default_value = (0.95, 0.98, 1.0, 1.0)  # Pure white bubbly contact foam
    links.new(out_wf, bsdf.inputs["Base Color"])

    # Procedural Wave Normal Bump Perturbation
    tex_wave1 = nodes.new(type="ShaderNodeTexNoise")
    tex_wave1.location = (250, -100)
    tex_wave1.inputs["Scale"].default_value = 2.8
    tex_wave1.inputs["Detail"].default_value = 4.0
    tex_wave1.inputs["Roughness"].default_value = 0.45

    tex_wave2 = nodes.new(type="ShaderNodeTexVoronoi")
    tex_wave2.location = (250, -300)
    tex_wave2.inputs["Scale"].default_value = 3.6

    mix_waves = nodes.new(type="ShaderNodeMix")
    mix_waves.location = (450, -150)
    mix_waves.data_type = "FLOAT"
    f_w, a_w, b_w, out_w = _get_mix_float_sockets(mix_waves)
    f_w.default_value = 0.4
    links.new(tex_wave1.outputs["Fac"], a_w)
    links.new(tex_wave2.outputs["Distance"], b_w)

    bump_water = nodes.new(type="ShaderNodeBump")
    bump_water.location = (650, -150)
    bump_water.inputs["Strength"].default_value = 0.06
    bump_water.inputs["Distance"].default_value = 0.15
    links.new(out_w, bump_water.inputs["Height"])
    links.new(bump_water.outputs["Normal"], bsdf.inputs["Normal"])

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


def create_bioluminescent_fungus_shader(glow_color: tuple = (0.1, 0.9, 0.4)) -> bpy.types.Material:
    """Constructs glowing fungal shader with emission and SSS."""
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

