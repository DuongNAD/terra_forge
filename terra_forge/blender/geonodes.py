"""
Parametric Geometry Nodes Scattering Engine.
Constructs Geometry Nodes modifier trees with exposed input sockets for
real-time, zero-code parameter tweaking by AI and developers.

Features:
- Exposed input sockets: Density, Distance_Min, Min_Altitude, Max_Altitude, Max_Slope_Deg, Water_Level, Water_Clearance, Is_Aquatic, Seed
- Cliff slope angle exclusion: normal.z >= cos(radians(Max_Slope_Deg)), excluding steep cliffs (>45°)
- Deep water exclusion: terrestrial vegetation requires elevation >= Water_Level + Water_Clearance,
  while aquatic flora/pebbles/reefs populate at/below water level.
- Poisson disk blue-noise point distribution with random rotation and scale.
"""

from __future__ import annotations
import math
from typing import Optional, Any

try:
    import bpy
    HAS_BPY = True
except ImportError:
    bpy = None
    HAS_BPY = False

from terra_forge.schema.map_config import BiomeScatterConfig


def build_biome_geonodes_tree(
    tree_name: str,
    instance_obj: Optional[Any] = None,
) -> Any:
    """
    Creates or retrieves a dedicated Geometry Node Tree with comprehensive ecological filtering masks:
    1. Altitude Range Mask (min_altitude <= z <= max_altitude)
    2. Cliff Slope Normal Mask (normal.z >= cos(radians(max_slope_deg))), excluding cliffs > 45°
    3. Deep Water Exclusion: terrestrial vegetation excluded below water; aquatic flora/pebbles allowed submerged
    4. Poisson Disk Minimum Distance distribution
    5. Random instance rotation (around Z) and variation
    All key controls are exposed via clean interface sockets for live viewport tweaking.
    """
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to build Geometry Nodes trees.")

    tree = bpy.data.node_groups.get(tree_name)
    if tree is not None:
        if instance_obj is not None:
            for n in tree.nodes:
                if n.type == "OBJECT_INFO" and "Object" in n.inputs:
                    n.inputs["Object"].default_value = instance_obj
        return tree

    tree = bpy.data.node_groups.new(name=tree_name, type="GeometryNodeTree")

    # Define Interface Sockets (Blender 4.0+ / 5.x LTS)
    if hasattr(tree, "interface"):
        tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")

        s_dens = tree.interface.new_socket("Density", in_out="INPUT", socket_type="NodeSocketFloat")
        s_dens.default_value = 0.05

        s_dist = tree.interface.new_socket("Distance_Min", in_out="INPUT", socket_type="NodeSocketFloat")
        s_dist.default_value = 2.5

        s_alt_min = tree.interface.new_socket("Min_Altitude", in_out="INPUT", socket_type="NodeSocketFloat")
        s_alt_min.default_value = -10.0

        s_alt_max = tree.interface.new_socket("Max_Altitude", in_out="INPUT", socket_type="NodeSocketFloat")
        s_alt_max.default_value = 40.0

        s_slope = tree.interface.new_socket("Max_Slope_Deg", in_out="INPUT", socket_type="NodeSocketFloat")
        s_slope.default_value = 45.0  # Cliff exclusion (> 45°)

        s_water = tree.interface.new_socket("Water_Level", in_out="INPUT", socket_type="NodeSocketFloat")
        s_water.default_value = 0.0

        s_clear = tree.interface.new_socket("Water_Clearance", in_out="INPUT", socket_type="NodeSocketFloat")
        s_clear.default_value = 0.1

        s_aqua = tree.interface.new_socket("Is_Aquatic", in_out="INPUT", socket_type="NodeSocketFloat")
        s_aqua.default_value = 0.0

        s_smin = tree.interface.new_socket("Scale_Min", in_out="INPUT", socket_type="NodeSocketFloat")
        s_smin.default_value = 0.75

        s_smax = tree.interface.new_socket("Scale_Max", in_out="INPUT", socket_type="NodeSocketFloat")
        s_smax.default_value = 1.25

        s_seed = tree.interface.new_socket("Seed", in_out="INPUT", socket_type="NodeSocketInt")
        s_seed.default_value = 42

        tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    # 1. Group Input & Group Output
    node_in = nodes.new(type="NodeGroupInput")
    node_in.location = (-1100, 0)

    node_out = nodes.new(type="NodeGroupOutput")
    node_out.location = (1100, 0)

    # 2. Distribute Points on Faces (Poisson Disk)
    dist_node = nodes.new(type="GeometryNodeDistributePointsOnFaces")
    dist_node.location = (-200, 200)
    dist_node.distribute_method = "POISSON"
    links.new(node_in.outputs["Geometry"], dist_node.inputs["Mesh"])

    if "Density" in node_in.outputs:
        links.new(node_in.outputs["Density"], dist_node.inputs["Density Max"])
    if "Distance_Min" in node_in.outputs:
        links.new(node_in.outputs["Distance_Min"], dist_node.inputs["Distance Min"])
    if "Seed" in node_in.outputs:
        links.new(node_in.outputs["Seed"], dist_node.inputs["Seed"])

    # 3. Position and Normal Inputs
    pos_node = nodes.new(type="GeometryNodeInputPosition")
    pos_node.location = (-950, -400)
    sep_pos = nodes.new(type="ShaderNodeSeparateXYZ")
    sep_pos.location = (-750, -400)
    links.new(pos_node.outputs["Position"], sep_pos.inputs["Vector"])

    norm_node = nodes.new(type="GeometryNodeInputNormal")
    norm_node.location = (-950, -150)
    sep_norm = nodes.new(type="ShaderNodeSeparateXYZ")
    sep_norm.location = (-750, -150)
    links.new(norm_node.outputs["Normal"], sep_norm.inputs["Vector"])

    # 4. Altitude Filtering (Min_Altitude <= Z <= Max_Altitude)
    cmp_alt_min = nodes.new(type="FunctionNodeCompare")
    cmp_alt_min.location = (-550, -350)
    cmp_alt_min.operation = "GREATER_EQUAL"
    links.new(sep_pos.outputs["Z"], cmp_alt_min.inputs["A"])
    if "Min_Altitude" in node_in.outputs:
        links.new(node_in.outputs["Min_Altitude"], cmp_alt_min.inputs["B"])

    cmp_alt_max = nodes.new(type="FunctionNodeCompare")
    cmp_alt_max.location = (-550, -500)
    cmp_alt_max.operation = "LESS_EQUAL"
    links.new(sep_pos.outputs["Z"], cmp_alt_max.inputs["A"])
    if "Max_Altitude" in node_in.outputs:
        links.new(node_in.outputs["Max_Altitude"], cmp_alt_max.inputs["B"])

    bool_alt = nodes.new(type="FunctionNodeBooleanMath")
    bool_alt.location = (-350, -420)
    bool_alt.operation = "AND"
    links.new(cmp_alt_min.outputs["Result"], bool_alt.inputs[0])
    links.new(cmp_alt_max.outputs["Result"], bool_alt.inputs[1])

    # 5. Cliff Slope Normal Filtering: Nz >= cos(radians(Max_Slope_Deg))
    # Excludes steep vertical and cliff faces (> 45 deg)
    math_rad = nodes.new(type="ShaderNodeMath")
    math_rad.location = (-750, 50)
    math_rad.operation = "RADIANS"
    if "Max_Slope_Deg" in node_in.outputs:
        links.new(node_in.outputs["Max_Slope_Deg"], math_rad.inputs[0])

    math_cos = nodes.new(type="ShaderNodeMath")
    math_cos.location = (-550, 50)
    math_cos.operation = "COSINE"
    links.new(math_rad.outputs["Value"], math_cos.inputs[0])

    cmp_slope = nodes.new(type="FunctionNodeCompare")
    cmp_slope.location = (-350, -100)
    cmp_slope.operation = "GREATER_EQUAL"
    links.new(sep_norm.outputs["Z"], cmp_slope.inputs["A"])
    links.new(math_cos.outputs["Value"], cmp_slope.inputs["B"])

    # Combine Altitude and Slope
    bool_alt_slope = nodes.new(type="FunctionNodeBooleanMath")
    bool_alt_slope.location = (-150, -250)
    bool_alt_slope.operation = "AND"
    links.new(bool_alt.outputs["Boolean"], bool_alt_slope.inputs[0])
    links.new(cmp_slope.outputs["Result"], bool_alt_slope.inputs[1])

    # 6. Deep Water Exclusion (Terrestrial vs. Aquatic)
    # Terrestrial: Z >= Water_Level + Water_Clearance
    # Aquatic: Z <= Water_Level + Water_Clearance
    math_water_thresh = nodes.new(type="ShaderNodeMath")
    math_water_thresh.location = (-750, -650)
    math_water_thresh.operation = "ADD"
    if "Water_Level" in node_in.outputs:
        links.new(node_in.outputs["Water_Level"], math_water_thresh.inputs[0])
    if "Water_Clearance" in node_in.outputs:
        links.new(node_in.outputs["Water_Clearance"], math_water_thresh.inputs[1])

    cmp_water_terr = nodes.new(type="FunctionNodeCompare")
    cmp_water_terr.location = (-550, -650)
    cmp_water_terr.operation = "GREATER_EQUAL"
    links.new(sep_pos.outputs["Z"], cmp_water_terr.inputs["A"])
    links.new(math_water_thresh.outputs["Value"], cmp_water_terr.inputs["B"])

    cmp_water_aqua = nodes.new(type="FunctionNodeCompare")
    cmp_water_aqua.location = (-550, -800)
    cmp_water_aqua.operation = "LESS_EQUAL"
    links.new(sep_pos.outputs["Z"], cmp_water_aqua.inputs["A"])
    links.new(math_water_thresh.outputs["Value"], cmp_water_aqua.inputs["B"])

    # Switch between terrestrial water mask and aquatic water mask based on Is_Aquatic
    cmp_is_aquatic = nodes.new(type="FunctionNodeCompare")
    cmp_is_aquatic.location = (-550, -950)
    cmp_is_aquatic.operation = "GREATER_THAN"
    if "Is_Aquatic" in node_in.outputs:
        links.new(node_in.outputs["Is_Aquatic"], cmp_is_aquatic.inputs["A"])
    cmp_is_aquatic.inputs["B"].default_value = 0.5

    sw_water = nodes.new(type="GeometryNodeSwitch")
    sw_water.location = (-350, -750)
    sw_water.input_type = "BOOLEAN"
    links.new(cmp_is_aquatic.outputs["Result"], sw_water.inputs["Switch"])
    links.new(cmp_water_terr.outputs["Result"], sw_water.inputs["False"])
    links.new(cmp_water_aqua.outputs["Result"], sw_water.inputs["True"])

    # Final Combined Selection Mask: (Altitude & Slope) AND Water_Mask
    bool_filter_final = nodes.new(type="FunctionNodeBooleanMath")
    bool_filter_final.location = (50, -400)
    bool_filter_final.operation = "AND"
    links.new(bool_alt_slope.outputs["Boolean"], bool_filter_final.inputs[0])
    links.new(sw_water.outputs["Output"], bool_filter_final.inputs[1])

    links.new(bool_filter_final.outputs["Boolean"], dist_node.inputs["Selection"])

    # 7. Object Info (Instance prototype)
    obj_info = nodes.new(type="GeometryNodeObjectInfo")
    obj_info.location = (-50, 450)
    obj_info.transform_space = "RELATIVE"
    if instance_obj:
        obj_info.inputs["Object"].default_value = instance_obj

    # 8. Instance on Points
    inst_node = nodes.new(type="GeometryNodeInstanceOnPoints")
    inst_node.location = (250, 200)
    links.new(dist_node.outputs["Points"], inst_node.inputs["Points"])
    links.new(obj_info.outputs["Geometry"], inst_node.inputs["Instance"])

    # Random Z Rotation
    rand_rot = nodes.new(type="FunctionNodeRandomValue")
    rand_rot.location = (0, 50)
    rand_rot.data_type = "FLOAT_VECTOR"
    rand_rot.inputs["Min"].default_value = (0.0, 0.0, 0.0)
    rand_rot.inputs["Max"].default_value = (0.0, 0.0, 6.28318)
    links.new(rand_rot.outputs["Value"], inst_node.inputs["Rotation"])

    # Random Scale Variation
    rand_scale = nodes.new(type="FunctionNodeRandomValue")
    rand_scale.location = (0, -100)
    rand_scale.data_type = "FLOAT"
    if "Scale_Min" in node_in.outputs:
        links.new(node_in.outputs["Scale_Min"], rand_scale.inputs["Min"])
    else:
        rand_scale.inputs["Min"].default_value = 0.75
    if "Scale_Max" in node_in.outputs:
        links.new(node_in.outputs["Scale_Max"], rand_scale.inputs["Max"])
    else:
        rand_scale.inputs["Max"].default_value = 1.25
    links.new(rand_scale.outputs["Value"], inst_node.inputs["Scale"])

    # Join Geometry (Terrain + Instances)
    join_node = nodes.new(type="GeometryNodeJoinGeometry")
    join_node.location = (650, 0)
    links.new(node_in.outputs["Geometry"], join_node.inputs["Geometry"])
    links.new(inst_node.outputs["Instances"], join_node.inputs["Geometry"])

    links.new(join_node.outputs["Geometry"], node_out.inputs["Geometry"])

    return tree


def apply_biome_scatter_modifier(
    target_obj: Any,
    instance_obj: Any,
    config: BiomeScatterConfig,
    water_level: Optional[float] = None,
    water_clearance: float = 0.45,
) -> Any:
    """
    Applies the parametric Geometry Nodes modifier to target_obj with specified config values.
    Exposes and sets all sockets: Density, Distance_Min, Altitude range, Cliff slope (<45°),
    Water clearance/exclusion, and Is_Aquatic flag.
    """
    if not HAS_BPY or bpy is None:
        raise RuntimeError("Blender environment (bpy) is required to apply Geometry Nodes modifiers.")

    tree_name = f"TerraForge_Tree_{config.name}"
    tree = build_biome_geonodes_tree(tree_name, instance_obj)
    mod_name = f"Scatter_{config.name}"

    mod = target_obj.modifiers.get(mod_name)
    if mod is None:
        mod = target_obj.modifiers.new(name=mod_name, type="NODES")

    mod.node_group = tree

    # Determine whether biome is aquatic
    proto = (config.prototype or "").lower()
    name = (config.name or "").lower()
    aquatic_keywords = ["seaweed", "waterweed", "pebble", "coral", "reef", "aquatic", "kelp", "submerged"]
    is_aquatic = False
    if any(kw in proto or kw in name for kw in aquatic_keywords):
        is_aquatic = True
    elif config.max_altitude <= 0.5 and any(kw in proto or kw in name for kw in ["reed", "reeds"]):
        is_aquatic = True

    # Configure interface socket values
    if hasattr(tree, "interface"):
        for socket in tree.interface.items_tree:
            if socket.in_out == "INPUT":
                val = None
                if socket.name == "Density":
                    val = float(config.density)
                elif socket.name == "Distance_Min":
                    val = float(config.poisson_dist_min)
                elif socket.name == "Min_Altitude":
                    val = float(config.min_altitude)
                elif socket.name == "Max_Altitude":
                    val = float(config.max_altitude)
                elif socket.name == "Max_Slope_Deg":
                    val = float(config.max_slope_deg)
                elif socket.name == "Water_Level":
                    if water_level is not None:
                        val = float(water_level)
                    else:
                        val = float(getattr(config, "water_level", 0.0))
                elif socket.name == "Water_Clearance":
                    val = float(water_clearance)
                elif socket.name == "Is_Aquatic":
                    val = 1.0 if is_aquatic else 0.0
                elif socket.name == "Scale_Min":
                    val = float(config.scale_min)
                elif socket.name == "Scale_Max":
                    val = float(config.scale_max)
                elif socket.name == "Seed":
                    val = int(config.seed)

                if val is not None:
                    if hasattr(socket, "default_value"):
                        socket.default_value = val
                    try:
                        mod[socket.identifier] = val
                    except Exception:
                        pass

    if instance_obj is not None and hasattr(tree, "nodes"):
        for n in tree.nodes:
            if n.type == "OBJECT_INFO" and "Object" in n.inputs:
                n.inputs["Object"].default_value = instance_obj

    return mod
