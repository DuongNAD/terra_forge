"""
TerraForge Blender Addon.
Provides a native 3D Viewport Sidebar (N-Panel) interface in Blender 4.x / 5.x LTS
to select presets, adjust parameters, generate dioramas, and export assets in 1 click.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

bl_info = {
    "name": "TerraForge World Generator",
    "author": "Antigravity",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > TerraForge",
    "description": "Procedural AAA 3D Diorama & Terrain Generation Engine",
    "category": "3D View",
}

try:
    import bpy
    from bpy.props import StringProperty, IntProperty, FloatProperty, EnumProperty, BoolProperty
    from bpy.types import Panel, Operator, PropertyGroup
except ImportError:
    bpy = None


# Ensure terra_forge package is discoverable
def _ensure_sys_path():
    pkg_dir = Path(__file__).resolve().parent.parent.parent
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))


PRESET_ITEMS = [
    ("alpine_fjord", "Alpine Fjord", "High alpine summits, glaciated valleys, and glacial lake"),
    ("volcanic_marsh", "Volcanic Marsh", "Caldera dome, basalt formations, and wetland marshes"),
    ("karst_cavern", "Karst Tower Cavern", "Tower karst pillars, subterranean chambers, and speleothems"),
]


class TerraForgeSceneProperties(PropertyGroup):
    preset: EnumProperty(
        name="Preset",
        description="Choose landscape profile preset",
        items=PRESET_ITEMS,
        default="alpine_fjord",
    )
    seed: IntProperty(
        name="Seed",
        description="Random seed for topography and erosion",
        default=42,
        min=0,
        max=999999,
    )
    erosion_droplets: IntProperty(
        name="Erosion Droplets",
        description="Number of hydraulic erosion droplet iterations",
        default=25000,
        min=0,
        max=100000,
        step=5000,
    )
    lake_level: FloatProperty(
        name="Water Level",
        description="Elevation of central lake and wetland table",
        default=4.5,
        min=-10.0,
        max=20.0,
    )
    tree_density: FloatProperty(
        name="Forest Density",
        description="Vegetation scatter density modifier",
        default=0.08,
        min=0.01,
        max=0.5,
        precision=3,
    )
    export_glb_on_gen: BoolProperty(
        name="Auto Export GLB",
        description="Export Three.js ready GLB asset upon generation",
        default=True,
    )


class TERRAFORGE_OT_generate(Operator):
    """Generates the full procedural diorama in the active scene"""
    bl_idname = "terraforge.generate"
    bl_label = "Generate 3D World"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        _ensure_sys_path()
        from terra_forge.blender.runner import run_pipeline

        props = context.scene.terraforge
        pkg_dir = Path(__file__).resolve().parent.parent.parent
        preset_file = pkg_dir / "terra_forge" / "schema" / "presets" / f"{props.preset}.json"

        if not preset_file.exists():
            self.report({"ERROR"}, f"Preset not found: {preset_file}")
            return {"CANCELLED"}

        with open(preset_file, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        # Apply user overrides from N-panel
        cfg_dict["seed"] = props.seed
        cfg_dict["erosion"]["droplet_count"] = props.erosion_droplets
        cfg_dict["hydrology"]["lake_level"] = props.lake_level
        cfg_dict["export"]["export_glb"] = props.export_glb_on_gen

        # Adjust main biome density
        if cfg_dict.get("biomes") and len(cfg_dict["biomes"]) > 0:
            cfg_dict["biomes"][0]["density"] = props.tree_density

        try:
            results = run_pipeline(cfg_dict, render_preview=False)
            val = results.get("validation", {})
            msg = f"TerraForge: '{results.get('name')}' generated in {results.get('execution_time_sec')}s! (Passed: {val.get('passed')})"
            self.report({"INFO"}, msg)

            # Redraw viewports
            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()

            return {"FINISHED"}
        except Exception as e:
            self.report({"ERROR"}, f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"CANCELLED"}


class TERRAFORGE_OT_set_camera(Operator):
    """Switches active viewport camera to standard inspection view"""
    bl_idname = "terraforge.set_camera"
    bl_label = "Set Camera"

    camera_preset: StringProperty(default="topdown")

    def execute(self, context):
        cam_name = f"CAM_Inspect_{self.camera_preset}"
        cam_obj = bpy.data.objects.get(cam_name)
        if cam_obj:
            context.scene.camera = cam_obj
            self.report({"INFO"}, f"Switched view to {cam_name}")
            return {"FINISHED"}
        else:
            self.report({"WARNING"}, f"Camera '{cam_name}' not found. Generate diorama first.")
            return {"CANCELLED"}


class TERRAFORGE_PT_main_panel(Panel):
    """Creates a TerraForge Panel in the 3D Viewport N-Sidebar"""
    bl_label = "TerraForge Generator"
    bl_idname = "TERRAFORGE_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "TerraForge"

    def draw(self, context):
        layout = self.layout
        props = context.scene.terraforge

        # Preset Selection Box
        box = layout.box()
        box.label(text="World Profile & Presets", icon="WORLD")
        box.prop(props, "preset", text="")
        box.prop(props, "seed")

        # Geological & Environmental Parameters
        box_geo = layout.box()
        box_geo.label(text="Geology & Hydrology", icon="MOD_FLUID")
        box_geo.prop(props, "erosion_droplets")
        box_geo.prop(props, "lake_level")
        box_geo.prop(props, "tree_density")

        # Generation Button
        col = layout.column(align=True)
        col.scale_y = 1.6
        col.operator("terraforge.generate", icon="PLAY", text="Generate Diorama")

        # Export & Inspection Subpanel
        box_inspect = layout.box()
        box_inspect.label(text="Inspection Cameras", icon="CAMERA_DATA")
        row = box_inspect.row(align=True)
        op1 = row.operator("terraforge.set_camera", text="Top-Down")
        op1.camera_preset = "topdown"
        op2 = row.operator("terraforge.set_camera", text="SW Iso")
        op2.camera_preset = "iso_sw"

        row2 = box_inspect.row(align=True)
        op3 = row2.operator("terraforge.set_camera", text="NE Iso")
        op3.camera_preset = "iso_ne"
        op4 = row2.operator("terraforge.set_camera", text="Water Margin")
        op4.camera_preset = "closeup_water"


classes = (
    TerraForgeSceneProperties,
    TERRAFORGE_OT_generate,
    TERRAFORGE_OT_set_camera,
    TERRAFORGE_PT_main_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.terraforge = bpy.props.PointerProperty(type=TerraForgeSceneProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.terraforge


if __name__ == "__main__":
    register()
