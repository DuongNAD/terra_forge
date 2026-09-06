#!/usr/bin/env python3
"""
TerraForge Model Context Protocol (MCP) Server.
Connects Antigravity (and other AI agents) to Blender 5.2+ for autonomous,
high-fidelity 3D terrain and diorama generation, real-time inspection, and tuning.

Protocol: JSON-RPC 2.0 over standard I/O (MCP stdio transport).
Pure standard library implementation — zero extra server dependencies.
"""

from __future__ import annotations
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {
    "name": "terraforge",
    "version": "0.1.0",
}

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

PRESETS_DIR = ROOT_DIR / "terra_forge" / "schema" / "presets"

from terra_forge.blender.bridge import BlenderBridge


def _send_to_blender_socket(code: str, host: str = "localhost", port: int = 9876, timeout: float = 15.0) -> dict:
    """Sends python code directly to the active Blender MCP socket (blender_mcp addon)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))

        payload = {
            "type": "execute_code",
            "params": {
                "code": code,
                "user_prompt": "TerraForge MCP automated command",
            },
        }
        s.sendall(json.dumps(payload).encode("utf-8"))

        # Read response
        chunks = []
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            # Try to parse early if complete json
            try:
                data = json.loads(b"".join(chunks).decode("utf-8"))
                s.close()
                return data
            except json.JSONDecodeError:
                continue

        s.close()
        full_res = b"".join(chunks).decode("utf-8")
        return json.loads(full_res) if full_res else {"status": "error", "message": "Empty response from Blender"}
    except ConnectionRefusedError:
        return {"status": "error", "message": f"Blender MCP socket refused connection on {host}:{port}. Is Blender open with MCP addon running?"}
    except Exception as e:
        return {"status": "error", "message": f"Blender socket error: {str(e)}"}


# -----------------------------------------------------------------------------
# MCP Tool Implementations
# -----------------------------------------------------------------------------

def t_list_presets(args: dict) -> dict:
    """Lists all landscape presets available in TerraForge."""
    presets = []
    for p in sorted(PRESETS_DIR.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            presets.append({
                "id": p.stem,
                "name": data.get("name", p.stem),
                "theme": data.get("theme", ""),
                "peaks_count": len(data.get("peaks", [])),
                "biomes": [b.get("name") for b in data.get("biomes", [])],
                "lake_level": data.get("hydrology", {}).get("lake_level", 4.5),
            })
        except Exception as e:
            presets.append({"id": p.stem, "error": str(e)})

    return {
        "presets_count": len(presets),
        "presets": presets,
    }


def t_get_preset(args: dict) -> dict:
    """Gets the full JSON configuration for a preset."""
    preset_name = args.get("preset", "alpine_fjord")
    preset_file = PRESETS_DIR / f"{preset_name}.json"
    if not preset_file.exists():
        return {"status": "error", "message": f"Preset '{preset_name}' not found."}

    with open(preset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "preset": preset_name,
        "config": data,
    }


def t_generate(args: dict) -> dict:
    """Generates a complete 3D terrain diorama via headless Blender or live viewport."""
    from terra_forge.blender.bridge import BlenderBridge

    preset = args.get("preset", "alpine_fjord")
    seed = args.get("seed")
    mode = args.get("mode", "headless")
    render_preview = args.get("render_preview", True)
    out_dir = args.get("out_dir", "models")

    # Load base config
    if isinstance(preset, dict):
        cfg_dict = preset
    else:
        preset_file = PRESETS_DIR / f"{preset}.json"
        if not preset_file.exists():
            return {"status": "error", "message": f"Preset '{preset}' not found."}
        with open(preset_file, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

    if seed is not None:
        cfg_dict["seed"] = int(seed)

    bridge = BlenderBridge()

    if mode == "live":
        # Check if Blender MCP is listening on port 9876
        live_code = bridge.generate_live_blender_code(cfg_dict)
        socket_resp = _send_to_blender_socket(live_code)
        if socket_resp.get("status") == "success":
            return {
                "status": "success",
                "mode": "live",
                "message": "Map successfully generated directly in active Blender viewport!",
                "blender_response": socket_resp.get("result"),
            }
        else:
            # Fallback advice
            return {
                "status": "warning",
                "mode": "live",
                "socket_error": socket_resp.get("message"),
                "hint": "Could not communicate with active Blender GUI socket. Run in mode='headless' or ensure Blender is open with MCP Server active.",
            }
    else:
        # Headless generation
        try:
            results = bridge.run_headless(cfg_dict, render_preview=render_preview, output_dir=out_dir)
            return results
        except Exception as e:
            return {"status": "error", "message": str(e)}


def t_tweak_socket(args: dict) -> dict:
    """Tweaks an exposed Geometry Node socket in active Blender viewport."""
    biome_name = args.get("biome_name")
    socket_name = args.get("socket_name", "Density")
    value = args.get("value")

    if not biome_name or value is None:
        return {"status": "error", "message": "biome_name and value are required."}

    from terra_forge.blender.bridge import BlenderBridge
    mod_name = f"Scatter_{biome_name}"
    code = BlenderBridge.generate_modifier_tweak_code(mod_name, socket_name, value)

    resp = _send_to_blender_socket(code)
    return {
        "status": resp.get("status", "unknown"),
        "tweak": {
            "modifier": mod_name,
            "socket": socket_name,
            "value": value,
        },
        "response": resp,
    }


def t_inspect(args: dict) -> dict:
    """Renders fresh multi-angle inspection snapshots and 2x2 vision grid."""
    out_dir = args.get("output_dir", "previews")
    cmd = [
        sys.executable,
        "-m",
        "terra_forge.cli",
        "generate",
        "-p",
        "alpine_fjord",
        "-r",
        "-o",
        "models",
    ]
    # Fast re-render
    try:
        bridge = BlenderBridge()
        # Renders snapshots via runner
        preset_file = PRESETS_DIR / "alpine_fjord.json"
        with open(preset_file, "r") as f:
            cfg = json.load(f)
        results = bridge.run_headless(cfg, render_preview=True)
        return {
            "status": "success",
            "preview_grid": results.get("preview_grid"),
            "validation": results.get("validation"),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def t_doctor(args: dict) -> dict:
    """Runs full system and environment diagnostic check."""
    bridge = BlenderBridge()
    blender_ok = os.path.exists(bridge.blender_bin)

    # Check Blender MCP port 9876
    mcp_active = False
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect(("localhost", 9876))
        s.close()
        mcp_active = True
    except Exception:
        mcp_active = False

    return {
        "python_version": sys.version,
        "terraforge_root": str(ROOT_DIR),
        "blender_executable": bridge.blender_bin,
        "blender_found": blender_ok,
        "blender_mcp_socket_active": mcp_active,
        "presets_available": [p.stem for p in PRESETS_DIR.glob("*.json")],
        "status": "healthy" if blender_ok else "attention_required",
    }


# Tool Definitions
TOOLS = [
    (
        "tf_list_presets",
        "Lấy danh sách các preset cảnh quan 3D có sẵn trong TerraForge kèm đặc tả địa chất và sinh thái.",
        {},
        t_list_presets,
    ),
    (
        "tf_get_preset",
        "Lấy toàn bộ cấu hình JSON (MapConfig) của một preset để đọc hoặc tùy biến trước khi sinh địa hình.",
        {
            "preset": ("string", "Tên preset (ví dụ: 'alpine_fjord', 'volcanic_marsh', 'karst_cavern')"),
        },
        t_get_preset,
    ),
    (
        "tf_generate",
        "Sinh thế giới 3D diorama hoàn chỉnh (địa hình xói mòn thủy lực, sông hồ, hang karst, rải cây cối Geometry Nodes, PBR materials, Three.js GLB, .blend master, và ma trận độ cao heightmap JSON).",
        {
            "preset": ("string", "Tên preset hoặc một object cấu hình MapConfig JSON tùy biến"),
            "seed": ("integer", "Số hạt ngẫu nhiên (seed) ghi đè"),
            "mode": ("string", "Chế độ chạy: 'headless' (mặc định, xuất file GLB/.blend) hoặc 'live' (bơm trực tiếp vào viewport Blender đang mở)"),
            "render_preview": ("boolean", "Có render 4 góc nhìn chuẩn và ghép thành ảnh lưới 2x2 hay không (mặc định: true)"),
            "out_dir": ("string", "Thư mục lưu trữ asset xuất ra (mặc định: 'models')"),
        },
        t_generate,
    ),
    (
        "tf_tweak_socket",
        "Vi chỉnh tham số thời gian thực trên Geometry Nodes của Blender (mật độ cây Density, khoảng cách Distance_Min, độ dốc Max_Slope_Deg) mà không cần sinh lại toàn bộ địa hình.",
        {
            "biome_name": ("string", "Tên biome (ví dụ: 'Alpine_High_Pine', 'Basalt_Slope_Conifers', 'Karst_Cliff_Moss')"),
            "socket_name": ("string", "Tên socket cần chỉnh: 'Density', 'Distance_Min', 'Min_Altitude', 'Max_Altitude', 'Max_Slope_Deg', 'Seed'"),
            "value": ("number", "Giá trị số mới cần gán cho socket"),
        },
        t_tweak_socket,
    ),
    (
        "tf_inspect",
        "Chụp 4 góc máy chuẩn (Top-Down, SW Iso, NE Iso, Water Margin) và trả về đường dẫn ảnh lưới 2x2 phục vụ Vision LLM đánh giá chất lượng thị giác.",
        {
            "output_dir": ("string", "Thư mục lưu ảnh preview (mặc định: 'previews')"),
        },
        t_inspect,
    ),
    (
        "tf_doctor",
        "Chẩn đoán sức khỏe hệ thống: kiểm tra Python, Blender 5.2 executable, Blender MCP socket 9876, và các preset.",
        {},
        t_doctor,
    ),
]

HANDLERS = {name: fn for name, _, _, fn in TOOLS}


def tool_schema() -> list[dict]:
    out = []
    for name, desc, props, _ in TOOLS:
        schema = {
            "type": "object",
            "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()},
        }
        out.append({"name": name, "description": desc, "inputSchema": schema})
    return out


# -----------------------------------------------------------------------------
# JSON-RPC Protocol Transport (stdio)
# -----------------------------------------------------------------------------

def reply(rid, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": rid}
    if error:
        msg["error"] = error
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def handle_request(req: dict):
    m = req.get("method")
    rid = req.get("id")
    params = req.get("params") or {}

    if m == "initialize":
        return reply(rid, {
            "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
            "instructions": (
                "TerraForge MCP Server: Chuyên tạo và tinh chỉnh địa hình 3D AAA, diorama kín nước (watertight), "
                "mô phỏng xói mòn địa chất (hydraulic erosion), rải sinh vật Geometry Nodes, và xuất asset game chuẩn Three.js/GLB. "
                "Hỗ trợ cả xuất file Headless và cập nhật Viewport thời gian thực qua Blender."
            ),
        })

    if m in ("notifications/initialized", "initialized"):
        return

    if m == "ping":
        return reply(rid, {})

    if m == "tools/list":
        return reply(rid, {"tools": tool_schema()})

    if m == "tools/call":
        name = params.get("name")
        fn = HANDLERS.get(name)
        if not fn:
            return reply(rid, error={"code": -32601, "message": f"Tool '{name}' not found."})

        arguments = params.get("arguments") or {}
        try:
            res = fn(arguments)
            # Standard MCP tool response wrapper
            return reply(rid, {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(res, indent=2, ensure_ascii=False),
                    }
                ]
            })
        except Exception as e:
            return reply(rid, error={"code": -32603, "message": str(e)})

    # Fallback for unknown methods
    if rid is not None:
        reply(rid, error={"code": -32601, "message": f"Method '{m}' not supported."})


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            handle_request(req)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"Invalid JSON: {e}\n")


if __name__ == "__main__":
    main()
