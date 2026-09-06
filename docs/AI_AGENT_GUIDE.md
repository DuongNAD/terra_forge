# TerraForge: Sổ Tay Tích Hợp Cho AI Coding Agents (AI Agent Integration Guide)

> **Mục tiêu**: Cung cấp tài liệu kỹ thuật chi tiết dành riêng cho các AI Coding Agents (**Google Antigravity 2.0**, **Claude Desktop / Claude Code**, **OpenAI Codex / Cursor**) để tự động hóa việc giao tiếp, sinh cấu hình, điều khiển Blender và phân tích chất lượng map 3D mà không cần can thiệp thủ công của con người.

---

## 1. Vai trò của AI Coding Agent trong Hệ sinh thái TerraForge

Trong mô hình kiến trúc của TerraForge, AI Agent giữ vai trò là **Bộ não Thiết kế & Lập lịch (Architect & Orchestrator)**:

```
+-------------------------------------------------------------+
|                     AI CODING AGENT                         |
|  1. Nhận prompt người dùng (ví dụ: "Tạo map hòn đảo karst")|
|  2. Quyết định thông số địa hình (Peaks, Hydrology, Biomes)|
|  3. Kích hoạt TerraForge Middleware                        |
|  4. Phân tích kết quả thẩm định (QualityValidator Metrics)  |
|  5. Tự động sửa sai (Self-Correction Loop)                 |
+-------------------------------------------------------------+
                              |
                     [MapConfig JSON / CLI]
                              v
+-------------------------------------------------------------+
|                 TERRAFORGE ENGINE (PYTHON)                  |
|  - Xử lý toán học bề mặt, xói mòn hạt, thủy văn, sinh thái |
+-------------------------------------------------------------+
                              |
                      [Headless / Live MCP]
                              v
+-------------------------------------------------------------+
|                     BLENDER 4.x / 5.x                       |
|  - Dựng mesh, gán vật liệu PBR, Geometry Nodes, xuất GLB    |
+-------------------------------------------------------------+
```

---

## 2. Đặc tả Cấu hình Khai báo (`MapConfig` Schema)

Khi Agent sinh cấu hình cho TerraForge, cấu trúc JSON cần tuân thủ định dạng sau:

```json
{
  "name": "Ten_Map_Khong_Dau",
  "theme": "alpine_fjord | karst_cavern | volcanic_marsh | custom",
  "seed": 42,
  "diorama": {
    "width": 160.0,
    "depth": 160.0,
    "z_base": -16.0,
    "resolution": 256
  },
  "peaks": [
    {
      "name": "Ten_Dinh_Nui",
      "position": [0.0, 20.0],
      "height": 35.0,
      "radius": 22.0,
      "shape": "horn",
      "ridge_angle_deg": 0.0
    }
  ],
  "erosion": {
    "enabled": true,
    "droplet_count": 25000,
    "droplet_inertia": 0.05,
    "sediment_capacity_factor": 4.0,
    "erode_speed": 0.35,
    "deposit_speed": 0.35,
    "evaporate_speed": 0.015,
    "gravity": 4.0,
    "max_droplet_steps": 64,
    "thermal_iterations": 8,
    "talus_angle_deg": 38.0,
    "scree_talus_deposition": 0.4
  },
  "hydrology": {
    "lake_center": [-14.0, -10.0],
    "lake_radius": 24.0,
    "lake_level": 4.5,
    "lake_depth": 12.0,
    "sea_level": 0.0,
    "river_enabled": true,
    "river_carve_depth": 2.0,
    "river_meander_amplitude": 1.0
  },
  "cavern": {
    "enabled": true,
    "center": [14.0, 16.0, -4.5],
    "radius_x": 18.0,
    "radius_y": 14.0,
    "height": 7.0,
    "entrance_pos": [0.0, 12.0, 1.5],
    "entrance_radius": 4.5,
    "min_roof_clearance": 5.0,
    "stalactite_count": 16,
    "stalagmite_count": 14,
    "column_count": 4,
    "subterranean_pool": true,
    "pool_level": -7.2,
    "bioluminescent": true,
    "glow_color": [0.1, 0.9, 0.4]
  },
  "biomes": [
    {
      "name": "Rung_Thong_Cao",
      "prototype": "pine_tall",
      "min_altitude": 8.0,
      "max_altitude": 26.0,
      "min_slope_deg": 0.0,
      "max_slope_deg": 35.0,
      "min_water_dist": 3.0,
      "max_water_dist": 999.0,
      "density": 0.08,
      "poisson_dist_min": 3.0,
      "scale_min": 0.8,
      "scale_max": 1.3,
      "seed": 42
    }
  ],
  "shaders": {
    "triplanar": true,
    "rock_texture": "cliff_rock",
    "soil_texture": "forest_ground",
    "sand_texture": "wet_sand",
    "snow_altitude": 22.0,
    "water_absorption_color": [0.05, 0.45, 0.65],
    "water_roughness": 0.04,
    "water_contact_foam": true
  },
  "export": {
    "export_glb": true,
    "export_blend": true,
    "export_heightmap_json": true,
    "glb_path": "models/my_diorama.glb",
    "blend_path": "models/my_diorama.blend",
    "heightmap_path": "models/my_diorama_heightmap.json"
  }
}
```

---

## 3. Hướng dẫn Tích hợp Chi Tiết theo Từng Agent

### 3.1. Google Antigravity 2.0
Antigravity là agent có năng lực thực thi trực tiếp trên hệ thống qua công cụ `run_command` và kiểm tra file qua `view_file`.

#### Quy trình chuẩn của Antigravity Agent:
1. **Lắng nghe yêu cầu**: Người dùng yêu cầu địa hình mới hoặc thay đổi địa hình cũ.
2. **Đọc preset làm nền tảng**:
   Agent đọc một trong các file tại `terra_forge/schema/presets/` để làm mẫu tham số.
3. **Sinh file cấu hình tùy biến**:
   Agent tạo file tạm hoặc lưu trực tiếp vào thư mục dự án (ví dụ `custom_map.json`).
4. **Kích hoạt lệnh CLI**:
   ```bash
   python3 -m terra_forge.cli generate -p custom_map.json --render -o models/run_01
   ```
5. **Kiểm tra JSON đầu ra**:
   Agent phân tích stdout để tìm chuỗi `RESULT_JSON:`.
   - Nếu `validation.passed == true`: Tiếp tục xem các ảnh render tại `previews/`.
   - Nếu `validation.passed == false`: Đọc mảng `validation.errors` để biết chính xác lỗi (ví dụ: vi phạm độ dày vòm đá trần hang) và sửa lại `custom_map.json`.

---

### 3.2. Claude Desktop (Anthropic Claude Code / MCP)
Claude Desktop tích hợp qua **Model Context Protocol (MCP)** kết nối với Blender server.

#### Bước 1: Agent nạp thư viện `BlenderBridge`
Agent sinh mã Python sử dụng hàm hỗ trợ có sẵn:
```python
import json
from terra_forge.blender.bridge import BlenderBridge

bridge = BlenderBridge()
# config_dict được Claude sinh ra từ prompt
live_python_code = bridge.generate_live_blender_code(config_dict)
```

#### Bước 2: Claude gọi công cụ Blender MCP
Claude gửi đoạn code trên vào tool:
```json
{
  "tool": "execute_blender_code",
  "arguments": {
    "code": "<live_python_code>"
  }
}
```
*Hiệu ứng*: Viewport 3D của Blender sẽ lập tức xóa scene cũ, dựng lại khối diorama mới, gán shader PBR và rải cây cối trong vòng chưa đầy 1 giây.

#### Bước 3: Tinh chỉnh cục bộ (Zero-rebuild live tweaking)
Nếu người dùng chỉ muốn chỉnh nhỏ (ví dụ: tăng mật độ nấm phát quang hoặc thay đổi góc dốc tối đa của rừng thông), Claude sử dụng hàm:
```python
tweak_code = BlenderBridge.generate_modifier_tweak_code(
    modifier_name="Scatter_Rung_Thong_Cao",
    socket_name="Density",
    value=0.15
)
```
Gửi `tweak_code` qua `execute_blender_code` sẽ cập nhật ngay lập tức mà không cần tính toán lại mô phỏng xói mòn.

---

### 3.3. OpenAI Codex / ChatGPT / Cursor Agent
Đối với Codex hoặc môi trường chạy script tự động:

1. **Agent tạo script Python độc lập `generate_world.py`**:
   ```python
   from terra_forge.schema.map_config import MapConfig, PeakConfig
   from terra_forge.blender.bridge import BlenderBridge

   cfg = MapConfig(name="Codex_Island")
   cfg.peaks.append(PeakConfig(name="Central_Volcano", position=(0.0, 0.0), height=40.0, radius=30.0, shape="dome"))
   cfg.hydrology.lake_level = 3.0

   bridge = BlenderBridge()
   result = bridge.run_headless(cfg.model_dump(), render_preview=True)
   print("STATUS:", result["status"])
   ```
2. **Agent chạy script và thu thập file GLB**: File GLB xuất ra sẵn sàng được nhúng vào ứng dụng web Three.js hoặc Unity/Unreal Engine.

---

## 4. Hệ Thống Invariants & Quy Tắc Tự Động Sửa Lỗi (Self-Healing Rules)

Khi `QualityValidator` trả về cảnh báo hoặc lỗi trong `RESULT_JSON`, AI Agent nên áp dụng bảng quy tắc sau:

| Lỗi / Cảnh báo phát hiện | Nguyên nhân hình học | Hành động khắc phục tự động của Agent |
| :--- | :--- | :--- |
| `Cavern clearance violation: Rock thickness is X.Xm (< 5.0m)` | Đỉnh buồng hang động khoét quá cao, làm trần đá mỏng có nguy cơ thủng ra bề mặt đất. | Giảm `cavern.height`, hạ tọa độ `cavern.center[2]` xuống sâu hơn, hoặc tăng độ cao `peaks` ở khu vực phía trên hang. |
| `Lake containment warning: Freeboard berm is negative` | Mực nước hồ (`lake_level`) cao hơn thành bờ đất thấp nhất xung quanh hồ, gây ngập lụt. | Hạ `hydrology.lake_level` xuống thấp hơn 1.0m - 2.0m, hoặc tăng bán kính đào hồ `lake_radius`. |
| `Watertight violation: boundary_edges > 0` | Khối diorama bị rách mặt, không tạo thành thể tích kín đa tạp. | Giữ nguyên lưới $NxN$ đồng nhất trong `diorama.resolution` (mặc định 256). |
| `Zero scatter points generated for biome` | Bộ lọc sinh thái quá khắt khe khiến không có ô địa hình nào thỏa mãn đồng thời (ví dụ: vừa đòi hỏi độ cao $>30\text{m}$ vừa đòi hỏi cách nước $<1\text{m}$). | Nới rộng dải `min_altitude` - `max_altitude`, tăng `max_water_dist`, hoặc nới lỏng `max_slope_deg`. |

---

## 5. Mẫu System Prompt Gợi Ý Cho Agent

Khi triển khai AI Agent làm chuyên gia tạo thế giới 3D với TerraForge, hãy nhúng đoạn hướng dẫn sau vào System Prompt:

```text
Bạn là "TerraForge 3D World Architect" - chuyên gia thiết kế thế giới 3D và địa hình thủ tục.
Nhiệm vụ của bạn là lắng nghe ý tưởng thế giới của người dùng và chuyển đổi thành cấu hình MapConfig chuẩn của TerraForge.

Nguyên tắc bắt buộc:
1. Đảm bảo tính khoa học địa chất:
   - Núi dốc phải đi kèm xói mòn thủy lực (droplet_count từ 20,000 đến 35,000).
   - Vách núi đá phải có talus_angle_deg khoảng 35 - 42 độ để tạo triền đá vụn tự nhiên.
   - Sông phải bắt nguồn từ độ cao lớn và đổ dồn về hồ hoặc vịnh biển.
2. An toàn cấu trúc hang động:
   - Khoảng cách từ đỉnh hang đến mặt đất bên trên (clearance) phải luôn >= 5.0m.
3. Sinh thái hài hòa:
   - Cây thông (pine) ở sườn dốc và độ cao lớn.
   - Cây lá rộng (broadleaf) ở thung lũng gần nguồn nước.
   - Sậy (reeds) ở mép bờ nước (khoảng cách nước 0.1m - 3.5m).
   - Nấm phát quang (fungus_glow) phân bố trong hang ngầm hoặc khe nứt núi lửa.
4. Tự động kiểm tra: Luôn đọc kết quả `validation` sau khi chạy. Nếu có lỗi, tự động điều chỉnh tham số và chạy lại trước khi báo cáo cho người dùng.
```
