# Original User Request

## 2026-09-04T10:47:29Z

Use a very large team of agents.

Nghiên cứu các giải pháp mã nguồn mở hàng đầu, các thuật toán mô phỏng địa chất/đồ họa hiện đại và các kỹ thuật tiên tiến nhất để nâng cấp toàn diện engine TerraForge, hướng tới mục tiêu duy nhất: tạo ra các bản đồ thế giới 3D tự nhiên siêu chân thực, sống động, chi tiết và hoàn chỉnh các môi trường tự nhiên cho Blender và AI Coding Agents.

Working directory: /Users/duongnad/Documents/tool/terra_forge
Integrity mode: benchmark

## Requirements

### R1. Mô phỏng Hình học & Vật lý Địa chất Nâng cao (Advanced Geological & Terrain Physics)
Nghiên cứu và triển khai các thuật toán sinh địa hình tự nhiên có cấu trúc địa chất thực tế:
- Nâng cấp mô phỏng xói mòn thủy lực (hydraulic droplet & sediment transport) tạo các đường rãnh xói mòn tự nhiên (rills and gullies), dòng chảy phân nhánh và bãi bồi trầm tích chân núi.
- Nâng cấp sạt lở nhiệt cân bằng góc dốc (thermal talus relaxation), tạo các triền dốc vụn đá chân vách (scree slopes).
- Bổ sung cấu trúc nếp gấp địa tầng (strata folding), vách đá dựng đứng kiến tạo và địa hình ven bờ (coastal cliffs, beaches, riverbanks).

### R2. Vật liệu Bề mặt Chân thực & Thủy văn Tự nhiên (Photorealistic Shading, Texturing & Hydrology)
Phát triển hệ thống vật liệu PBR và dòng chảy tự nhiên theo quy luật địa chất:
- Hệ thống Triplanar PBR shader tự động hòa trộn nhiều lớp vật liệu (multi-splatting) không kéo giãn UV: đá vách dựng đứng, đất mùn hữu cơ, sỏi cát lắng đọng, rêu phong ẩm ướt, và tuyết đọng đỉnh núi theo độ dốc, độ cao và bản đồ trầm tích xói mòn.
- Hệ thống nước (biển, sông, hồ) động học với khúc xạ/phản xạ trong suốt, gradient độ sâu, bọt nước ven bờ (shoreline foam) và lưu lượng dòng chảy suối/thác nước.

### R3. Hệ sinh thái Quần xã & Thảm thực vật Đa tầng (Multi-tier Natural Biomes & Ecological Vegetation)
Xây dựng hệ sinh thái tự nhiên phong phú với phân tầng sinh học chân thực:
- Mô phỏng phân bố sinh thái theo vi khí hậu: nhiệt độ theo độ cao, độ ẩm theo khoảng cách nguồn nước và hướng đón ánh nắng mặt trời (slope aspect / solar insolation).
- Phân tầng thảm thực vật đa lớp: tầng cây gỗ lớn (canopy trees), tầng cây bụi (understory shrubs), tầng thảm cỏ dại/rêu ẩm, và các chi tiết tự nhiên mặt đất (đá tảng rải rác, thân cây đổ).
- Rải đối tượng tự nhiên theo cụm sinh học (Poisson-Disk clustering) và tích hợp cấu hình Geometry Nodes linh hoạt, tối ưu hiệu năng hiển thị.

### R4. Môi trường Ánh sáng Khí quyển & Kiểm định Chất lượng Tự động (Atmospheric Environment & Quality Assurance)
Cung cấp giải pháp ánh sáng hoàn thiện và cơ chế xác thực kết quả khách quan:
- Thiết lập hệ thống chiếu sáng tự nhiên (mặt trời theo góc giờ trong ngày, sương mù khí quyển atmospheric haze/volumetrics) tạo cảm giác chiều sâu không gian (depth cueing).
- Tích hợp kiểm định hình học tự động: đảm bảo lưới 3D kín nước (watertight manifold diorama slab), không có mặt lỗi (non-manifold edges = 0), bờ hồ giữ nước vững chắc.
- Tự động hóa kết xuất các góc nhìn chuẩn (isometric tổng thể, góc ngang mắt người, cận cảnh địa hình) phục vụ đánh giá trực quan.

## Acceptance Criteria

### Tính toàn vẹn Mã nguồn & Kiểm thử Tự động (Source Code Integrity & Automated Tests)
- [ ] Toàn bộ bộ kiểm thử tự động (`pytest`) trong dự án vượt qua 100% (không có lỗi hồi quy).
- [ ] Các mô-đun mới được tích hợp liền mạch vào kiến trúc của TerraForge (`terra_forge.core`, `terra_forge.blender`, `terra_forge.schema`).

### Khả năng Sinh Bản đồ & Độ phong phú Môi trường (World Generation & Environment Presets)
- [ ] Hệ thống hỗ trợ sinh hoàn chỉnh ít nhất 4 preset môi trường tự nhiên đặc trưng với độ chi tiết cao:
  - Vùng núi cao tuyết phủ & hồ băng (Alpine Peak & Glacial Lake)
  - Thung lũng rừng ôn đới & sông uốn khúc (Temperate Forest River Basin)
  - Vách đá bờ biển & đảo nhiệt đới (Tropical Island & Coastal Karst Cliffs)
  - Vùng cao nguyên đá bán khô hạn & hẻm vực (Canyon & Semi-arid Plateau)

### Chất lượng Địa hình & Độ kín Hình học (Geological Quality & Mesh Topology)
- [ ] Mesh địa hình sinh ra đạt chuẩn 100% watertight manifold diorama block, được xác nhận bởi `quality_validator` với 0 non-manifold edges.
- [ ] Bản đồ độ cao thể hiện rõ các đặc trưng xói mòn dòng chảy tự nhiên (rills, sediment fans) và triền đá vụn talus chân núi.

### Hiển thị Trực quan & Đồ họa Blender (Visual Rendering & Asset Exports)
- [ ] Pipeline kết xuất Blender headless (`-b -P`) chạy thông suốt, tự động xuất file `.blend` và các ảnh render chất lượng cao ở nhiều góc nhìn.
- [ ] Cây cối, thực vật rải tự nhiên theo đúng đặc tính vi khí hậu, không bị mọc xuyên vách đá dựng đứng hoặc ngập dưới lòng nước sâu.

## Follow-up — 2026-09-04T10:56:57Z

[USER REQUIREMENT UPDATE FOR R3 & ECOSYSTEMS]
Người dùng đã bổ sung các yêu cầu cụ thể và chi tiết cho hệ sinh thái và môi trường tự nhiên:
1. Chi tiết thực vật mặt đất: Đầy đủ các loại hoa dại (wildflowers), cụm cỏ rậm rạp, thảm cỏ, dương xỉ, nấm rừng và rêu phong bám đá.
2. Cây gãy đổ tự nhiên: Cây có thể bị gãy, bật gốc, thân cây đổ mục (fallen logs, snapped trunks, rotting wood, stumps do bão gió/tuổi thọ).
3. Tác động của sinh vật lên môi trường (Biological/Creature Impacts): Lối mòn thú đi qua lại (game trails / animal paths), vùng cỏ bị giẫm đạp, các mảng đất xới xáo tự nhiên.
4. Hệ sinh thái dưới nước (Underwater Ecosystems): Thảm thực vật thủy sinh (rong biển, cỏ nước, lau sậy), sỏi cuội lòng sông/hồ, rạn san hô hoặc đá ngầm dưới nước.

Hãy cập nhật ngay các yêu cầu này vào Milestone M3 (Ecosystem & Biomes) và bộ kiểm thử E2E Test Suite của dự án.
