# TerraForge: Tài Liệu Kiến Trúc & Thuật Toán Chuyên Sâu (Technical Architecture)

> **Mục tiêu**: Phân tích chi tiết các thuật toán toán học, mô phỏng vật lý địa chất, xử lý hình học đa tạp (manifold geometry) và kỹ thuật shader được hiện thực hóa trong TerraForge.

---

## 1. Mô Hình Hóa Độ Cao Địa Hình (`Heightfield2D`)

Địa hình cơ sở được biểu diễn bằng một ma trận 2 chiều liên tục:
$$\mathbf{Z} \in \mathbb{R}^{N_y \times N_x}$$
Trong đó $N_x, N_y$ là độ phân giải lưới (thường là $256 \times 256$). Không gian thế giới thực được chuẩn hóa xung quanh gốc tọa độ $(0, 0)$:
$$x \in \left[-\frac{W}{2}, \frac{W}{2}\right], \quad y \in \left[-\frac{D}{2}, \frac{D}{2}\right]$$
Bước lưới rời rạc: $\Delta x = \frac{W}{N_x - 1}$, $\Delta y = \frac{D}{N_y - 1}$.

### 1.1. Các Dạng Hình Học Núi (Peak Morphology)
- **Dạng Đỉnh Nhọn (Horn Peak)**:
  Mô phỏng đỉnh chóp sắc nhọn kiểu Matterhorn được tạo thành từ băng hà gọt đẽo từ nhiều phía:
  $$d = \sqrt{(x - x_0)^2 + (y - y_0)^2}, \quad \tilde{d} = \text{clip}\left(\frac{d}{R}, 0, 1\right)$$
  $$Z_{\text{base}} = H \cdot (1 - \tilde{d})^{2.2}$$
  $$Z_{\text{horn}} = Z_{\text{base}} \cdot \left[1 + 0.25 \cdot \cos(3\theta)\right], \quad \theta = \text{atan2}(y - y_0, x - x_0)$$
  Hàm bậc 3 của góc $\theta$ tạo ra 3 sống núi phân nhánh đối xứng đặc trưng.

- **Dạng Sống Núi Lưỡi Dao (Arête Ridge)**:
  Kéo dài theo góc phương vị $\alpha$:
  $$x' = (x - x_0)\cos\alpha + (y - y_0)\sin\alpha, \quad y' = -(x - x_0)\sin\alpha + (y - y_0)\cos\alpha$$
  $$Z_{\text{ridge}} = H \cdot \left(1 - \frac{|x'|}{1.5R}\right)^{1.5} \cdot \left(1 - \frac{|y'|}{0.4R}\right)^{2.0}$$

- **Dạng Cao Nguyên Bờ Dốc (Plateau with Talus Rim)**:
  Sử dụng hàm sigmoid mượt mà:
  $$Z_{\text{plateau}} = H \cdot \frac{1}{1 + \exp\left[12 \cdot \left(\frac{d}{R} - 0.7\right)\right]}$$

### 1.2. Phép Tính Vi Phân Mặt Bề Mặt & Vector Pháp Tuyến
Để tính độ dốc và hướng pháp tuyến mà không phụ thuộc Blender, TerraForge sử dụng sai phân trung tâm (central differences):
$$\frac{\partial Z}{\partial x} \approx \frac{Z[y, x+1] - Z[y, x-1]}{2\Delta x}, \quad \frac{\partial Z}{\partial y} \approx \frac{Z[y+1, x] - Z[y-1, x]}{2\Delta y}$$
Vector pháp tuyến bề mặt chưa chuẩn hóa là $\mathbf{N}' = \left(-\frac{\partial Z}{\partial x}, -\frac{\partial Z}{\partial y}, 1\right)$. Sau khi chuẩn hóa $\|\mathbf{N}\| = 1$, góc nghiêng bề mặt so với mặt đất nằm ngang được xác định bởi:
$$\theta_{\text{slope}} = \arccos(N_z) \in [0^\circ, 90^\circ]$$

---

## 2. Mô Phỏng Xói Mòn Địa Chất (`ErosionSimulator`)

TerraForge kết hợp hai cơ chế xói mòn bổ trợ lẫn nhau:

### 2.1. Xói Mòn Thủy Lực Dạng Hạt (Particle-Based Hydraulic Erosion)
Mô phỏng hàng ngàn giọt nước ảo tương tác với địa hình theo thời gian:
1. **Khởi tạo giọt nước**: Vị trí ngẫu nhiên $\mathbf{p} = (p_x, p_y)$, vận tốc $v = 1$, lượng nước mang theo $w = 1$, trầm tích mang theo $s = 0$.
2. **Cập nhật hướng chuyển động có quán tính**:
   $$\mathbf{g} = \left(\frac{\partial Z}{\partial x}(\mathbf{p}), \frac{\partial Z}{\partial y}(\mathbf{p})\right)$$
   $$\mathbf{d}_{t+1} = \mathbf{d}_t \cdot I - \mathbf{g} \cdot (1 - I)$$
   Trong đó $I \in [0, 1]$ là quán tính của giọt nước (`droplet_inertia = 0.05`).
3. **Tính toán dung lượng mang trầm tích tối đa (Sediment Capacity)**:
   $$C = \max(-\Delta h, C_{\min}) \cdot v \cdot w \cdot k_c$$
4. **Quyết định Bồi tích (Deposition) hoặc Xói mòn (Erosion)**:
   - Nếu $\Delta h > 0$ (giọt nước đi ngược dốc) hoặc $s > C$:
     $$\Delta s = \min(\Delta h, s) \quad \text{hoặc} \quad (s - C) \cdot k_d$$
     Trầm tích được trả lại làm đầy 4 đỉnh lân cận theo trọng số song tuyến tính (bilinear weights).
   - Nếu $s < C$:
     $$\Delta s = \min\left((C - s) \cdot k_e, -\Delta h\right)$$
     Đất đá bị đào sâu xuống từ 4 đỉnh lân cận và nạp vào giọt nước.
5. **Cập nhật trạng thái**: Vận tốc tăng dần theo trọng lực $v_{t+1} = \sqrt{v_t^2 + \Delta h \cdot g}$, lượng nước bốc hơi dần $w_{t+1} = w_t \cdot (1 - k_{\text{evap}})$.

### 2.2. Xói Mòn Nhiệt & Cân Bằng Talus (Thermal Weathering)
Các vách đá dốc dựng đứng theo thời gian sẽ nứt vỡ vì chu kỳ đóng băng - tan băng (freeze-thaw).
Góc nghỉ tự nhiên (angle of repose) $\phi = \text{talus\_angle\_deg}$ (thường $36^\circ - 40^\circ$).
Độ dốc tối đa cho phép giữa 2 đỉnh kề cận:
$$\Delta Z_{\max} = \tan(\phi) \cdot \Delta x$$
Nếu chênh lệch độ cao thực tế $|\Delta Z| > \Delta Z_{\max}$, một phần đất đá dư thừa sẽ sạt lở xuống ô thấp hơn:
$$\text{Slip} = (|\Delta Z| - \Delta Z_{\max}) \cdot \frac{1}{4} \cdot k_{\text{talus}} \cdot \text{sign}(\Delta Z)$$
Quá trình này tự động làm mềm các vách đá nhân tạo và bồi tụ thành các triền dốc đá vụn (scree slopes) tự nhiên ở chân núi.

---

## 3. Động Học Thủy Văn (`HydrologyEngine`)

Hệ thống thủy văn được điều khiển bằng đường cong tham số đa đoạn liên tục $C^1$:
$$\mathbf{R}(t) = \big(R_x(t), R_y(t), R_z(t), R_w(t)\big), \quad t \in [0.0, 1.0]$$
- $t \in [0.00, 0.32)$: **Mountain Cascades** - độ dốc đứng, dòng chảy hẹp, ít uốn khúc.
- $t \in [0.32, 0.58)$: **Valley Meander** - uốn khúc mạnh theo hàm điều hòa:
  $$R_x(u) = x_{\text{start}} + \Delta x \cdot u - 8.0 \cdot A_{\text{meander}} \cdot \sin(2\pi u)$$
  $$R_y(u) = y_{\text{start}} + \Delta y \cdot u + 4.0 \cdot A_{\text{meander}} \cdot \sin(\pi u)$$
- $t \in [0.58, 0.74)$: **Central Lake Transit** - dòng sông chảy vào hồ trung tâm và cân bằng tại độ cao $Z = \text{lake\_level}$.
- $t \in [0.74, 1.00)$: **Outlet Gorge & Bay** - hẻm núi thoát nước chảy thẳng ra biển ở $Z = \text{sea\_level} = 0.0$.

### Chạm Khắc Lòng Sông Parabol (Parabolic Channel Carving) & Dải Ruy Băng Nước Liên Tục
Tại mỗi điểm mẫu dọc theo spline sông với tâm $(c_x, c_y, c_z)$ và bề rộng lòng sông $w$, bán kính ảnh hưởng $R_{\text{inf}} = 1.5w$. Với mọi điểm lưới trong bán kính ảnh hưởng:
$$\tilde{d} = \frac{\sqrt{(x - c_x)^2 + (y - c_y)^2}}{R_{\text{inf}}}$$
Độ sâu rãnh:
$$Z_{\text{bed}} = c_z - (1 - \tilde{d}^2) \cdot D_{\text{carve}}$$
Cập nhật cao độ: $Z[y, x] = \min\big(Z[y, x], Z_{\text{bed}}\big)$.

Dải nước 3D (`Water_River_Meander`) được sinh liên tục thành 2 phân đoạn liên kết hoàn chỉnh:
1. **Phân đoạn Thượng nguồn** ($t \in [0.00, 0.58]$): Thác núi $\to$ Khúc quanh thung lũng đổ trực tiếp vào hồ trung tâm.
2. **Phân đoạn Hạ nguồn** ($t \in [0.74, 1.00]$): Cửa tràn hẻm núi thoát nước từ hồ trung tâm ra vịnh biển.

### Bờ Chắn Giữ Nước Lòng Hồ (Lake Basin Bathymetry & Retaining Berm)
Để đảm bảo mặt nước hồ nhân tạo ở cao độ $L = \text{lake\_level}$ không bao giờ bị "lơ lửng trên không trung" và luôn thỏa mãn tiêu chuẩn kiểm định $B_{\text{freeboard}} \ge 0.35\text{m}$, TerraForge áp dụng mô hình phân vùng 3 cung bán kính với bán kính hồ $R$:
1. **Vùng đáy hồ sâu** ($\tilde{d} = r / 0.95R < 1$):
   $$Z_{\text{basin}} = (L - D) + (D + B) \cdot (3\tilde{d}^2 - 2\tilde{d}^3)$$
2. **Vành đỉnh bờ chắn (Berm Crest)** ($r \in [0.95R, 1.15R]$):
   $$Z_{\text{crest}} = L + B \quad (B = \text{berm\_height} = 1.2\text{m})$$
3. **Sườn dốc thoải ngoài bờ chắn (Outer Falloff)** ($r \in [1.15R, 1.50R]$):
   $$u = \frac{r - 1.15R}{0.35R}, \quad Z_{\text{outer}} = (L + B) \cdot \big[1 - (3u^2 - 2u^3)\big]$$

---


## 4. Rải Thực Vật Gia Tốc Ô Lưới (`BiomeEngine`)

Thay vì kiểm tra toàn bộ khoảng cách giữa tất cả các cặp điểm ($O(N^2)$), TerraForge sử dụng thuật toán **Fast Spatial-Grid Accelerated Poisson Disk Sampling**:
1. Kích thước ô lưới kiểm tra:
   $$r_{\text{cell}} = \frac{r_{\min}}{\sqrt{2}}$$
   Với kích thước này, mỗi ô lưới trong mảng 2D chỉ có thể chứa tối đa 1 điểm duy nhất.
2. Khi xét một điểm ứng viên $(w_x, w_y)$, thuật toán chỉ cần tra cứu trong phạm vi $5 \times 5$ ô lưới lân cận.
3. Độ phức tạp giảm xuống $O(N)$, cho phép rải hàng nghìn cá thể cây cối, đá tảng chỉ trong chưa đầy $0.01$ giây.

---

## 5. Cấu Trúc Khối Diorama Kín Nước (`DioramaSlabBuilder`)

Để đảm bảo tệp GLB xuất ra là một hình đa diện đóng kín hoàn hảo (closed manifold watertight solid):
- **Mặt đỉnh**: $(N_x - 1) \times (N_y - 1)$ quads với chiều quay ngược chiều kim đồng hồ (CCW).
- **Mặt đáy**: $(N_x - 1) \times (N_y - 1)$ quads tại $Z = z_{\text{base}}$ với chiều quay thuận chiều kim đồng hồ (CW) để vector pháp tuyến hướng xuống dưới.
- **4 Vách cắt**:
  - *Vách Nam ($y = 0$)*: $(N_x - 1)$ quads hướng về $-Y$.
  - *Vách Bắc ($y = N_y - 1$)*: $(N_x - 1)$ quads hướng về $+Y$.
  - *Vách Tây ($x = 0$)*: $(N_y - 1)$ quads hướng về $-X$.
  - *Vách Đông ($x = N_x - 1$)*: $(N_y - 1)$ quads hướng về $+X$.

Tổng số mặt quads:
$$F_{\text{total}} = 2(N_x - 1)(N_y - 1) + 2(N_x - 1) + 2(N_y - 1)$$
Tổng số đỉnh:
$$V_{\text{total}} = 2 \cdot N_x \cdot N_y$$
Công thức này đảm bảo Euler characteristic $\chi = V - E + F = 2$ chuẩn mực cho khối hình học dạng cầu đồng phôi (homeomorphic to sphere), bảo đảm không có lỗ thủng cạnh biên (`boundary_edges == 0`).

---

## 6. Lọc Độ Dốc Hình Học Trong Geometry Nodes (`terra_forge.blender.geonodes`)

Nhằm ngăn chặn thực vật (cây thông, cây lá rộng) mọc ngang trên vách núi dựng đứng hoặc 4 mặt cắt thẳng đứng của khối diorama slab ($N_z = 0$):
1. Tham số `Max_Slope_Deg` ($\theta_{\max}$) được đưa vào ShaderNodeMath `RADIANS` để đổi sang radian $\theta_{\text{rad}}$.
2. Giá trị ngưỡng cosin được tính toán bằng ShaderNodeMath `COSINE`:
   $$N_{z,\min} = \cos(\theta_{\text{rad}})$$
3. Vector pháp tuyến bề mặt của địa hình $N_z$ được so sánh bằng FunctionNodeCompare:
   $$\text{Slope\_Valid} = (N_z \ge N_{z,\min})$$
4. Điều kiện độ dốc được kết hợp cùng khoảng độ cao bằng cổng logic FunctionNodeBooleanMath `AND`:
   $$\text{Selection} = (Z \ge Z_{\min}) \land (Z \le Z_{\max}) \land (N_z \ge N_{z,\min})$$
Cơ chế này loại bỏ hoàn toàn hiện tượng cây mọc xuyên vách cắt của diorama.

---

## 7. Phân Loại Khả Năng Di Chuyển Không Gian (Spatial Passability Navmesh)

Tệp `*_heightmap.json` xuất ra ma trận `passability_matrix` chuẩn hóa phục vụ pathfinding (A*, NavMesh) của các AI agent:
- `0`: **Nước sâu / Không thể vượt qua** — Giới hạn trong lòng hồ trung tâm ($(X - x_L)^2 + (Y - y_L)^2 \le 1.05 R^2$ và $Z \le L + 0.1$), hành lang dòng sông thủy văn ($d_{\text{river}} \le 0.75 w$), vịnh biển duyên hải và các vùng trũng dưới mực nước biển ($Z < S - 0.5$).
- `1`: **Đất bằng phẳng / Khả thi cho nhân vật di chuyển** ($Z > 0.0\text{m}$, độ dốc $\le 40^\circ$).
- `2`: **Vách đá dốc hiểm trở** (Góc nghiêng bề mặt $\theta_{\text{slope}} > 40^\circ$).
- `3`: **Đỉnh núi cao / Băng tuyết vĩnh cửu** ($Z > 22.0\text{m}$).

