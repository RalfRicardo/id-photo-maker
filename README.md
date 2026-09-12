# ID Photo Studio Master v1.0 (Export Multi-Layer PSD)

> **Tác giả:** Vinh | **Phiên bản:** 1.0  
> **Phần mềm xử lý ảnh thẻ chuyên nghiệp cho Studio & Lab ảnh (Tối ưu hóa Windows 10/11 & Đa nền tảng)**  
> Tự động nhận diện khuôn mặt, xoay ngang trục mắt, cắt cúp chuẩn theo quy chuẩn ID/Visa quốc tế, bóc tách nền thông minh bằng AI, và xuất file Adobe Photoshop (.psd) **giữ nguyên từng lớp riêng biệt với Layer Mask không phá hủy (Non-destructive)**.

---

## 🎯 Điểm nổi bật & Tính năng cốt lõi

1. **Chuẩn kích thước đa quốc gia (Configurable Presets):**
   - `3x4 cm` (Căn cước công dân, Bằng lái xe Việt Nam): Tỉ lệ đầu 70–75%, mắt 58% từ đáy.
   - `4x6 cm` (Hộ chiếu Việt Nam & Quốc tế): Tỉ lệ đầu 70–80%, mắt 58% từ đáy.
   - `2x2 inch` (5x5 cm - US Visa / Passport): Tỉ lệ đầu 50–69%, mắt 56–69% từ đáy.
   - `3.5x4.5 cm` (Schengen Visa, Visa Hàn Quốc): Tỉ lệ đầu 70–80%.

2. **Quy trình xử lý tự động (Processing Pipeline):**
   - **Tự động cân bằng trục mắt (Auto-Roll Leveling):** Tính toán góc nghiêng giữa 2 mắt và xoay ảnh về góc $0^\circ$ hoàn hảo.
   - **Căn cúp thông minh (Smart Crop):** Xác định đỉnh đầu (crown), đường mắt (eyes level) và đáy cằm (chin) theo chuẩn sinh trắc học để căn cúp chuẩn xác từng pixel.
   - **Tách nền AI tối ưu CPU & Khử lem màu (CPU Matting Pipeline):**
     - **Mô hình ONNX RMBG-1.4 Quantized (CPU-Only):** Sử dụng `onnxruntime` với `CPUExecutionProvider` (không yêu cầu CUDA/GPU rời), chạy mượt mà trên PC văn phòng phổ thông (Core i3/i5, 8GB RAM). Mô hình chạy ngầm trên `QThread` độc lập, chống giật/đơ giao diện hoàn toàn.
     - **Lọc viền tóc siêu mịn (Fast Guided Filter):** Tinh chỉnh mặt nạ thô bằng bộ lọc dẫn hướng tốc độ cao (radius = 4–8, eps = 1e-4) lấy ảnh xám sắc nét gốc làm ảnh dẫn đường, giữ trọn từng sợi tóc mai, râu mép và cấu trúc viền áo.
     - **Khử lem màu nền cũ (Color Decontamination / Spill Suppression):** Tự động bóc tách và khử sạch ánh màu nền phòng chụp (quầng xanh lam, xanh lá) vướng trong các sợi tóc bán trong suốt ($10 < \alpha < 240$), đảm bảo khi đặt lên nền trắng hoặc nền khác tóc không bị viền màu lem luốc.
     - **Mặt nạ 8-bit mềm mại (8-bit Soft Grayscale Mask):** Bảo toàn đầy đủ sắc độ xám mượt mà trực tiếp vào Layer Mask của file PSD, không bị răng cưa hay nhị phân hóa.
   - **Chỉnh sửa khuôn mặt & Xóa mụn (Facial Retouching):**
     - **Làm mịn da & Xóa mụn (Skin Smoothing):** Sử dụng bộ lọc bảo toàn cạnh (Edge-Preserving Bilateral Filter) kết hợp bảo toàn vân da vi mô, giúp xóa mụn, làm mịn lỗ chân lông mà không bị bết dính hay giả tạo ("waxy effect").
     - **Làm nét chi tiết (Smart Sharpening):** Tăng độ sắc nét thông minh theo kênh sáng (LAB Luminance Unsharp Masking), làm rõ từng sợi lông mi, ánh mắt, viền tóc và nếp vải áo mà không gây viền nhiễu màu.
   - **Công cụ cọ tẩy & Khôi phục nét thủ công (Interactive Eraser / Restore Brush):**
     - **🧹 Chế độ tẩy (Eraser):** Nút công cụ riêng cho phép rê chuột xóa các sợi tóc bay thừa hoặc mép áo dính vào màu nền cũ.
     - **🖌️ Chế độ khôi phục (Restore):** Quét chuột để lấy lại các chi tiết tóc mai hoặc nếp áo lỡ bị AI bóc nhầm.
     - **↩ Hoàn tác & ↪ Làm lại (Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z):** Hỗ trợ đầy đủ phím tắt và nút bấm Undo/Redo cho từng nét cọ vẽ.
     - **Đồng bộ hóa nét tẩy theo chuyển động ảnh (Lockstep Motion):** Nét cọ được gắn chặt vào không gian tọa độ chân dung gốc. Khi phóng to/thu nhỏ (Zoom), dịch chuyển tọa độ (X/Y Shift) hay xoay góc nghiêng (Tilt), phần bị tẩy xóa **tự động di chuyển và xoay bám dính theo sợi tóc**, không bao giờ bị lệch vị trí.
     - **Tùy chỉnh cỡ cọ & Khôi phục gốc:** Thanh trượt kích thước cọ trực quan kèm vòng tròn dẫn hướng và nút "Khôi phục mặt nạ gốc" bất cứ lúc nào.
     - Mọi nét cọ vẽ thủ công được **đồng bộ trực tiếp vào Layer Mask** của file `.psd` khi xuất, không làm hỏng pixel ảnh gốc.
   - **Huy hiệu kiểm định sinh trắc học (Biometric Compliance Badge):** Báo ngay màu xanh `✓ ĐẠT CHUẨN` hoặc cảnh báo màu cam nếu tỉ lệ đầu/mắt chưa đạt.

3. **Cấu trúc File Adobe Photoshop (.psd) đa lớp chuẩn Studio:**
   - **Lớp 1 (Đáy) — "Nền (Background)":** Lớp màu nền đơn sắc (Xanh chuẩn `#0090FF`, Tr trắng `#FFFFFF`, Xanh đậm `#1E60AA` hoặc màu tùy chọn).
   - **Lớp 2 — "Chủ thể (Portrait)":** Giữ nguyên dữ liệu RGB gốc của ảnh đã cắt, gắn kèm **Layer Mask** (mặt nạ mờ) có thể chỉnh sửa trực tiếp bằng cọ Brush trong Photoshop.
   - **Lớp 3 (Đỉnh, Ẩn) — "Khung căn chuẩn (Guide lines)":** Các đường chỉ dẫn kỹ thuật đánh dấu vị trí Đỉnh đầu, Đường mắt, Cằm và Trục đứng để studio đối chiếu.
   - **Metadata chuẩn 300 DPI:** Photoshop tự động nhận diện đúng kích thước in ấn thực tế theo milimet/inch.

4. **Xếp bảng in 10x15 cm (Print Sheet Generator):**
   - Tự động dàn trang in khổ giấy ảnh tiêu chuẩn `10x15 cm @ 300 DPI`.
   - Hỗ trợ các mẫu bố cục:
     - Combo Studio: 4 ảnh 3x4 + 2 ảnh 4x6.
     - 8 ảnh 3x4 cm.
     - 3 ảnh 4x6 cm.
     - 2 ảnh 2x2 inch (5x5 cm).
   - Có sẵn **đường căn cắt (Cut Guidelines)** và dấu góc chữ L giúp thợ cắt ảnh nhanh chóng và chính xác.

5. **Chế độ Studio Watch Folder:**
   - Tự động theo dõi thư mục máy ảnh chụp trực tiếp (Tethered shooting). Khi thợ bấm máy lưu file vào thư mục, phần mềm tự động nạp và căn chỉnh tức thì.

---

## 🏗️ Cấu trúc thư mục dự án

```text
id-photo-maker/
├── app/
│   ├── core/
│   │   ├── face_aligner.py     # Nhận diện MediaPipe 478 landmarks, cân bằng trục mắt, smart-crop
│   │   ├── matting.py          # Bóc tách nền AI bằng rembg, làm mịn viền tóc (feathering)
│   │   ├── psd_builder.py      # Xây dựng file PSD đa lớp, Layer Mask, UTF-8 unicode name, 300 DPI
│   │   └── retouch.py          # Làm mịn da, xóa mụn, làm nét LAB USM, cọ tẩy/khôi phục viền tóc
│   ├── presets/
│   │   └── photo_specs.json    # Thông số tiêu chuẩn ảnh thẻ và bảng in khổ 10x15cm
│   ├── ui/
│   │   ├── components.py       # Giao diện Dark UI: Bộ chọn preset, bảng màu, thanh trượt vi mô, canvas
│   │   └── main_window.py      # Cửa sổ chính, luồng xử lý nền QThread, Watch Folder, nút xuất PSD
│   └── main.py                 # Điểm khởi động ứng dụng (hỗ trợ cả giao diện GUI và lệnh CLI)
├── tests/
│   └── test_psd_layers.py      # Bộ kiểm thử tự động (Assert 3 lớp, Layer Mask, DPI, độ chính xác)
└── requirements.txt            # Danh mục thư viện phụ thuộc
```

---

## 🚀 Hướng dẫn cài đặt & Khởi chạy

### 1. Cài đặt trên Windows 10/11 (Khuyến nghị cho Studio)

Yêu cầu: Đã cài đặt [Python 3.10+](https://www.python.org/downloads/) (Nhớ tích chọn `"Add Python to PATH"` khi cài đặt).

Mở **PowerShell** hoặc **Command Prompt** tại thư mục dự án:
```powershell
# 1. Tạo môi trường ảo
python -m venv .venv

# 2. Kích hoạt môi trường ảo
.venv\Scripts\activate

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 4. Khởi chạy giao diện Desktop App
python app/main.py
```

### 2. Cài đặt trên Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/main.py
```

---

## 💻 Hướng dẫn sử dụng

### 1. Chế độ Giao diện đồ họa (GUI)
1. **Mở ảnh:** Kéo thả trực tiếp file ảnh chân dung vào khung làm việc hoặc nhấn nút **"📂 Mở ảnh..."**.
2. **Chọn tiêu chuẩn:** Chọn preset tương ứng (ví dụ: `3x4 cm Việt Nam` hoặc `2x2 inch US Visa`).
3. **Chọn màu nền:** Bấm chọn màu nền nhanh (Xanh thẻ, Trắng, Xanh đậm) hoặc chọn mã màu HEX tùy ý.
4. **Cân chỉnh vi mô (nếu cần):** Sử dụng các thanh trượt bên trái để phóng to/thu nhỏ tỉ lệ đầu, dịch chuyển vị trí hoặc tinh chỉnh độ mịn viền tóc.
5. **Xuất file:**
   - Bấm **"💾 Xuất file PSD nhiều lớp"** để lưu file `.psd` mở được trên Photoshop với đầy đủ Layer Mask.
   - Bấm **"🖨️ Xuất bảng in 10x15 cm"** để tạo ngay trang in sẵn sàng cho máy in nhiệt/lab.

### 2. Chế độ Dòng lệnh tự động (Headless CLI Batch Processing)
Dành cho tự động hóa hàng loạt hoặc tích hợp vào hệ thống máy chủ lab ảnh:

```bash
# Xử lý 1 ảnh ra file PSD 3x4 nền xanh
python app/main.py --input portrait.jpg --preset 3x4_vn --bg "#0090FF" --output output_3x4.psd --headless

# Xử lý ảnh US Visa nền trắng và tạo kèm bảng in 10x15 cm
python app/main.py --input portrait.jpg --preset 2x2_us_visa --bg "#FFFFFF" --output us_visa.psd --sheet sheet_10x15.psd --headless
```

---

## 🧪 Chạy bộ kiểm thử tự động (Unit Tests)

Bộ kiểm thử tự động xác minh toàn diện tính toàn vẹn của file PSD đầu ra (đúng 3 lớp, tên Unicode UTF-8, Layer Mask chuẩn, kích thước 300 DPI):

```bash
pytest -v tests/test_psd_layers.py
```
Kết quả mong đợi:
```text
tests/test_psd_layers.py::test_psd_layers_and_alpha_mask PASSED
tests/test_psd_layers.py::test_face_aligner_roll_correction PASSED
tests/test_psd_layers.py::test_matting_composite PASSED
tests/test_psd_layers.py::test_print_sheet_layout PASSED
tests/test_psd_layers.py::test_face_retoucher_smooth_and_sharpen PASSED
tests/test_psd_layers.py::test_face_retoucher_eraser_brush PASSED
tests/test_psd_layers.py::test_psd_export_with_retouch_and_eraser_mask PASSED
tests/test_psd_layers.py::test_matting_engine_cpu_execution PASSED
tests/test_psd_layers.py::test_fast_guided_filter_edge_preservation PASSED
tests/test_psd_layers.py::test_color_decontamination_spill_suppression PASSED
tests/test_psd_layers.py::test_psd_export_preserves_soft_grayscale_mask PASSED
tests/test_psd_layers.py::test_brush_undo_redo_and_transform_synchronization PASSED
tests/test_psd_layers.py::test_app_about_dialog_and_metadata PASSED
============================== 13 passed in 4.88s ==============================
```
