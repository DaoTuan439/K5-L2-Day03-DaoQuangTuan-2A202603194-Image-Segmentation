# Báo cáo thực hành Day 3 — điền trực tiếp trong fork của bạn

**Cách dùng:** Thay mọi dấu `…` bằng bài làm thật của bạn trước khi nộp link fork trên [LMS](https://aithucchien.vinuni.edu.vn). Giữ nguyên bốn mục và bảng để coach đọc nhanh. Viết ngắn, cụ thể theo ảnh/vùng; không cần thuật ngữ chuyên sâu. Ví dụ trong [hướng dẫn mẫu](../../reports/REPORT_TEMPLATE.md) chỉ giúp hiểu cách điền, không phải câu trả lời để chép lại.

- Mã học viên theo lớp: 2A202603194
- Ngày / CVAT local: 11/11/2026 / CVAT local
- Công cụ đã dùng: CVAT Polygon/Brush (toàn bộ Easy, object Medium đầu tiên, các chỗ sửa); nhãn nháp AI từ `tools/ai-prelabel` cho phần còn lại của Medium và Hard: `facebook/mask2former-swin-large-coco-panoptic` + SAM 2.1 (`facebook/sam2.1-hiera-large`) tinh chỉnh biên, chạy trên Colab GPU, ngưỡng tin cậy 0.5 (theo `prelabel_log.json`: Medium 41 giây, 58 object nháp, SAM 2.1 tinh chỉnh 49 mask; Hard 3,2 giây, 42 segment nháp). Nhãn nháp Medium/Hard được kiểm và sửa trong CVAT trước khi export. Checkpoint: `cp3_thin` và `cp4_curb` vẽ tay hoàn toàn; `cp1_holes`, `cp2_slice`, `cp5_occlusion`, `cp6_coverage` dùng nhãn nháp từ cùng notebook nhưng lần chạy này trên CPU nên **không có SAM** (`USE_SAM_REFINE = False`; cp1–cp5 dùng Mask2Former COCO panoptic, cp6 dùng `facebook/mask2former-swin-large-cityscapes-semantic`).

Mã học viên là mã lớp cấp; không cần ghi họ tên trong report nếu kênh [LMS](https://aithucchien.vinuni.edu.vn) đã nhận diện bạn. Chỉ ghi công cụ thật sự đã dùng; không có SAM vẫn làm bài bình thường.

## 1. Bài đã nộp

Ghi tên ZIP đúng như file trong `submissions/` và số ảnh đã vẽ, Save. Chưa làm hoặc export lỗi thì ghi `chưa có`, không tạo ZIP rỗng. Cột điểm là điểm tối đa của task, **không phải điểm tự chấm**.

| Task            | File ZIP đúng tên     | Hoàn thành mấy ảnh | Điểm tối đa (coach chấm sau) |
| --------------- | --------------------- | -----------------: | ---------------------------: |
| easy_semantic   | `easy_semantic.zip`   |              3 / 3 |                           20 |
| medium_instance | `medium_instance.zip` |              3 / 3 |                           32 |
| hard_panoptic   | `hard_panoptic.zip`   |              2 / 2 |                           30 |
| cp1_holes       | `cp1_holes.zip`       |              1 / 1 |                            3 |
| cp2_slice       | `cp2_slice.zip`       |              1 / 1 |                            3 |
| cp5_occlusion   | `cp5_occlusion.zip`   |              1 / 1 |                            3 |
| cp3_thin        | `cp3_thin.zip`        |              1 / 1 |                            3 |
| cp4_curb        | `cp4_curb.zip`        |              1 / 1 |                            3 |
| cp6_coverage    | `cp6_coverage.zip`    |              1 / 1 |                            3 |
| **Tổng tối đa** |                       |                    |                      **100** |

Nếu export lỗi, ghi task, dữ liệu đã Save đến đâu và lỗi đã báo coach.

Ghi chú checkpoint:

- `cp1_holes` (xe mô tô Honda): mask xe giữ cả kính chắn gió và phần trong vành bánh, không khoét lỗ; chân chống đỏ không tính vào xe. Sửa nhãn nháp: xóa mask `bus` ở mái lều đen (không phải xe), đổi xe van "Kawasaki" từ `bus` sang `car`. Giữ xe nhà di động màu kem bên trái là `bus` (6 object).
- `cp2_slice` (bãi đỗ xe): giữ nhãn nháp, 12 object; xe trắng và xe xanh mở cửa đỗ sát nhau ở góc trái là hai instance `car` riêng.
- `cp5_occlusion` (Athens): giữ nhãn nháp, 21 object; xe máy bị người lái che vẫn là một mask, người lái tách riêng.
- `cp3_thin`: vẽ tay bằng Polygon ở zoom lớn; pole ~1,8%, traffic sign ~4,4% ảnh (biển xanh lớn, biển "PASSENGER CARS ONLY", các biển nhỏ), còn lại sky và road.
- `cp4_curb`: vẽ tay; road dừng ở mép bó vỉa, vỉa hè bê tông bên phải và mảng bê tông ở góc đường là `sidewalk`; dải đất có túi rác để trống.
- `cp6_coverage`: giữ nhãn nháp semantic (road, sidewalk, building, sky, car, person), export Segmentation mask 1.1. Xe buýt và xe tải "CAPITAL" không thuộc 7 lớp nên để trống.

## 2. Một quyết định trước khi dùng gợi ý

Chọn object đầu tiên bạn tự vẽ ở `medium_instance`, trước khi xem bất kỳ đề xuất tự động nào cho object đó. Ghi ảnh/vị trí đủ để tìm lại; “quy tắc biên” là lý do bạn chọn hoặc dừng mask ở ranh đó.

- Ảnh, vị trí và object Medium đầu tiên tự vẽ: `000000181542.jpg`, người phụ nữ mặc áo dài trắng đi bộ ở giữa ảnh, trên vạch qua đường, giữa hai xe máy (bbox khoảng x 187–305, y 131–461). Vẽ tay bằng Polygon, export riêng thành `medium_instance_manual.zip` trước khi chạy nhãn nháp AI.
- Class và quy tắc tôi dùng để chọn biên: `person`, vẽ một Polygon theo phần nhìn thấy của người. Tính vào mask: tóc, mặt, cổ, hai tay và bàn tay, dây túi trên vai, toàn bộ áo dài kể cả hai vạt áo buông xuống, quần, hai bàn chân và dép — vì đó là người và đồ đang mặc/mang, di chuyển cùng người. Không tính: bóng trên mặt đường, người lái và xe máy phía sau lộ ra hai bên thân, phần bánh xe máy bên phải sát vạt áo. Biên đặt ở mép áo trắng nơi tiếp giáp xe máy/người lái phía sau, vì tương phản sáng–tối ở đó rõ nhất; chỗ ngón tay chạm vạt áo vẫn giữ trong mask.
- Nếu dùng gợi ý sau đó: vùng gợi ý sai/đúng, hành động sửa/giữ và lý do: notebook giữ object tự vẽ và bỏ 1 object AI trùng với nó. Sau khi nạp nhãn nháp vào CVAT, tôi sửa 2 lỗi: (1) `000000458325` — xe nôi do nhóm người đẩy ở cuối phố bị AI gán `motorcycle` → xóa, vì xe nôi không thuộc 6 lớp (ảnh còn 20 object); (2) `000000373353` — AI bỏ sót người mặc áo khoác đỏ trong nhóm đi qua đường bên phải → vẽ thêm `person`, tổng cộng thêm 2 người (ảnh từ 20 lên 22 object). Giữ đề xuất ở `000000181542` (17 object), gồm xe bán tải là `truck` và người lái tách riêng khỏi xe máy, vì đúng quy ước COCO.
- Nếu không dùng gợi ý: ghi “không dùng”; vẫn giải thích một quyết định gán nhãn của mình.

**Model, thời gian và lỗi model theo từng task.** Thời gian là *working time* do CVAT local tự ghi (analytics event `send:working_time`, chỉ tính lúc thao tác trong job), không tính thời gian chạy Colab/upload. Model lấy từ Hugging Face qua `transformers`, chạy bằng notebook `tools/ai-prelabel`, ngưỡng tin cậy 0.5.

| Task | Cách làm | Model đã dùng | Model chạy | Thời gian trong CVAT | Model sai gì → tôi sửa gì |
| --- | --- | --- | --- | --- | --- |
| easy_semantic (3 ảnh) | Vẽ tay | — | — | 49,3 phút ≈ **16 phút/ảnh** | — (lỗi tự mắc ở mục 3) |
| medium — object đầu | Vẽ tay | — | — | **3,7 phút** cho 1 người (94 điểm) | — |
| medium_instance (3 ảnh) | Nhãn nháp + sửa | `facebook/mask2former-swin-large-coco-panoptic` + `facebook/sam2.1-hiera-large` | Colab GPU, 41 giây, 58 object | 17,7 phút tổng; bỏ object đầu còn 14 phút ≈ **4,7 phút/ảnh** | Xe nôi gán `motorcycle` → xóa; sót người áo khoác đỏ và 1 người khác → vẽ thêm 2 `person`; ảnh Sài Gòn giữ nguyên 17 object |
| hard_panoptic (2 ảnh) | Nhãn nháp + sửa | như Medium (stuff + thing) | Colab GPU, 3,2 giây, 42 segment | 16,3 phút ≈ **8 phút/ảnh** | Làn đường bên phải ảnh cao tốc gán `vegetation` → chuyển sang `road`; 2 mask `truck` trùng ở xe chở ô tô → xóa 1; dải cỏ ven đường ảnh đại lộ gán `building` → xóa khỏi building; sót đèn giao thông giữa đường → vẽ thêm `traffic light` |
| cp3_thin, cp4_curb | Vẽ tay | — | — | 4,7 và 4,2 phút | — |
| cp1_holes | Nhãn nháp + sửa | Mask2Former COCO panoptic (không SAM) | Colab **CPU** | 4,6 phút | Mái lều gán `bus` → xóa; xe van gán `bus` → đổi `car`. Mask xe mô tô giữ kính và vành bánh, không cần sửa |
| cp2_slice | Nhãn nháp, giữ nguyên | như cp1 | Colab CPU | 4,3 phút (kiểm) | Không sửa: hai xe sát nhau ở góc trái đã tách đúng thành 2 instance |
| cp5_occlusion | Nhãn nháp, giữ nguyên | như cp1 | Colab CPU | 0,6 phút (kiểm) | Không sửa: xe máy bị người lái che vẫn là 1 mask |
| cp6_coverage | Nhãn nháp, giữ nguyên | `facebook/mask2former-swin-large-cityscapes-semantic` | Colab CPU | 2,5 phút (kiểm) | Không sửa |

Nhận xét: vẽ tay semantic mất khoảng 16 phút/ảnh, còn dùng nhãn nháp rồi sửa chỉ mất khoảng 5–8 phút/ảnh dù ảnh Medium/Hard nhiều vật hơn. Hai bên khác loại task và Easy là phần làm đầu tiên (còn làm quen CVAT), nên đây là so sánh thô, không phải đo đối chứng. Model làm nhanh phần vật rõ, nhưng vẫn sai class (xe nôi, mái lều, xe van), gộp sai vùng nền (mặt đường thành vegetation) và bỏ sót vật nhỏ hoặc bị che (người áo đỏ, đèn giao thông), nên vẫn cần người kiểm từng ảnh.

## 3. Một lỗi tôi tìm thấy và sửa

Chọn một lỗi **có thật** trong bài. Nếu công cụ lỗi khiến bạn chưa sửa được, ghi rõ đã thử gì và cần coach hỗ trợ gì; không ghi “đã sửa” khi chưa sửa.

- Task/ảnh/vùng: `easy_semantic` / `81ae7cbb-6bc63a4a.jpg` (đường phố có cây trụi lá, biển "30") / vỉa hè hai bên đường, nhất là vỉa hè bê tông bên phải.
- Lỗi thuộc loại: sai lớp + biên (ranh road–sidewalk).
- Bằng chứng tôi nhìn thấy: export lần 1 không có lớp `sidewalk` trong ảnh này (0% pixel), dù ảnh có vỉa hè rõ hai bên. Lần sửa thứ hai lại tô `road` tràn qua bó vỉa bên phải, phủ cả dải bê tông chỗ cột biển "30" (sidewalk chỉ còn ~1,5%, toàn ở lề trái).
- Quy tắc và hành động sửa: road dừng đúng ở mép bó vỉa; dải bê tông dọc đường và lối vào nhà là `sidewalk`; dải cỏ/bãi cỏ (có tuyết) phía sau để trống vì không thuộc 5 lớp. Kéo lại đỉnh polygon road về mép bó vỉa bên phải và vẽ polygon sidewalk cho dải bê tông. Sau sửa sidewalk ~4% ảnh, nằm đúng dải bê tông hai bên.
- Sau sửa đã Save và export lại chưa? Đã Save, export lại Segmentation mask 1.1 và thay `submissions/easy_semantic.zip`.

Nếu bạn **đã xem Summary tự đánh giá trên GitHub Actions hoặc tự chạy script**, ghi ngắn một kết quả liên quan lỗi vừa sửa (ví dụ task, metric trước/sau nếu có): chưa có điểm. Scorecard ba tier tối đa **82**, không phải điểm cuối trên 100. Không tự ghi PASS/top 3/bonus; người phụ trách xác nhận theo tiêu chí lớp. Không đưa file ground truth vào fork.

## 4. Ba ca chưa chắc hoặc đã cân nhắc

Mỗi ca là một **vùng cụ thể** khiến bạn phải cân nhắc hai cách hiểu. Ghi dấu hiệu nhìn thấy hoặc quy tắc đã dùng, rồi nêu quyết định hoặc câu hỏi cho coach. Không cần ba lỗi; ca đã quyết định được cũng hợp lệ.

| Ảnh/vị trí                                                     | Hai cách hiểu có thể                                                                  | Quy tắc/chứng cứ                                                                                                        | Quyết định hoặc câu hỏi cho coach                                                                                                                                                                   |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. `7ee6d192` (xa lộ) — sườn đồi cỏ khô màu nâu hai bên đường  | `vegetation` hay để trống                                                             | Theo Cityscapes, cỏ/thảm thấp là `terrain`, không phải `vegetation` (cây, bụi mọc đứng); `terrain` không có trong 5 lớp | Chỉ tô `vegetation` cho cụm cây xanh đậm trên đồi, sườn cỏ khô để trống. Coach xác nhận reference có theo cách này không?                                                                           |
| 2. `81ae7cbb` — cây trụi lá bên trái, trời lộ giữa các cành    | Tô cả tán thành một mảng `vegetation` hay khoét trời giữa cành thành `sky`            | Cành mảnh, khe trời nhỏ và dày; tô từng khe tốn thời gian nhưng biên chính xác hơn                                      | Tô cả tán thành một mảng `vegetation`. Có thể làm giảm IoU của `sky`/`vegetation` ở vùng này                                                                                                        |
| 3. Hard `000000460147` (cao tốc Nhật) — xe chở ô tô ở làn giữa | Cả xe là một `truck`, hay `truck` cho khung/cabin + từng ô tô được chở là `car` riêng | AI tạo 2 mask `truck` chồng nhau; ô tô chở trên xe nhìn thấy rõ nhưng không tự chạy trên đường                          | Xóa mask `truck` trùng, giữ 1 `truck` phủ cả xe; ô tô ở tầng trên vẫn là `car` riêng (hai mask còn chồng một phần). Câu hỏi cho coach: reference có tách các ô tô được chở thành `car` riêng không? |
