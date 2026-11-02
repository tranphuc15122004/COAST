#!/usr/bin/env python3
"""Build Master AAAI 2027 Rebuttal Experimental Report (Single Consolidated Dossier).

Consolidates:
1. Executive Reviewer Analysis & Experimental Mapping (Reviewers GYgK, 9bi6, Bxhy, 3zAe, fpwa)
2. Detailed Empirical Sections for G1 through G9 (preserving 100% of tables, metrics, CIs, and mechanisms)
3. Verified Artifact & Source Parity Inventory

Outputs:
- AAAI_2027/README.md
- AAAI_2027_REBUTTAL_PACKAGE/EXPERIMENT_RESULTS.md
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path("/home/tuantb/mardam")
AAAI_DIR = ROOT / "AAAI_2027"
PACKAGE_DIR = ROOT / "AAAI_2027_REBUTTAL_PACKAGE"

ROMAN_MAP = {
    1: "I", 2: "II", 3: "III", 4: "IV", 5: "V",
    6: "VI", 7: "VII", 8: "VIII", 9: "IX"
}

OPENING_SECTION = """# COAST (VECTRA) — Báo Cáo Tổng Lực Thực Nghiệm Phản Biện AAAI 2027
## (Comprehensive Rebuttal Experimental Dossier: Reviewer Analysis & Full G1–G9 Empirical Results)

> **Tài liệu Bàn giao Khoa học Duy nhất (Master Scientific Document):**  
> Tài liệu này hợp nhất toàn bộ kết quả thực nghiệm mới, phân tích cơ chế, dữ liệu đo đạc, kiểm định thống kê và mạch lập luận phản hồi cho kỳ phản biện **AAAI 2027** của bài báo:  
> **"COAST: Coordinated Ownership and Anticipatory Sequential routing for Dynamic Vehicle Routing with Time Windows"** (Code name: `VECTRA`).  
> 
> Nhằm phục vụ việc rà soát và chấp bút phản hồi của các đồng tác giả, toàn bộ nội dung của các hồ sơ thực nghiệm từ **G1 đến G9** cùng các kiểm định phi tham số Wilcoxon mới nhất được giữ **nguyên vẹn 100% chi tiết**, không lược bỏ bất kỳ bảng số liệu, khoảng tin cậy (CI) hay diễn giải kỹ thuật nào.

---

## MỤC LỤC TỔNG QUAN

* **[PHẦN MỞ ĐẦU: PHÂN TÍCH YÊU CẦU CỦA REVIEWERS & THIẾT LẬP HỆ THỐNG G1–G9](#phần-mở-đầu-phân-tích-yêu-cầu-của-reviewers--thiết-lập-hệ-thống-thực-nghiệm-g1g9)**
  * [1. Bối cảnh phản biện & Đánh giá tổng quan](#1-bối-cảnh-phản-biện--đánh-giá-tổng-quan)
  * [2. Bóc tách chi tiết từng Reviewer (GYgK, 9bi6, Bxhy, 3zAe, fpwa)](#2-bóc-tách-chi-tiết-từng-reviewer-gygk-9bi6-bxhy-3zae-fpwa)
  * [3. Ma trận ánh xạ từ yêu cầu Reviewer sang 9 nhóm thực nghiệm G1–G9](#3-ma-trận-ánh-xạ-từ-yêu-cầu-reviewer-sang-9-nhóm-thực-nghiệm-g1g9)
  * [4. Nguyên tắc phương pháp luận học thuật & Chuẩn kiểm soát công bằng](#4-nguyên-tắc-phương-pháp-luận-học-thuật--chuẩn-kiểm-soát-công-bằng)
* **[PHẦN I: G1 — FAIRNESS, TRAINING PROTOCOL & MULTI-SEED STABILITY](#phần-i-g1--fairness-training-protocol--multi-seed-stability)**
* **[PHẦN II: G2 — STRONG ONLINE BASELINES, CLASSICAL HEURISTICS & DECISION LATENCY BUDGETS](#phần-ii-g2--strong-online-baselines-classical-heuristics--decision-latency-budgets)**
* **[PHẦN III: G3 — COST DECOMPOSITION, FLEET CAPACITY & OBJECTIVE WEIGHT SENSITIVITY](#phần-iii-g3--cost-decomposition-fleet-capacity--objective-weight-sensitivity)**
* **[PHẦN IV: G4 — FUSION MECHANISM: TÍNH CẦN THIẾT CỦA NONLINEAR MLP & CƠ CHẾ SUY GIẢM B5](#phần-iv-g4--fusion-mechanism-tính-cần-thiết-của-nonlinear-mlp--cơ-chế-suy-giảm-b5)**
* **[PHẦN V: G5 — CAPACITY CONTROL & INFORMATION FLOW: BÓC TÁCH DUNG LƯỢNG MÔ HÌNH & CẤU TRÚC INDUCTIVE BIAS](#phần-v-g5--capacity-control--information-flow-bóc-tách-dung-lượng-mô-hình--cấu-trúc-inductive-bias)**
* **[PHẦN VI: G6 — FLEET COORDINATION & OWNERSHIP HEAD: CƠ CHẾ PHỐI HỢP ĐỘI XE & GIỚI HẠN KHOA HỌC](#phần-vi-g6--fleet-coordination--ownership-head-cơ-chế-phối-hợp-đội-xe--giới-hạn-khoa-học)**
* **[PHẦN VII: G7 — CANDIDATE-CONDITIONED SCORER & DOWNSTREAM CONTINUATION COST: BẢN CHẤT VÀ GIỚI HẠN CỦA NHÁNH LOOKAHEAD](#phần-vii-g7--candidate-conditioned-scorer--downstream-continuation-cost-bản-chất-và-giới-hạn-của-nhánh-lookahead)**
* **[PHẦN VIII: G8 — INFORMATION HORIZON & ACTION SPACE: CLAIRVOYANT HINDSIGHT REFERENCE & CHIẾN LƯỢC CHỜ TẠI CHỖ](#phần-viii-g8--information-horizon--action-space-clairvoyant-hindsight-reference--chiến-lược-chờ-tại-chỗ)**
* **[PHẦN IX: G9 — DECOUPLED SCALE & DISTRIBUTION GENERALIZATION: KHẢO SÁT ĐỘC LẬP QUY MÔ KHÁCH HÀNG, ĐỘI XE VÀ OOD TRANSFER](#phần-ix-g9--decoupled-scale--distribution-generalization-khảo-sát-độc-lập-quy-mô-khách-hàng-đội-xe-và-ood-transfer)**
* **[PHẦN KẾT: TỔNG KẾT ARTIFACTS & MÃ NGUỒN TÁI LẬP HỌC THUẬT](#phần-kết-tổng-kết-artifacts--mã-nguồn-tái-lập-học-thuật)**

---

## PHẦN MỞ ĐẦU: PHÂN TÍCH YÊU CẦU CỦA REVIEWERS & THIẾT LẬP HỆ THỐNG THỰC NGHIỆM (G1–G9)

### 1. Bối cảnh phản biện & Đánh giá tổng quan

Hội đồng thẩm định bài báo tại **AAAI 2027** bao gồm **5 Reviewer (`GYgK`, `9bi6`, `Bxhy`, `3zAe`, `fpwa`)**. Tất cả các reviewer đều công nhận:
1. **Tính thời sự và tầm quan trọng thực tiễn:** Bài toán Dynamic VRPTW với khách hàng xuất hiện ngẫu nhiên theo thời gian thực và ràng buộc cửa sổ thời gian cứng/mềm là bài toán thực tế rất khó, đòi hỏi cân bằng đa mục tiêu phức tạp.
2. **Thiết kế mô hình gọn nhẹ, tốc độ suy luận nhanh:** COAST phân rã chính sách thành 3 tín hiệu trực giao có thể giải thích được, chỉ mất khoảng $\sim 12\text{ ms}$ cho mỗi quyết định thời gian thực, bộ nhớ phụ trợ gọn gàng ($O(m \cdot n \cdot d)$).
3. **Phạm vi đối chuẩn rộng:** Đã đánh giá trên nhiều bộ benchmark (Solomon, PYTH, DoD $\times$ TW Grid).

Tuy nhiên, các reviewer đồng loạt chỉ ra các **thiếu sót cốt tử về phương pháp luận và đối chứng thực nghiệm**:
* **Nghi vấn về tính công bằng & ngẫu nhiên hạt giống (Fairness & Multi-Seed):** Toàn bộ kết quả trong bài báo chỉ dựa trên 1 checkpoint duy nhất của 1 seed huấn luyện; khoảng cách $1\% - 4\%$ trước MARDAM liệu có bị nuốt chửng bởi phương sai hạt giống? Các baseline (AM, PolyNet, MARDAM) có được huấn luyện công bằng trên cùng một objective và data split không?
* **Thiếu vắng các bộ giải online cạnh tranh thực sự:** Chưa so sánh với Heuristics cổ điển minh bạch (Nearest, EDD, Regret-2), chưa so sánh với Rolling-horizon Re-optimization (như Google OR-Tools) hay các phương pháp dự đoán theo kịch bản (Multiple Scenario Approach - MSA) dưới cùng trần thời gian thực tế.
* **Bản chất của hàm mục tiêu (Routing vs. Customer Selection):** Trong kịch bản $n=50, m=3, C=200$, sức chứa đội xe chỉ đáp ứng $53\%$ tổng nhu cầu. Chi phí phạt bỏ đơn ($w_P=2$) chiếm tới $76\%$ tổng cost. Liệu COAST thực sự tối ưu lộ trình hay chỉ đơn giản là học mẹo chọn đơn nhỏ để bỏ ít khách hơn?
* **Hoài nghi về tính xác thực của các module (Inductive Bias vs. Over-claiming):** Nhánh "Lookahead" có thực sự nhìn thấy trước tương lai khi không có cơ chế mô phỏng hay dự báo? Nhánh "Ownership" thực sự phối hợp đội xe hay chỉ là thêm tham số? Có thực sự cần mạng phi tuyến MLP để hợp nhất không hay chỉ cần tổ hợp tuyến tính?

---

### 2. Bóc tách chi tiết từng Reviewer (GYgK, 9bi6, Bxhy, 3zAe, fpwa)

#### Reviewer GYgK (Điểm khuyến nghị: 4)
* **W1 (Chất lượng văn phong & Định nghĩa tham số):** Chưa định nghĩa tường minh $w_L, w_P$, mức độ động (DoD), tỷ lệ cửa sổ thời gian (TW ratio).
* **W2 (Độ nhạy hàm mục tiêu):** Tác động của hai siêu tham số phạt $w_L$ (late penalty) và $w_P$ (pending penalty) đến kết quả.
* **W3a (So sánh Heuristic đơn giản):** Yêu cầu đối chứng với Nearest Feasible, Uniform Random, Earliest Due Date (EDD).
* **W3b (So sánh Bộ giải truyền thống mạnh):** Yêu cầu đối chuẩn với Rolling-horizon Re-optimization (OR-Tools/HGS), Multiple Scenario Approach (MSA), và các chiến lược chờ tại chỗ (Waiting strategies - Mitrović-Minić & Laporte 2004).
* **W3c (Khoảng cách tới Hindsight):** Cần đo lường Value of Information: COAST cách mốc biết trước tương lai hoàn hảo (Clairvoyant / Hindsight) bao xa?
* **W3d (Ngân sách thời gian thực tế):** Đo đạc latency theo từng quyết định và so sánh dưới các trần thời gian $100\text{ ms}$, $1\text{ s}$, $5\text{ s}$.
* **W4 (Tính cần thiết của MLP Fusion):** Biến thể B5 dùng trọng số $1:1:1$ chưa tune. Liệu một tổ hợp tuyến tính cố định (Fixed Linear) được tinh chỉnh tốt có đánh bại MLP? Cần bổ sung khoảng tin cậy theo cặp (paired CIs).
* **W5 (Tính công bằng của Baseline):** AM và PolyNet là bài toán tĩnh đơn xe; chúng được chuyển đổi sang dynamic multi-vehicle thế nào và có được retrain không?
* **W6 (Không gian hành động):** Cho phép xe chờ tại chỗ (`WAIT`) thay vì bắt buộc quay về depot.

#### Reviewer 9bi6 (Điểm khuyến nghị: 5)
* **W1 (Tính mới phương pháp):** Đặt câu hỏi về tính mới so với MatNet, EFormer, 2D-Ptr. Cho rằng nhánh Lookahead chỉ là một head chấm điểm phi tuyến thứ hai chứ không thực sự nhìn trước tương lai.
* **W2 (Bản chất bài toán & Multi-seed):** Khẳng định benchmark chủ yếu đo lường việc chọn khách để bỏ ($16.4/50$ khách bị bỏ, pending penalty chiếm $32.9/44.0$). Cần huấn luyện đa seed để chứng minh mức chênh lệch $1\% - 4\%$ không phải do seed noise. Làm rõ quy trình thích ứng AM/PolyNet và kiểm tra sự sai lệch standard deviation của MARDAM.
* **Q1–Q5:** Yêu cầu bóc tách chi phí thành phần (distance, late, pending); số khách bỏ rơi của từng baseline; cơ chế thực sự của Peer memory; tương quan của Lookahead với chi phí thực tế tương lai.

#### Reviewer Bxhy (Điểm khuyến nghị: 4)
* **W1 & Q1 (Bản chất Lookahead):** Head Lookahead khác một generic scoring MLP ở chỗ nào khi nhận cùng đầu vào? Có thực sự dự báo tương lai?
* **W2 & Q2 (Bản chất Ownership):** Ownership Head thực sự giúp phối hợp đội xe hay chỉ đơn thuần là biểu diễn lịch sử của xe? Khác gì so với joint vehicle-node scoring trong các nghiên cứu trước?
* **W3 & Q3 (Cơ chế Fusion):** Mạng MLP phi tuyến có vượt trội một hàm tuyến tính học được (Learned Linear) không?
* **W4 & Q4 (Khảo sát Quy mô Độc lập):** Các bảng thử nghiệm quy mô tăng đồng thời cả số khách $n$ và số xe $m$. Yêu cầu tách bạch độc lập: giữ $m$ tăng $n$, giữ $n$ tăng $m$, và kiểm tra sự thay đổi của tỷ lệ tải (`load_ratio`).

#### Reviewer 3zAe (Điểm khuyến nghị: 4)
* **W1 (Tính mô-đun của 3 nhánh):** Ba nhánh có chức năng độc lập thực sự hay chỉ là cộng dồn tham số và thông tin? Cần đối chứng với Generic Scorer có cùng tham số.
* **W2 (Cơ chế phân công của Ownership):** Ownership score có phản ánh đúng phân công lao động của đội xe hay không?
* **W3 & Q8 (Tương quan Lookahead & Chế độ Sparse):** Lookahead score có tương quan với downstream cost không? Tại sao Lookahead lại làm giảm hiệu năng trong chế độ mạng thưa thớt (`Sparse`)?
* **W4 (Cơ chế suy giảm của B5):** Tại sao biến thể Equal-weight B5 lại suy giảm thảm họa ở quy mô lớn ($n=200, n=400$ trong Figure 4)?
* **W5 & W6 (Baselines mạnh & Khảo sát hàm mục tiêu):** Cần đối chứng với các baseline online mạnh và khảo sát độ nhạy với trọng số phạt.

#### Reviewer fpwa (Điểm khuyến nghị: 5)
* **W1 (Tính công bằng của Baselines):** Các baseline có được huấn luyện trên cùng một pool dữ liệu và cùng hàm mục tiêu reward không?
* **W2 (Đa hạt giống ngẫu nhiên):** Cần đánh giá trên nhiều random seeds huấn luyện để báo cáo Mean $\pm$ SD và paired confidence intervals.
* **W5 (Bóc tách chi phí):** Yêu cầu phân rã rõ ràng chi phí: bao nhiêu là quãng đường, bao nhiêu là phạt trễ, bao nhiêu là phạt bỏ đơn.
* **W6 (Khoảng cách tới Hindsight):** Đo lường khoảng cách từ Online COAST tới nghiệm biết trước tương lai hoàn hảo (perfect-information reference).

---

### 3. Ma trận ánh xạ từ yêu cầu Reviewer sang 9 nhóm thực nghiệm G1–G9

Để giải quyết triệt để và có hệ thống toàn bộ các yêu cầu trên, chúng tôi thiết lập **Ma trận ánh xạ 5 Reviewer $\times$ 9 Nhóm thực nghiệm có kiểm soát**:

| Reviewer | Nhóm thực nghiệm chịu trách nhiệm chính | Các câu hỏi trọng tâm được giải quyết |
|:---:|:---:|---|
| **GYgK** | **G1, G2, G3, G4, G8** | Multi-seed; Baselines mạnh (OR-Tools, MSA, Heuristics); Latency budgets; Penalty sensitivity; MLP vs. Fixed weights; Hindsight gap; Waiting strategies. |
| **9bi6** | **G1, G2, G3, G5, G6, G7** | Multi-seed stability; Kiểm toán dung lượng; Phân rã chi phí; Bản chất phân công của Ownership; Bản chất thực sự của Lookahead. |
| **Bxhy** | **G1, G4, G5, G6, G7, G9** | Multi-seed; MLP vs. Learned Linear; Đối chứng Generic Scorer; Peer memory coordination; Tách độc lập quy mô $n$, $m$, tải trọng. |
| **3zAe** | **G1, G2, G3, G4, G5, G6, G7, G9** | Phân rã chức năng 3 nhánh; Giải mã suy giảm B5; Downstream correlation của Lookahead & giải thích Sparse; Penalty sensitivity; Baselines mạnh. |
| **fpwa** | **G1, G3, G8** | Cùng shared training pool; Multi-seed stability; Phân rã chi phí chi tiết; Đo lường khoảng cách tới Clairvoyant Hindsight. |

#### Bảng Danh mục 9 Nhóm Thực Nghiệm Kiểm Soát (G1 đến G9):

| Mã nhóm | Tên gọi Nhóm Thực nghiệm | Câu hỏi Khoa học Cốt lõi | Thiết kế Thực nghiệm Đối chứng | Tiêu chí Nghiệm thu / Metrics |
|:---:|---|---|---|---|
| **G1** | **Fairness, Protocol & Multi-Seed Stability** | Ưu thế của COAST có thực chất và bất biến qua các random seeds? | Huấn luyện lại 12 checkpoints độc lập (4 models $\times$ 3 seeds `42`, `123`, `456`) trên cùng shared data pools ($512.000$ train, $10.240$ val). Byte-for-byte simulator audit. | Mean $\pm$ SD trên Ref-500 và 10k; Kiểm định phi tham số Wilcoxon Signed-Rank ($p < 10^{-15}$). |
| **G2** | **Strong Baselines & Decision Latency** | COAST có hơn Heuristics và Online Solvers trong cùng ngân sách thời gian? | Đối chuẩn với Nearest, EDD, Random, Regret-2, Rolling OR-Tools, MSA trên 500 ID và 5 RQ3 stress suites ($2.500$ instances) ở $100\text{ ms}$, $1\text{ s}$, $5\text{ s}$. | Paired difference 95% Bootstrap CI; Quality–Latency Pareto Frontier; Latency p50/p95. |
| **G3** | **Cost Decomposition & Penalty Sensitivity** | Gain đến từ routing hay bỏ khách? Thứ hạng có đổi khi đổi trọng số phạt? | Phân rã 100% trajectory return ($8.11 \times 10^{-6}$ error); quét 4 mức sức chứa ($C=150 \to 500$); Lưới thừa số đầy đủ 36 ô ($4\text{ Caps} \times 9\text{ Penalty Weights}$). | Tỷ lệ Distance / Late / Pending; Thứ hạng và Paired CI trên toàn bộ 36/36 ô của ma trận. |
| **G4** | **Fusion Mechanism & B5 Degradation** | Có cần mạng phi tuyến MLP? Tại sao B5 suy giảm thảm họa ở $n=200, 400$? | Quét 102 vector có dấu trên frozen backbone; Huấn luyện 3 seeds $\times$ 3 kiến trúc (MLP vs Signed Fixed vs Learned Linear); Can thiệp nhân quả gỡ bỏ ownership trên trajectory B5. | Paired CI; Phân tích tương quan nghịch Attention-Lookahead; Phân tích bùng nổ cực trị $z$-score. |
| **G5** | **Capacity Control & Generic Scorer** | COAST thắng nhờ nhiều tham số hay nhờ Inductive Bias chuyên biệt? | Kiểm toán tham số 9 checkpoints; Xây dựng `GenericVECTRA` cân bằng $99.975\%$ tham số (-193 params); Huấn luyện scratch 3 seeds và Warm-start frozen backbone; Can thiệp cùng trạng thái P1. | Named vs Active params; Paired $\Delta$ Cost; Phân rã $93\%$ khoảng cách do lãng phí quãng đường ($+7.7\%$). |
| **G6** | **Fleet Coordination & Ownership Bias** | Peer Memory có tác động nhân quả? Ownership có giảm trùng lặp lộ trình? | Can thiệp snapshot (Zero-peer và Permutation invariance); Rollout gỡ bỏ peer memory trên 5 bộ dữ liệu; Đối chứng `Role-Aware Generic Scorer` (cân bằng $+0.054\%$ params); Đo Route IoU. | Tỷ lệ lật hành động ($25.15\%$); Permutation invariance ($0.00\%$ flip, $\text{KL} < 10^{-8}$); Common-served Route IoU. |
| **G7** | **Candidate-Conditioned Scorer (Lookahead)** | Head này thực sự đo lường điều gì? Tại sao suy giảm ở chế độ Sparse? | Khảo sát $4.281$ candidate rollouts trên 150 decision states dưới cùng realized futures; Phân rã phương sai thứ hạng 8D local edges; Phân tích tương quan riêng phần và Downstream Regret. | Spearman $\rho$; Concordance index; Rank $R^2 = 84.1\%$ từ đặc trưng cục bộ; Partial $\rho = +0.107$. |
| **G8** | **Information Horizon & Waiting Action** | Giá trị của thông tin tương lai là bao nhiêu? Cho xe chờ tại chỗ có lợi không? | Xây dựng Clairvoyant COAST & One-Shot Static OR-Tools tuân thủ release date $s_i \ge \max(e_i, a_i)$; Mở rộng simulator hỗ trợ `WAIT(Δ=5%)` không rò rỉ tương lai trên 4 chính sách $\times$ 4 chế độ chờ. | Empirical Value of Information ($\Delta_{\text{info}} = +0.39$, $+0.89\%$); Thay đổi chi phí và số khách khi có `wait_strategic`. |
| **G9** | **Decoupled Scale & OOD Transfer** | Khả năng tổng quát hóa khi tách riêng quy mô khách $n$, xe $m$, và tải trọng? | Thiết lập 12 điều kiện độc lập (8.000 episodes): Trục A (tách $n$), Trục B (tách $m$ cố định $C$), Trục B2 (tách $m$ đẳng tổng tải $m \times C = 600$), Trục C (OOD không gian), Trục D (OOD xuất hiện muộn). | Paired Difference-in-Differences (DiD); Phân tích load ratio; Sản lượng tải trọng vận chuyển. |

---

### 4. Nguyên tắc phương pháp luận học thuật & Chuẩn kiểm soát công bằng

Toàn bộ quá trình thực nghiệm bổ sung trong tài liệu này tuân thủ nghiêm ngặt 3 nguyên tắc khoa học chuẩn mực:

1. **Khoa học Khiêm tốn & Trung thực (Sober Science):**
   * Chúng tôi sẵn sàng thừa nhận các giới hạn của mô hình và tiếp thu nghiêm túc nhận xét của Reviewer:
     - **Chính thức đổi tên `LookaheadHead` thành `Auxiliary Candidate-Conditioned Scoring Head`** (G7), thừa nhận rằng $84.1\%$ phương sai thứ hạng đến từ đặc trưng cạnh cục bộ bước hiện tại và không tuyên bố mô hình có khả năng "tiên tri tương lai".
     - **Định vị chính xác `OwnershipHead` là một Soft Coordination Bias** (G6) giúp giảm thiểu chồng chéo tuyến đường, rút lại tuyên bố sơ bộ "nhanh hơn 2.4×" vì phép so sánh trước đây chỉ dựa trên proxy cấp toán tử thô.
     - **Thừa nhận ranh giới giữa MLP và Linear:** Trên checkpoint đã hội tụ, tổ hợp tuyến tính có dấu (Signed Linear) được tinh chỉnh tốt bám rất sát MLP về suy luận, nhưng chứng minh MLP vẫn vượt trội khi huấn luyện từ đầu (G4).
2. **Kiểm toán Parity & Chống rò rỉ dữ liệu (Byte-for-Byte Parity & No Leakage):**
   * Toàn bộ 7 file simulator, hàm mất mát và bộ sinh dữ liệu đạt byte-for-byte parity SHA-256 giữa COAST và baselines.
   * Tách biệt 100% giữa tập Train, Validation, và Test. Các tập kiểm tra fresh seeds đều được kiểm tra chống trùng lặp băm dữ liệu.
3. **Tiêu chuẩn Thống kê Nghiêm ngặt (Rigorous Statistical Standards):**
   * Mọi so sánh theo cặp đều đi kèm **khoảng tin cậy 95% Bootstrap phân tầng (Hierarchical Bootstrap 95% CI)** với ít nhất 5.000 lần lấy mẫu lại.
   * Bổ sung kiểm định thống kê phi tham số **Wilcoxon Signed-Rank Test** độc lập với giả định chuẩn, khẳng định tính ưu việt của COAST đạt mức ý nghĩa thống kê vượt trội ($p < 10^{-15}$).

---
"""

CLOSING_SECTION = """
---

## PHẦN KẾT: TỔNG KẾT ARTIFACTS & MÃ NGUỒN TÁI LẬP HỌC THUẬT

Toàn bộ các thực nghiệm phản biện từ G1 đến G9 đã được lập trình, kiểm toán và lưu trữ thành các tài sản khoa học (artifacts) độc lập, sẵn sàng cho việc kiểm chứng và tái lập mã nguồn:

### 1. Bản đồ Thư mục Lưu trữ Dữ liệu Tổng hợp (`AAAI_2027_REBUTTAL_PACKAGE/key_data/`)

| Mã nhóm | File Dữ liệu Tổng hợp Cốt lõi | Mô tả Nội dung Đo đạc |
|:---:|---|---|
| **G1** | [`G1_multiseed_evaluation_full.json`](file:///home/tuantb/mardam/AAAI_2027_REBUTTAL_PACKAGE/key_data/G1_multiseed_evaluation_full.json) | Kết quả chi tiết 12 checkpoints trên Ref-500 và Test 10.240 |
| **G1** | [`output/multiseed_g1_prebest/wilcoxon/wilcoxon_results.json`](file:///home/tuantb/mardam/output/multiseed_g1_prebest/wilcoxon/wilcoxon_results.json) | Kết quả kiểm định Wilcoxon signed-rank, rank-biserial $r$ và t-test |
| **G1** | [`G1_ARTIFACTS.json`](file:///home/tuantb/mardam/AAAI_2027_REBUTTAL_PACKAGE/key_data/G1_ARTIFACTS.json) | Manifest kiểm toán SHA-256 mã nguồn và checkpoints |
| **G2** | [`G2_id500_rolling_per_instance.csv`](file:///home/tuantb/mardam/AAAI_2027_REBUTTAL_PACKAGE/key_data/G2_id500_rolling_per_instance.csv) | Dữ liệu so sánh từng instance Rolling OR-Tools vs COAST |
| **G2** | [`G2_frontier_summary.csv`](file:///home/tuantb/mardam/AAAI_2027_REBUTTAL_PACKAGE/key_data/G2_frontier_summary.csv) | Đánh giá Pareto đánh đổi chất lượng - thời gian tính toán (100ms, 1s, 5s) |
| **G3** | [`G3_capacity_penalty_cross_summary.csv`](file:///home/tuantb/mardam/AAAI_2027_REBUTTAL_PACKAGE/key_data/G3_capacity_penalty_cross_summary.csv) | Ma trận 36 ô quét sức chứa đội xe $C$ và hệ số phạt $w_L, w_P$ |
| **G4** | [`G4_multiseed_paired_summary.csv`](file:///home/tuantb/mardam/AAAI_2027_REBUTTAL_PACKAGE/key_data/G4_multiseed_paired_summary.csv) | Kết quả huấn luyện đối chứng MLP vs Linear 3 seeds |
| **G5** | [`G5_multiseed_paired_summary.csv`](file:///home/tuantb/mardam/AAAI_2027_REBUTTAL_PACKAGE/key_data/G5_multiseed_paired_summary.csv) | Kết quả huấn luyện đối chứng Generic Scorer 3 seeds |
| **G5** | [`G5_parameter_inventory.csv`](file:///home/tuantb/mardam/AAAI_2027_REBUTTAL_PACKAGE/key_data/G5_parameter_inventory.csv) | Bảng kiểm kê tham số Named vs Active của 9 mô hình |
| **G6** | [`G6_role_control_hierarchical_summary.csv`](file:///home/tuantb/mardam/AAAI_2027_REBUTTAL_PACKAGE/key_data/G6_role_control_hierarchical_summary.csv) | Phân tích Hierarchical Bootstrap đối chứng Role-Aware Scorer |
| **G6** | [`G6_onestep_interventions_summary.csv`](file:///home/tuantb/mardam/AAAI_2027_REBUTTAL_PACKAGE/key_data/G6_onestep_interventions_summary.csv) | Can thiệp một bước lật hành động Zero-peer và Permutation |
| **G7** | [`G7_summary_results.json`](file:///home/tuantb/mardam/AAAI_2027_REBUTTAL_PACKAGE/key_data/G7_summary_results.json) | Kết quả 21 chỉ số tương quan 4.281 rollouts hạ nguồn |
| **G7** | [`G7_regime_and_regret_summary.json`](file:///home/tuantb/mardam/AAAI_2027_REBUTTAL_PACKAGE/key_data/G7_regime_and_regret_summary.json) | Bảng Regret và đối chiếu 4 chế độ ID/Burst/Tight/Sparse |
| **G8** | [`G8_hindsight_summary.csv`](file:///home/tuantb/mardam/AAAI_2027_REBUTTAL_PACKAGE/key_data/G8_hindsight_summary.csv) | So sánh Online COAST vs Clairvoyant Reference và One-Shot OR |
| **G8** | [`G8_waiting_summary.csv`](file:///home/tuantb/mardam/AAAI_2027_REBUTTAL_PACKAGE/key_data/G8_waiting_summary.csv) | Đánh giá 4 chiến lược chờ trên 4 chính sách |
| **G9** | [`G9_condition_summary.csv`](file:///home/tuantb/mardam/AAAI_2027_REBUTTAL_PACKAGE/key_data/G9_condition_summary.csv) | Đánh giá toàn diện 12 điều kiện tách trục (8.000 episodes) |
| **G9** | [`G9_paired_did.csv`](file:///home/tuantb/mardam/AAAI_2027_REBUTTAL_PACKAGE/key_data/G9_paired_did.csv) | Phân tích Paired Difference-in-Differences (DiD) |

---

### 2. Bộ Scripts Thực Thi & Unit Tests Tự Động (`AAAI_2027/tests/`)

Toàn bộ các khẳng định định lượng đều được bảo đảm bằng các bài kiểm thử tự động (PyTest):
* `AAAI_2027/tests/test_g4*.py`: **19/19 PASSED** (kiểm tra tính đúng đắn của phép quét trọng số và can thiệp B5).
* `AAAI_2027/tests/test_g5.py`: **7/7 PASSED** (kiểm tra cân bằng tham số và tính bất biến hoán vị).
* `AAAI_2027/tests/test_g7.py`: **3/3 PASSED** (kiểm tra tính bitwise parity của 4.281 rollouts).
* `AAAI_2027/tests/test_g9.py`: **3/3 PASSED** (kiểm tra tính toàn vẹn của 12 điều kiện tách trục).

Báo cáo này cung cấp đầy đủ cơ sở thực nghiệm vững chắc, dữ liệu minh bạch và luận cứ khoa học để hoàn thiện bản phản hồi chính thức tới Ban chương trình và các Reviewer của hội nghị **AAAI 2027**.
"""

def format_g_content(idx: int, raw_text: str) -> str:
    """Format single G file content to fit nested master document hierarchy."""
    lines = raw_text.strip().split("\n")
    out = []
    
    # First line: # G{idx} — Title
    first_line = lines[0]
    m = re.match(r'^#\s*G\d+\s*[—\-–:]\s*(.*)$', first_line)
    if m:
        title = m.group(1).strip()
    else:
        title = first_line.lstrip("#").strip()
        
    out.append(f"## PHẦN {ROMAN_MAP[idx]}: G{idx} — {title}")
    out.append("")
    
    for line in lines[1:]:
        # Adjust headers
        if line.startswith("##### "):
            out.append("###### " + line[6:])
        elif line.startswith("#### "):
            out.append("##### " + line[5:])
        elif line.startswith("### "):
            out.append("#### " + line[4:])
        elif line.startswith("## "):
            out.append("### " + line[3:])
        elif line.startswith("# "):
            out.append("## " + line[2:])
        else:
            out.append(line)
            
    return "\n".join(out)

def build_master_document():
    print("Building Master AAAI 2027 Rebuttal Report...")
    parts = [OPENING_SECTION.strip()]
    
    for idx in range(1, 10):
        file_path = AAAI_DIR / f"G{idx}.md"
        if not file_path.exists():
            print(f"Error: {file_path} does not exist!")
            sys.exit(1)
        raw_text = file_path.read_text(encoding="utf-8")
        formatted = format_g_content(idx, raw_text)
        parts.append(formatted)
        print(f"  + Integrated G{idx}.md ({len(raw_text.splitlines())} lines)")
        
    parts.append(CLOSING_SECTION.strip())
    
    full_content = "\n\n---\n\n".join(parts) + "\n"
    
    # Output paths
    out_readme = AAAI_DIR / "README.md"
    out_package_results = PACKAGE_DIR / "EXPERIMENT_RESULTS.md"
    
    out_readme.write_text(full_content, encoding="utf-8")
    print(f"Successfully generated: {out_readme} ({len(full_content.splitlines())} lines, {len(full_content.encode('utf-8'))} bytes)")
    
    out_package_results.write_text(full_content, encoding="utf-8")
    print(f"Successfully synchronized: {out_package_results} ({len(full_content.splitlines())} lines)")

if __name__ == "__main__":
    build_master_document()
