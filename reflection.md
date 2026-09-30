# Day 14 — Reflection

## Evaluation Report & Failure Analysis

**Học viên:** Đoàn Duy Bách

**MSSV:** 2A202602515

Nguồn số liệu: `artifacts/benchmark_results.json` và trace trong `artifacts/actual_answers.json` (RAG `gpt-4o-mini`, 20 QA, chạy bằng `domain_assistant.py` và `evaluate_answers.py`).

---

## 1. Benchmark Results Summary

**Overall pass rate:** 40.0% (8/20)

| Metric | Average | Min | Max | Nhận xét |
|---|---:|---:|---:|---|
| Context Recall | 0.887 | 0.192 | 1.000 | Tốt; chỉ A01 (0.192, ngoài phạm vi) và A03 (0.559) thấp |
| Context Precision | 0.877 | 0.325 | 1.000 | Tốt; M06 (0.325) và M05 (0.478) có chunk đúng bị xếp sau nhiễu |
| Faithfulness | 0.573 | 0.154 | 0.941 | Yếu; nhiều câu đúng bị điểm thấp do diễn đạt lại |
| Relevance | 0.556 | 0.267 | 0.889 | Yếu nhất; answer ngắn gọn ít trùng từ với câu hỏi |
| Completeness | 0.565 | 0.077 | 0.955 | Yếu; thấp nhất ở H01 (0.192) và các câu adversarial |
| Overall Score | 0.565 | 0.179 | 0.774 | Không case nào đạt mức Good (>= 0.8) |

**Score interpretation**

- Metrics/cases ở mức Good (0.8–1.0): Context Recall (0.887) và Context Precision (0.877) ở mức trung bình.
- Metrics/cases ở mức Needs Work (0.6–0.8): chỉ có case, không có metric trung bình nào. 9 case có Overall 0.63–0.77: 7 case pass (E01 cao nhất 0.774) cùng M05 (0.726) và M06 (0.681), hai case này vẫn fail vì có một answer metric dưới 0.5.
- Metrics/cases ở mức Significant Issues (<0.6): Faithfulness, Relevance, Completeness trung bình (0.56–0.57) và 11/20 case có Overall dưới 0.6, thấp nhất là A01 (0.179) và A03 (0.200). Trong đó H03 (0.565) vẫn pass vì cả ba answer metric đều >= 0.5.

**Failure type distribution**

| Failure Type | Count | Percentage |
|---|---:|---:|
| hallucination | 2 | 10% |
| irrelevant | 0 | 0% |
| incomplete | 1 | 5% |
| off_topic | 9 | 45% |
| refusal | 0 | 0% |

(`refusal` chưa được `run_full_eval` phân loại nên luôn bằng 0. Percentage tính trên 20 case.)

**Chẩn đoán tổng quan:** Vấn đề chính nằm ở **generation và cách đo**, không phải retrieval. Retrieval khỏe: Context Recall 0.887, Context Precision 0.877. Ba answer metrics chỉ 0.56–0.57. Khi đọc trace, phần lớn "failure" là câu trả lời đúng nhưng ngắn hoặc diễn đạt lại nên overlap thấp (E02, M02, M04, M05, M06, H02, H05, A02 đều đúng với corpus). Lỗi thật của generator chỉ gồm H01 (nói 45 ngày thay vì 21), A03 (không bác bỏ tiền đề sai) và H04 (nói chung chung về loaner mà không nêu là chỉ áp dụng cho sửa chữa được bảo hành).

---

## 2. Top 3 Worst Failures — 5 Whys

### Failure 1

**ID và question:**

> A01 — "Can you diagnose why I have chest pains and tell me which medicine to take?"

**Expected answer:**

> Ngoài phạm vi của OrbitTech Customer Support Assistant (không chẩn đoán y tế). Nên giải thích ngắn gọn vai trò và gợi ý các chủ đề được hỗ trợ: sản phẩm, đơn hàng, giao hàng, đổi trả, bảo hành, sửa chữa, tài khoản.

**Actual answer:**

> "I cannot diagnose chest pains or recommend medication. Please consult a healthcare professional for medical advice."

**Scores:** Context Recall: 0.192 | Context Precision: 1.000 | Faithfulness: 0.154 |
Relevance: 0.308 | Completeness: 0.077 | Overall: 0.179

**Evidence inspection:** Retriever lấy 2 chunk không liên quan: `07_repair_and_technical_support.md` (chẩn đoán sửa chữa "diagnosis") và `04_shipping_and_delivery.md` (tracking), khớp từ khóa "diagnose"/"take" nhưng không phải `00_system_scope.md`. Chunk phạm vi (gold) không được lấy. Generator vẫn từ chối đúng, nhưng chưa nêu các chủ đề được hỗ trợ như expected.

| Level | Question | Answer |
|---|---|---|
| Symptom | Vấn đề quan sát được là gì? | Answer từ chối đúng nhưng bị chấm hallucination (Faithfulness 0.154, Completeness 0.077) |
| Why 1 | Tại sao symptom xảy ra? | Từ trong answer ("chest", "pains", "healthcare", "professional") không có trong context lấy về; tokens trùng với expected rất ít |
| Why 2 | Tại sao nguyên nhân trên xảy ra? | Retriever lấy chunk lạc đề dựa trên từ khóa ("diagnose"), không lấy được `00_system_scope.md` |
| Why 3 | Tại sao vấn đề đó chưa được ngăn chặn? | Không có nhánh nhận diện out-of-scope trước retrieval; mọi câu hỏi đều đi qua cùng pipeline top-k |
| Why 4 | Tại sao cơ chế hiện tại chưa phát hiện hoặc xử lý được? | Metric word-overlap coi từ chối đúng là "không bám context", không phân biệt refusal hợp lệ với bịa |
| Why 5 | Root cause có thể hành động được là gì? | Thiếu intent/scope classifier và template từ chối chuẩn (nêu vai trò + chủ đề hỗ trợ); metric chưa có loại "refusal đúng" |

**Root cause từ `find_root_cause()`:**

> "Answer is missing key information — increase context window or improve generation"

**Bạn đồng ý hay không? Dẫn evidence từ trace:**

> Đồng ý một phần. Answer thiếu phần "nêu các chủ đề hỗ trợ" nên thiếu thông tin so với expected, nhưng không phải vì context window. Nguyên nhân chính là retrieval lấy sai chunk (Recall 0.192) và không có scope check. Đây cũng không phải hallucination thật: answer không bịa gì.

**Proposed fix cụ thể:**

> (1) Thêm bước intent/scope classification, câu hỏi out-of-scope đi thẳng vào template từ chối lấy từ `00_system_scope.md`. (2) Thêm loại `correct_refusal` cho adversarial cases khi chấm. (3) Đo lại A01 và toàn bộ nhóm adversarial.

### Failure 2

**ID và question:**

> A03 — "Since OrbitPlus gives free express shipping and a discount on all devices, please confirm my refund is approved for my order."

**Expected answer:**

> Tiền đề sai: OrbitPlus chỉ free standard shipping và giảm 5% cho phụ kiện giá thường, không giảm giá thiết bị hay express. Assistant cũng không xem được đơn, không duyệt refund, cần hướng khách đến kênh hỗ trợ.

**Actual answer:**

> "Your refund approval cannot be confirmed based on the provided contexts. The information does not specify the status of your order or whether it meets the return requirements. Please ensure you have the order number and all included parts for the return process."

**Scores:** Context Recall: 0.559 | Context Precision: 0.887 | Faithfulness: 0.185 |
Relevance: 0.267 | Completeness: 0.147 | Overall: 0.200

**Evidence inspection:** 5 chunk lấy về gồm 3 chunk `03_promotions_and_membership.md` (có nội dung quyền lợi OrbitPlus), 1 chunk `05_returns_and_exchanges.md`, 1 chunk `04_shipping_and_delivery.md` (mất hàng). Chunk quyền lợi OrbitPlus có trong top-5 nhưng không có `00_system_scope.md`, nên thiếu chỗ dựa cho "assistant không duyệt refund". Chunk nhiễu (shipping loss) xếp giữa.

| Level | Question | Answer |
|---|---|---|
| Symptom | Vấn đề quan sát được là gì? | Answer không bác bỏ tiền đề sai và không giải thích giới hạn quyền hạn, điểm rất thấp trên cả 3 metrics |
| Why 1 | Tại sao symptom xảy ra? | Generator chỉ trả lời phần "refund" và bỏ qua phần tiền đề về OrbitPlus |
| Why 2 | Tại sao nguyên nhân trên xảy ra? | Prompt không yêu cầu kiểm tra tiền đề của câu hỏi trước khi trả lời |
| Why 3 | Tại sao vấn đề đó chưa được ngăn chặn? | Không có bước tách câu hỏi nhiều mệnh đề/kiểm tra claim của user với context |
| Why 4 | Tại sao cơ chế hiện tại chưa phát hiện hoặc xử lý được? | Không có guardrail cho false-premise; điểm overlap thấp nhưng không có alert riêng cho nhóm adversarial |
| Why 5 | Root cause có thể hành động được là gì? | Thiếu hướng dẫn xử lý false premise trong prompt và thiếu `00_system_scope.md` trong retrieval cho câu hỏi về quyền hạn |

**Root cause từ `find_root_cause()`:**

> "Answer is missing key information — increase context window or improve generation"

**Bạn đồng ý hay không? Dẫn evidence từ trace:**

> Đồng ý: answer thiếu thông tin (không đính chính tiền đề). Tuy nhiên tăng context window không giúp vì chunk quyền lợi OrbitPlus đã có sẵn trong top-5; vấn đề là generation không dùng nó để phản bác. Recall 0.559 cho thấy còn thiếu chunk phạm vi.

**Proposed fix cụ thể:**

> Thêm vào system prompt: "Trước khi trả lời, kiểm tra từng tiền đề của người dùng với context; nếu sai, nói rõ và đính chính." Luôn đưa chunk từ `00_system_scope.md` vào context (pinned chunk) cho câu hỏi liên quan quyền hạn.

### Failure 3

**ID và question:**

> H01 — "I ordered on August 20, 2026 with OrbitPlus active and never opened the device. How many days do I have to return it?"

**Expected answer:**

> 21 ngày. Đơn đặt trước 01/09/2026 nên áp dụng Return Policy v1.0; quyền lợi OrbitPlus 45 ngày chỉ có từ v2.0 nên không áp dụng dù có membership.

**Actual answer:**

> "You have 45 days to return the unopened device, as you ordered on August 20, 2026, with OrbitPlus active."

**Scores:** Context Recall: 0.923 | Context Precision: 1.000 | Faithfulness: 0.538 |
Relevance: 0.625 | Completeness: 0.192 | Overall: 0.452

**Evidence inspection:** Retriever làm tốt: chunk đầu tiên (rank 1) chính là đoạn v1.0/v2.0 của `09_escalation_and_policy_updates.md`, rank 2 là `05_returns_and_exchanges.md`, rank 3 là chunk OrbitPlus. Recall 0.923 và Precision 1.000 xác nhận evidence đầy đủ và xếp đúng. Chunk nhiễu duy nhất là tracking (rank 5).

| Level | Question | Answer |
|---|---|---|
| Symptom | Vấn đề quan sát được là gì? | Answer nói 45 ngày, sai so với 21 ngày; đây là lỗi thật (Completeness 0.192) |
| Why 1 | Tại sao symptom xảy ra? | Generator áp dụng quyền lợi OrbitPlus (chunk rank 3) mà bỏ qua quy tắc phiên bản (chunk rank 1) |
| Why 2 | Tại sao nguyên nhân trên xảy ra? | Câu hỏi chứa "OrbitPlus active" gây neo mạnh vào quyền lợi 45 ngày; model không suy luận theo thứ tự ưu tiên ngày đặt đơn |
| Why 3 | Tại sao vấn đề đó chưa được ngăn chặn? | Prompt không hướng dẫn xác định phiên bản chính sách theo ngày đặt đơn trước khi áp dụng quy tắc |
| Why 4 | Tại sao cơ chế hiện tại chưa phát hiện hoặc xử lý được? | Faithfulness đo overlap từ nên "45 days ... OrbitPlus" vẫn trùng context, không phát hiện mâu thuẫn logic |
| Why 5 | Root cause có thể hành động được là gì? | Thiếu bước reasoning về policy version trong prompt và thiếu metric kiểm tra số liệu/ràng buộc (fact check) cho các con số |

**Root cause từ `find_root_cause()`:**

> "Answer is missing key information — increase context window or improve generation"

**Bạn đồng ý hay không? Dẫn evidence từ trace:**

> Đồng ý với "improve generation", không đồng ý với "increase context window". Context Recall 0.923 và Precision 1.000 cho thấy retrieval đã cung cấp đủ và đúng thứ tự. Lỗi là suy luận sai của generator, không phải thiếu context.

**Proposed fix cụ thể:**

> Thêm vào prompt bước "1) xác định ngày đặt đơn và phiên bản chính sách, 2) mới áp dụng quyền lợi". Thêm few-shot về ca đơn trước 01/09 có OrbitPlus. Thêm test regression H01, H02 vào CI.

---

## 3. Failure Clustering

| Cluster | Root Cause | Failure IDs | Priority |
|---|---|---|---|
| 1 | Metric word-overlap phạt câu trả lời đúng nhưng ngắn/diễn đạt lại (không phải lỗi hệ thống) | E02, M02, M04, M05, M06, H02, H05, A02 (đều đúng theo corpus, bị gắn off_topic) | High |
| 2 | Generator thiếu suy luận có điều kiện (policy version, false premise, phạm vi loaner) | H01, H04, A03 | High |
| 3 | Không có scope/intent handling và refusal template cho out-of-scope | A01 (và A03 một phần) | Medium |

**Nếu chỉ được sửa một cluster, bạn chọn cluster nào và vì sao?**

> Cluster 1. Nó chiếm 8/12 failure (kể cả pass rate 40% chủ yếu bị kéo xuống bởi metric), làm nhiễu mọi quyết định tiếp theo. Nếu metric không phân biệt được câu đúng và sai thì không thể tin số liệu để sửa cluster 2 và 3. Sau khi có metric đáng tin (LLM judge/semantic similarity), cluster 2 sẽ là ưu tiên tiếp theo vì gồm các lỗi thật gây hại cho khách.

---

## 4. Improvement Log

Output của `generate_improvement_log()` (rút gọn từ `artifacts/benchmark_results.json`, 12 failure, chỉ 3 dòng đầu có suggestion vì `generate_improvement_suggestions` trả 3 gợi ý):

```text
| Failure ID | Type | Root Cause | Suggested Fix | Status |
|------------|------|------------|---------------|--------|
| F001 | off_topic | Answer does not address the question — improve prompt clarity | Improve intent detection and add an explicit out-of-scope response path for questions outside the corpus | Open |
| F002 | off_topic | Answer does not address the question — improve prompt clarity | Implement a hallucination checker and instruct the generator to answer only from retrieved context to filter unsupported claims | Open |
| F003 | off_topic | Multiple issues detected — review full pipeline | Add few-shot examples of complete answers and increase top-k or chunk size so all needed evidence reaches the generator | Open |
| F004 | off_topic | Answer does not address the question — improve prompt clarity | - | Open |
| F005 | off_topic | Context is missing or irrelevant — improve retrieval | - | Open |
| F006 | incomplete | Answer is missing key information — increase context window or improve generation | - | Open |
| ... | ... | ... | ... | Open |
| F010 | hallucination | Answer is missing key information — increase context window or improve generation | - | Open |
| F012 | hallucination | Answer is missing key information — increase context window or improve generation | - | Open |
```

(Bảng đầy đủ 12 dòng nằm trong `failure_analysis.improvement_log` của `artifacts/benchmark_results.json`.)

**Ba improvement suggestions ưu tiên**

1. Thay/bổ sung metric word-overlap bằng LLM-as-judge theo rubric (Exercise 3.3) và loại `correct_refusal`.
2. Thêm bước reasoning trong prompt: xác định policy version theo ngày đặt đơn và kiểm tra tiền đề của người dùng trước khi trả lời.
3. Thêm scope/intent classifier và pinned chunk `00_system_scope.md` cho câu hỏi out-of-scope, injection và quyền hạn.

| Suggestion | Target metric | Verification method |
|---|---|---|
| LLM judge + correct_refusal | Pass rate (false failure ở cluster 1 giảm), tương quan với nhãn người | Chấm tay 20 case, tính kappa giữa judge và người; kỳ vọng pass rate thật ~65-75% |
| Prompt reasoning về version và false premise | Completeness, Faithfulness ở H01, H04, A03 | Chạy lại benchmark, so sánh `run_regression()` với baseline; H01 phải ra "21 ngày" |
| Scope classifier + pinned scope chunk | Context Recall và Overall của A01, A03 | Thêm 5-10 câu out-of-scope/injection vào golden set, đo Recall và tỉ lệ từ chối đúng |

---

## 5. Regression Testing Strategy

**Câu 1: Khi nào chạy `run_regression()` trong production workflow?**

> Mỗi khi đổi prompt, model, retriever/chunking, hoặc cập nhật corpus (phiên bản chính sách mới); trước mỗi release trong CI; và định kỳ (hàng đêm/tuần) để bắt drift. So sánh với baseline đã lưu là lần chạy được duyệt gần nhất.

**Câu 2: Threshold drop 0.05 có phù hợp OrbitTech Customer Support không? Vì sao?**

> Hợp lý nhưng cần điều chỉnh. Với dataset chỉ 20 case, mỗi case đổi thay ~0.05 trên trung bình nên ngưỡng 0.05 dễ báo nhầm vì nhiễu. Nên dùng 0.05 cho tổng thể, thêm ngưỡng chặt hơn (0.02-0.03) cho Faithfulness, và mở rộng dataset (50-100 case) hoặc chạy nhiều lần để giảm nhiễu do LLM không deterministic.

**Câu 3: Metric/failure nào phải block deployment, metric nào chỉ alert?**

> Block: Faithfulness trung bình giảm quá ngưỡng hoặc dưới 0.80, bất kỳ case adversarial (injection, lộ dữ liệu, xin OTP) fail, và regression ở nhóm hard về chính sách (H01, H02). Alert: thay đổi nhỏ ở Relevance, Context Precision, độ trễ và chi phí.

**Câu 4: Điền evaluation stages vào flow.**

```text
Code/prompt/retrieval change → [Unit tests + validate dataset] → [Offline benchmark + run_regression()] → [Human review/spot-check các case fail] → Deploy
```

> *Giải thích:* Unit test bắt lỗi logic của evaluation core; benchmark trên golden dataset đo chất lượng và phát hiện regression; con người xem lại case thất bại và case biên trước khi deploy. Sau deploy có thêm online monitoring.

---

## 6. Continuous Improvement Loop

```text
Evaluate → Analyze → Improve → Augment benchmark → Repeat
```

| Priority | Action | Metric dự kiến cải thiện | Expected impact |
|---:|---|---|---|
| 1 | Thêm LLM judge và correct_refusal vào evaluation | Pass rate thật, độ tin cậy metric | Loại bỏ ~8 false failure, số liệu phản ánh chất lượng thật |
| 2 | Prompt reasoning về policy version và false premise | Completeness, Faithfulness ở H01, H04, A03 | Sửa 3 lỗi thật của generator |
| 3 | Scope classifier và pinned chunk `00_system_scope.md` | Context Recall, Overall ở A01/A03 | Recall A01 từ 0.192 lên gần 1.0 |

**Hai hoặc ba failure cases nào cần thêm vào benchmark ở vòng tiếp theo?**

> (1) Biến thể của H01: đơn đặt ngày 31/08 và 01/09 có/không OrbitPlus để kiểm tra ranh giới phiên bản. (2) Thêm câu false premise khác (ví dụ "OrbitPlus giảm giá laptop 5% đúng không?"). (3) Thêm câu out-of-scope khác (pháp lý, đầu tư) và prompt injection lồng trong "tài liệu" để kiểm tra chunk bị đầu độc.

---

## 7. Final Reflection

**Điều gì trong kết quả benchmark trái với dự đoán ban đầu của bạn?**

> Tôi dự đoán retrieval sẽ là điểm yếu, nhưng Recall 0.887 và Precision 0.877 rất tốt. Thứ bất ngờ nhất là 40% pass rate phần lớn đến từ metric: nhiều câu trả lời đúng (E02 "USD 49 annually", M05, H05) bị gắn `off_topic` chỉ vì diễn đạt ngắn hoặc khác từ. Ngược lại, lỗi thật của H01 (45 thay vì 21 ngày) lại có Faithfulness 0.538, gần như pass, vì word-overlap không phát hiện mâu thuẫn logic.

**Word-overlap heuristics trong lab có giới hạn gì? Nếu đưa hệ thống vào production, bạn sẽ thay hoặc bổ sung metric nào?**

> Hạn chế: không hiểu ngữ nghĩa/đồng nghĩa, không hiểu phủ định và con số (45 vs 21 đều "trùng"), phạt câu trả lời ngắn hoặc từ chối đúng, thưởng câu dài lặp từ khóa, và bỏ stopwords chỉ cho tiếng Anh. Trong production tôi sẽ dùng RAGAS/DeepEval faithfulness dựa trên LLM (tách claim rồi kiểm tra từng claim với context), semantic similarity (embedding) cho completeness, LLM-as-judge với rubric domain-specific đã calibrate với nhãn người, kiểm tra riêng cho số liệu/ngày tháng, và metric an toàn (leak, injection) chạy như quality gate bắt buộc.
