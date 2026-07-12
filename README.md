# GPT

> Xây dựng mô hình ngôn ngữ từ những thành phần cơ bản để học Deep Learning, NLP, Transformer và quy trình nghiên cứu mô hình có thể kiểm thử, giải thích và tái lập.

## MỤC LỤC

1. [GPT là gì?](#1-gpt-là-gì)
2. [Scope và non-goals](#2-scope-và-non-goals)
3. [Current status](#3-current-status)
4. [Learning path](#4-learning-path)
5. [Requirements](#5-requirements)
6. [Quick start](#6-quick-start)
7. [Project structure](#7-project-structure)
8. [Command và execution map](#8-command-và-execution-map)
9. [Development corpus](#9-development-corpus)
10. [Phase roadmap](#10-phase-roadmap)
11. [Review gate policy](#11-review-gate-policy)
12. [Reproducibility principles](#12-reproducibility-principles)
13. [Limitations](#13-limitations)
14. [Contributing và next steps](#14-contributing-và-next-steps)

---

## 1. GPT là gì?

GPT là viết tắt của **Generative Pre-trained Transformer**: một họ mô hình ngôn ngữ sử dụng kiến trúc Transformer để dự đoán token tiếp theo dựa trên các token đã xuất hiện trước đó.

Dự án này không bắt đầu bằng một mô hình GPT hoàn chỉnh. Thay vào đó, dự án xây dựng từng thành phần nhỏ theo thứ tự:

```text
text
→ tokenizer
→ token IDs
→ train/validation/test split
→ next-token batches
→ count baselines
→ neural baselines
→ longer-context models
→ causal self-attention
→ Transformer blocks
→ decoder-only MiniGPT
```

Cách tiếp cận này giúp mỗi bước đều có thể:

- giải thích được về mặt lý thuyết;
- kiểm thử độc lập;
- so sánh với baseline đơn giản hơn;
- tái lập bằng cùng dữ liệu, cấu hình và random seed;
- phát hiện lỗi trước khi độ phức tạp của mô hình tăng lên.

### Hai tầng định hướng của dự án

Dự án có hai tầng mục tiêu khác nhau.

#### Tầng 1 — Learning/research MVP

Mục tiêu hiện tại là xây dựng một mô hình ngôn ngữ tự hồi quy nhỏ bằng Julia để nghiên cứu:

- character-level tokenization;
- next-token prediction;
- unigram và bigram language models;
- neural language models;
- embeddings và context windows;
- causal self-attention;
- decoder-only Transformer;
- huấn luyện, đánh giá, checkpoint và generation;
- quy trình thí nghiệm có thể tái lập.

Tầng này ưu tiên:

- khả năng giải thích;
- tính đúng đắn;
- kiểm thử;
- khả năng tái lập;
- giá trị học tập và nghiên cứu.

Tầng này không ưu tiên:

- kích thước mô hình;
- chất lượng sinh văn bản ở quy mô thực tế;
- tốc độ huấn luyện lớn;
- cạnh tranh với các mô hình ngôn ngữ thương mại.

#### Tầng 2 — Tầm nhìn production dài hạn

Định hướng dài hạn là phát triển một trợ lý AI được cá nhân hóa cho việc học tập và công việc, có thể hỗ trợ:

- hội thoại và suy luận nhiều bước;
- tìm kiếm, tổng hợp và nghiên cứu thông tin;
- tự động hóa workflow;
- lập lịch và theo dõi công việc;
- kết nối với các website, ứng dụng và dịch vụ khác;
- sử dụng ngữ cảnh cá nhân để hỗ trợ công việc phù hợp hơn.

Tầm nhìn production:

- không phải bản sao của ChatGPT;
- không chỉ là một chatbot hỏi–đáp;
- không đặt mục tiêu cạnh tranh trực tiếp với các foundation model lớn;
- hướng tới vai trò assistant/agent cá nhân cho chủ dự án.

Các chức năng production trên là **định hướng tương lai**, chưa phải trạng thái hiện tại của repository.

### Dự án dành cho ai?

Đối tượng sử dụng chính là cá nhân chủ dự án.

Repository đồng thời đóng vai trò:

- tài liệu tự học Deep Learning và NLP;
- phòng thí nghiệm cho các mô hình ngôn ngữ nhỏ;
- nơi lưu bằng chứng thực nghiệm;
- nền tảng để tiến dần tới decoder-only Transformer;
- cơ sở nghiên cứu cho trợ lý AI cá nhân trong tương lai.

### Mục tiêu học tập chính

Mục tiêu cốt lõi của learning MVP là hiểu và tự xây dựng một **decoder-only Transformer** từ những thành phần cơ bản, thay vì chỉ gọi API hoặc sử dụng trực tiếp một implementation GPT có sẵn.

---

## 2. Scope và non-goals

### Trong scope hiện tại

Dự án tập trung vào:

- xử lý corpus nhỏ và cố định;
- tokenizer ở mức ký tự;
- encode/decode token;
- chia dữ liệu train, validation và test;
- tạo batch cho bài toán next-token prediction;
- unigram và count-based bigram baselines;
- neural bigram model;
- cross-entropy, negative log-likelihood và perplexity;
- gradient, SGD và training loop;
- checkpoint, evaluation và text generation;
- xây dựng dần embeddings, attention và Transformer;
- kiểm thử các invariant quan trọng;
- lưu bằng chứng và kết quả thí nghiệm.

### Ngoài scope hiện tại

Repository hiện tại không tuyên bố cung cấp:

- một foundation model đã được pre-train trên dữ liệu lớn;
- một sản phẩm thay thế ChatGPT;
- chất lượng sinh văn bản ở mức production;
- distributed training;
- huấn luyện đa GPU;
- inference service có khả năng mở rộng lớn;
- production security hoàn chỉnh;
- long-term memory hoàn chỉnh;
- web search hoặc deep research hoàn chỉnh;
- hệ thống agent production;
- cam kết tương thích ngược ổn định cho toàn bộ API.

### Nguyên tắc giới hạn phạm vi

Mỗi phase chỉ thêm một nhóm khái niệm mới.

Ví dụ:

```text
count bigram
→ thêm tham số học được
→ neural bigram
→ thêm context dài hơn
→ embedding + MLP
→ thêm attention
→ Transformer
```

Dự án không nhảy trực tiếp đến kiến trúc phức tạp nếu các thành phần nền tảng chưa được hiểu, kiểm thử và ghi nhận bằng chứng.

---

## 3. Current status

Review Gate 0 đã xác nhận nền tảng hiện tại có thể được thiết lập và kiểm tra từ một clean clone.

Điều đó có nghĩa là corpus, tokenizer, data pipeline và các baseline hiện tại có nền tảng tái lập được. Điều đó **không có nghĩa** Transformer hoặc MiniGPT hoàn chỉnh đã được triển khai.

| Thành phần | Trạng thái | Ghi chú |
| --- | --- | --- |
| Julia package setup | Hoàn thành | Có `Project.toml`, môi trường project và test target |
| Clean-clone reproducibility | Hoàn thành | Corpus cần thiết có sẵn trong repository |
| Development corpus | Hoàn thành trong scope hiện tại | Có checksum và canonical statistics |
| Character tokenizer | Hoàn thành trong scope hiện tại | Có encode, decode và deterministic vocabulary |
| Token split | Hoàn thành trong scope hiện tại | Có train, validation và test splits |
| Batch generation | Hoàn thành trong scope hiện tại | Có next-token shift và seeded reproducibility |
| Unigram baseline | Hoàn thành trong scope hiện tại | Có fit, evaluate và generation |
| Count bigram baseline | Hoàn thành trong scope hiện tại | Có smoothing, evaluate và generation |
| Neural bigram | Đã triển khai | Có training, evaluation, generation và checkpoint |
| Longer-context data pipeline | Chưa hoàn thành | Thuộc phase tiếp theo |
| Embedding + MLP context model | Chưa triển khai | Thuộc roadmap |
| Causal self-attention | Chưa triển khai | Thuộc roadmap |
| Multi-head attention | Chưa triển khai | Thuộc roadmap |
| Transformer block | Chưa triển khai | Thuộc roadmap |
| Decoder-only MiniGPT | Chưa triển khai | Mục tiêu của các phase sau |
| Local assistant product | Chưa triển khai | Tầm nhìn dài hạn |
| Production readiness | Chưa đạt | Không phải tuyên bố của trạng thái hiện tại |

Trạng thái tổng quát:

```text
Foundation: ready
Tokenizer and data pipeline: available
Count baselines: available
Neural bigram: available
Transformer/GPT: planned, not yet implemented
Production assistant: long-term vision
```

---

## 4. Learning path

### 4.1 Text và tokenizer

Mô hình không trực tiếp xử lý chuỗi văn bản. Văn bản trước tiên phải được biến đổi thành các số nguyên gọi là token IDs.

```text
"hello"
→ ['h', 'e', 'l', 'l', 'o']
→ [17, 9, 21, 21, 24]
```

Character tokenizer trong dự án duy trì hai chiều ánh xạ:

```text
character → token ID
token ID → character
```

Invariant quan trọng:

```julia
decode(tokenizer, encode(tokenizer, text)) == text
```

Nếu invariant này không đúng, dữ liệu đầu vào hoặc kết quả generation đã bị thay đổi ngoài ý muốn.

Vocabulary được tạo theo thứ tự ổn định để cùng một corpus luôn tạo ra cùng một mapping.

### 4.2 Token split và next-token batches

Sau khi encode, token IDs được chia theo thứ tự thành:

```text
train
validation
test
```

Không shuffle toàn bộ chuỗi trước khi chia vì thứ tự token là một phần của dữ liệu ngôn ngữ.

Bài toán huấn luyện là dự đoán token tiếp theo.

Ví dụ:

```text
input:   h e l l
target:  e l l o
```

Ở dạng token IDs, invariant tương ứng là:

```julia
x[2:end] == y[1:end-1]
```

Mỗi phần tử trong `y` là token xuất hiện ngay sau phần tử tương ứng trong `x`.

### 4.3 Unigram baseline

Unigram model bỏ qua context và chỉ học xác suất xuất hiện tổng thể của từng token:

\[
P(x_t)
\]

Unigram trả lời câu hỏi:

> Token nào thường xuất hiện trong corpus?

Đây là baseline thấp nhất để xác nhận:

- corpus đã được đọc đúng;
- distribution của token hợp lệ;
- evaluation pipeline hoạt động;
- generation bằng sampling hoạt động.

### 4.4 Count-based bigram

Bigram sử dụng token ngay trước đó để dự đoán token tiếp theo:

\[
P(x_t \mid x_{t-1})
\]

Count bigram học trực tiếp từ số lần các cặp token xuất hiện trong tập train.

Giai đoạn này giới thiệu:

- conditional probability;
- transition counts;
- smoothing;
- unseen transitions;
- negative log-likelihood;
- perplexity;
- deterministic generation với fixed seed.

Count bigram là baseline quan trọng vì nó đơn giản, dễ kiểm tra và có thể so sánh trực tiếp với neural bigram.

### 4.5 Neural bigram

Neural bigram vẫn giải quyết bài toán:

\[
P(x_t \mid x_{t-1})
\]

Điểm khác biệt là distribution không còn được tạo trực tiếp từ bảng đếm. Thay vào đó, mô hình học một bảng logits bằng gradient descent.

Giai đoạn này giới thiệu:

- trainable parameters;
- logits;
- softmax;
- cross-entropy;
- analytic gradients;
- gradient norm;
- stochastic gradient descent;
- training và validation loss;
- checkpoints;
- generation từ mô hình đã học.

Neural bigram giữ context rất đơn giản để người học tập trung vào optimization trước khi chuyển sang kiến trúc phức tạp hơn.

### 4.6 Longer-context model

Bigram chỉ nhìn một token trước đó.

Một mô hình context dài hơn sẽ học:

\[
P(x_t \mid x_{t-k},\ldots,x_{t-1})
\]

Giai đoạn này dự kiến giới thiệu:

- context window;
- block size;
- token embeddings;
- vector representation;
- flattening hoặc pooling;
- MLP prediction head.

### 4.7 Causal self-attention

Khi context dài hơn, việc gộp mọi token bằng một cấu trúc cố định trở nên hạn chế.

Self-attention cho phép mỗi vị trí học xem token nào trong quá khứ quan trọng đối với dự đoán hiện tại.

Causal self-attention phải bảo đảm:

```text
vị trí t chỉ được nhìn các vị trí ≤ t
```

Mô hình không được nhìn token tương lai, vì điều đó sẽ gây information leakage trong next-token prediction.

Các thành phần chính gồm:

```text
Query
Key
Value
attention scores
causal mask
softmax
weighted value aggregation
```

### 4.8 Transformer block

Một GPT-style Transformer block thường kết hợp:

```text
causal self-attention
+ residual connection
+ normalization
+ feed-forward network
+ residual connection
```

Multi-head attention cho phép nhiều attention head học những loại quan hệ khác nhau giữa các token.

### 4.9 Decoder-only MiniGPT

MiniGPT dự kiến được ghép từ:

```text
token embeddings
+ positional embeddings
+ stack of Transformer blocks
+ final normalization
+ vocabulary projection
+ next-token cross-entropy
```

Đây là kiến trúc decoder-only tự hồi quy.

Generation được thực hiện lặp lại:

```text
đọc prompt
→ dự đoán distribution của token tiếp theo
→ chọn hoặc sample một token
→ nối token vào context
→ lặp lại
```

### Vì sao phải đi theo thứ tự này?

Mỗi giai đoạn chỉ thêm một nguồn phức tạp mới:

| Giai đoạn | Khái niệm mới chính |
| --- | --- |
| Tokenizer | Biểu diễn văn bản bằng số |
| Split và batch | Tạo dữ liệu next-token |
| Unigram | Xác suất không có context |
| Count bigram | Xác suất có một token context |
| Neural bigram | Tham số học được và gradient descent |
| MLP context | Embeddings và context dài hơn |
| Attention | Học quan hệ động giữa các token |
| Transformer | Ghép attention, residual, normalization và MLP |
| MiniGPT | Huấn luyện và generation end-to-end |

Nhờ vậy, khi một lỗi xuất hiện, phạm vi nguyên nhân có thể được thu hẹp theo phase thay vì phải debug toàn bộ Transformer cùng lúc.

---

## 5. Requirements

### Bắt buộc

- Git
- Julia `1.12`
- Hệ điều hành có thể chạy Julia

Dự án được phát triển chủ yếu với Julia project environment.

Kiểm tra phiên bản Julia:

```bash
julia --version
```

Kết quả phải thuộc dòng phiên bản:

```text
julia version 1.12.x
```

### Không bắt buộc trong scope hiện tại

Luồng hiện tại không yêu cầu:

- Python;
- Node.js;
- CUDA;
- NVIDIA GPU;
- Docker;
- cloud account;
- external dataset download.

Các yêu cầu này có thể thay đổi trong những phase production hoặc integration tương lai.

---

## 6. Quick start

### 6.1 Clone repository

```bash
git clone https://github.com/Dyu20705/GPT.git
cd GPT
```

### 6.2 Cài dependencies

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Lệnh này sử dụng môi trường Julia được định nghĩa bởi `Project.toml`.

### 6.3 Kiểm tra package

```bash
julia --project=. -e "using MiniGPT; println(MiniGPT.hello())"
```

Expected output:

```text
MiniGPT is working
```

### 6.4 Kiểm tra corpus

```bash
julia --project=. scripts/inspect_corpus.jl
```

Các giá trị quan trọng cần đúng:

```text
File size bytes: 2156
SHA-256: 11c4625ab69b40e072e5a0b5a26084f52dc410821b4cdb58be326b376502f9b4
UTF-8 readable: true
Unicode NFC: true
Characters/tokens: 1677
Vocabulary size: 47
Train tokens: 1341
Validation tokens: 167
Test tokens: 169
Round-trip correct: true
Warnings: none
```

### 6.5 Chạy test suite

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

Test suite hiện kiểm tra:

- package loading;
- corpus SHA-256;
- UTF-8, NFC, BOM và line endings;
- deterministic vocabulary;
- encode/decode round trip;
- train/validation/test splits;
- next-token shift;
- batch shape;
- seeded batch reproducibility;
- unigram và count bigram;
- neural bigram;
- gradient và training behavior;
- checkpoint và generation behavior.

### 6.6 Kiểm tra count baselines

```bash
julia --project=. scripts/inspect_count_models.jl
```

Script này:

- đọc canonical corpus;
- tạo tokenizer và data splits;
- fit unigram;
- fit count bigram;
- tính NLL và perplexity;
- hiển thị các bigram phổ biến;
- sinh mẫu với fixed seed;
- chạy smoothing sweep;
- kiểm tra các invariant của probability tables.

### 6.7 Huấn luyện neural bigram

```bash
julia --project=. scripts/train_neural_bigram.jl
```

Script đọc cấu hình từ:

```text
configs/neural_bigram.toml
```

Checkpoint mặc định được lưu tại:

```text
checkpoints/phase2/neural_bigram.bin
```

### 6.8 Đánh giá neural bigram

Sau khi đã tạo checkpoint:

```bash
julia --project=. scripts/evaluate_neural_bigram.jl
```

Script báo các metric cho train, validation và test:

- negative log-likelihood;
- perplexity;
- accuracy;
- số lượng prediction targets.

### 6.9 Sinh văn bản

Dùng prompt mặc định:

```bash
julia --project=. scripts/generate_neural_bigram.jl
```

Dùng prompt tùy chỉnh:

```bash
julia --project=. scripts/generate_neural_bigram.jl "your prompt"
```

Lưu ý: tokenizer hiện tại hoạt động ở mức ký tự. Prompt chỉ nên chứa các ký tự đã có trong vocabulary của corpus.

---

## 7. Project structure

```text
GPT/
├── .github/
│   └── workflows/          GitHub Actions và CI
├── configs/
│   └── neural_bigram.toml  Cấu hình thí nghiệm neural bigram
├── data/
│   └── raw/
│       └── tiny_corpus.txt Canonical development corpus
├── experiments/            Báo cáo và bằng chứng thí nghiệm
├── scripts/                Entry points để inspect, train, evaluate và generate
├── src/                    Implementation có thể tái sử dụng
├── test/                   Automated test suite
├── checkpoints/            Generated model checkpoints
├── Project.toml            Julia package metadata và dependencies
├── Manifest.toml           Resolved dependency versions nếu được version-control
├── Review Gate 0.md        Bằng chứng review nền tảng
└── README.md               Learning và execution map
```

### `src/`

Chứa implementation chính:

```text
tokenizer
data split
batch generation
count models
objectives
neural bigram
training
checkpoint
```

Code trong `src/` nên có thể được import và sử dụng lại từ test hoặc script.

### `scripts/`

Chứa executable entry points cho người học và người phát triển.

Script chịu trách nhiệm:

- tải config;
- tải corpus;
- gọi API trong package;
- in kết quả dễ đọc;
- tạo checkpoint hoặc artifact khi cần.

Business logic hoặc model logic chính không nên bị nhốt hoàn toàn trong script.

### `test/`

Chứa automated tests bảo vệ:

- public API;
- mathematical invariants;
- edge cases;
- deterministic behavior;
- serialization và checkpoint compatibility trong scope hiện tại.

### `configs/`

Chứa cấu hình thí nghiệm như:

- corpus path;
- train/validation ratios;
- batch size;
- random seed;
- model initialization;
- learning rate;
- maximum training steps;
- generation settings;
- checkpoint location.

### `experiments/`

Chứa:

- kết quả chạy;
- bảng metric;
- generation samples;
- diễn giải kết quả;
- hạn chế;
- kết luận của từng phase.

### `checkpoints/`

Chứa model state được tạo trong quá trình huấn luyện.

Checkpoint là generated artifact, không phải source code và không mặc nhiên được xem là canonical nếu chưa có policy tương ứng.

---

## 8. Command và execution map

### Foundation verification

```text
Pkg.instantiate()
→ package smoke test
→ inspect corpus
→ Pkg.test()
```

Commands:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"

julia --project=. -e "using MiniGPT; println(MiniGPT.hello())"

julia --project=. scripts/inspect_corpus.jl

julia --project=. -e "using Pkg; Pkg.test()"
```

### Count-model workflow

```text
canonical corpus
→ tokenizer
→ token splits
→ fit unigram
→ fit count bigram
→ evaluate
→ inspect transitions
→ generate samples
→ smoothing sweep
```

Command:

```bash
julia --project=. scripts/inspect_count_models.jl
```

Script này không tạo neural checkpoint.

### Neural-bigram workflow

```text
configs/neural_bigram.toml
→ train_neural_bigram.jl
→ checkpoint
→ evaluate_neural_bigram.jl
→ generate_neural_bigram.jl
```

#### Train

```bash
julia --project=. scripts/train_neural_bigram.jl
```

Input chính:

```text
configs/neural_bigram.toml
data/raw/tiny_corpus.txt
```

Output chính:

```text
checkpoints/phase2/neural_bigram.bin
training metrics printed to the terminal
```

#### Evaluate

```bash
julia --project=. scripts/evaluate_neural_bigram.jl
```

Yêu cầu:

```text
checkpoint phải tồn tại
corpus phải khớp với tokenizer/checkpoint assumptions
```

#### Generate

```bash
julia --project=. scripts/generate_neural_bigram.jl
```

Hoặc:

```bash
julia --project=. scripts/generate_neural_bigram.jl "prompt"
```

Output gồm:

- sampled generation;
- greedy generation.

### Test workflow

```text
source change
→ run focused test nếu có
→ run full Pkg.test()
→ run relevant script
→ confirm clean working tree
```

Commands:

```bash
julia --project=. -e "using Pkg; Pkg.test()"
git status --short
```

### Corpus-change workflow

```text
edit corpus
→ inspect corpus
→ calculate new SHA-256
→ update tests
→ update documentation
→ rerun all relevant models
→ explain behavioral impact
```

Corpus không được thay đổi chỉ để làm một test đang lỗi trở thành pass.

---

## 9. Development corpus

MiniGPT sử dụng `data/raw/tiny_corpus.txt` làm **deterministic test fixture** cho giai đoạn nền tảng.

Corpus này được dùng để kiểm tra:

- character tokenizer và encode/decode round trip;
- phép chia train/validation/test;
- batch generation có thể tái lập;
- count-based và neural bigram baselines;
- các script kiểm tra dữ liệu cục bộ.

Corpus là dữ liệu phát triển và học tập có tính xác định. Nó không phải dataset đại diện cho ngôn ngữ tự nhiên ở quy mô thực tế.

### Reproducibility contract

Corpus do dự án tự xây dựng, được version-control trực tiếp và có sẵn trong mọi clean clone.

Dự án không yêu cầu:

- tải corpus từ nguồn ngoài;
- sao chép file ẩn từ máy phát triển;
- truy cập cloud storage;
- sử dụng một phiên bản dữ liệu không được ghi nhận.

| Thuộc tính | Giá trị chuẩn |
| --- | --- |
| Path | `data/raw/tiny_corpus.txt` |
| Encoding | UTF-8 |
| Unicode normalization | NFC |
| Line endings | LF |
| File size | 2156 bytes |
| Character tokens | 1677 |
| Vocabulary size | 47 |
| Train tokens | 1341 |
| Validation tokens | 167 |
| Test tokens | 169 |
| SHA-256 | `11c4625ab69b40e072e5a0b5a26084f52dc410821b4cdb58be326b376502f9b4` |

### Verify the fixture

Từ thư mục gốc của repository, chạy:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. scripts/inspect_corpus.jl
julia --project=. -e "using Pkg; Pkg.test()"
```

`inspect_corpus.jl` phải báo đúng SHA-256 và các thống kê chuẩn trong bảng trên.

Test suite cũng kiểm tra checksum để phát hiện mọi thay đổi ngoài ý muốn ở mức byte.

### Change policy

Mọi thay đổi đối với corpus được xem là thay đổi test fixture.

Pull request tương ứng phải:

1. giải thích lý do thay đổi dữ liệu;
2. cập nhật SHA-256;
3. cập nhật file size và canonical statistics;
4. chạy lại corpus inspection;
5. chạy lại tokenizer tests;
6. chạy lại split và batch tests;
7. chạy lại count-model baselines;
8. chạy lại neural-model tests hoặc experiments bị ảnh hưởng;
9. ghi rõ ảnh hưởng tới khả năng so sánh với kết quả cũ;
10. không chỉnh corpus chỉ để che một lỗi implementation hoặc làm test pass.

---

## 10. Phase roadmap

Roadmap dưới đây là bản đồ cấp dự án.

Một số file hoặc report cũ có thể sử dụng cách đánh số phase trước đây. Khi có khác biệt, roadmap trong README được xem là định hướng canonical cho công việc tiếp theo.

### Phase 0 — Foundation và reproducibility

Mục tiêu:

- Julia project có thể được cài từ clean clone;
- canonical corpus luôn tồn tại;
- checksum được ghi nhận;
- CI chạy corpus inspection và test suite;
- không có hidden local dependency;
- Review Gate 0 đưa ra quyết định rõ ràng.

Trạng thái:

```text
Completed for current scope
```

### Phase 1 — Data, tokenizer và batching

Mục tiêu:

- character tokenizer;
- deterministic vocabulary;
- encode/decode;
- train/validation/test split;
- next-token examples;
- reproducible batches;
- theory notes và edge-case tests.

Trạng thái:

```text
Core implementation available
Further documentation and edge-case strengthening may continue
```

### Phase 2 — Count baselines

Mục tiêu:

- unigram baseline;
- count-based bigram;
- smoothing;
- unseen-transition policy;
- NLL và perplexity;
- fixed-seed generation;
- baseline report.

Trạng thái:

```text
Core implementation available
Reporting and model-card work may continue
```

### Phase 3 — Neural bigram

Mục tiêu:

- learnable logits;
- stable cross-entropy;
- analytic gradients;
- SGD;
- training và validation loop;
- config-driven experiment;
- checkpoint;
- evaluation;
- generation;
- so sánh analytic gradient với autodiff.

Trạng thái:

```text
Core implementation and scripts available
Standardization and deeper evidence may continue
```

### Phase 4 — Longer-context neural model

Mục tiêu:

- block-size data pipeline;
- token embeddings;
- nhiều token context;
- MLP prediction head;
- training curves;
- experiment logging;
- so sánh với bigram baselines.

Trạng thái:

```text
Planned
```

### Phase 5 — Attention và Transformer block

Mục tiêu:

- causal self-attention;
- query, key và value projections;
- causal mask;
- multi-head attention;
- residual connections;
- normalization;
- feed-forward network;
- Transformer invariant tests;
- kiểm tra không có future-token leakage.

Trạng thái:

```text
Planned
```

### Phase 6 — Decoder-only MiniGPT

Mục tiêu:

- token embeddings;
- positional embeddings;
- stack of Transformer blocks;
- final normalization;
- language-model head;
- next-token cross-entropy;
- training trên tiny corpus;
- evaluation và generation;
- so sánh với toàn bộ baseline trước đó.

Trạng thái:

```text
Planned
```

### Phase 7 — Experiments và research reporting

Mục tiêu:

- experiment registry;
- config snapshot;
- corpus SHA;
- Git commit;
- random seed;
- train/validation/test metrics;
- ablation studies;
- generation samples;
- model card;
- research note;
- phân tích hạn chế và overfitting.

Trạng thái:

```text
Planned
```

### Phase 8 — Local product MVP

Mục tiêu:

- local CLI cho train, evaluate và generate;
- local inference API;
- target-user definition;
- workflow boundaries;
- safe generation defaults;
- quickstart demo;
- phân biệt research tool với end-user product.

Trạng thái:

```text
Planned
```

### Phase 9 — Production-style reliability

Mục tiêu:

- runtime profiles;
- config validation;
- structured logging;
- run IDs;
- metrics;
- latency reporting;
- checkpoint metadata;
- observability;
- failure handling;
- local production-style operation.

Trạng thái:

```text
Future work
```

### Long-term assistant/agent track

Sau khi learning MVP và model pipeline đủ ổn định, project có thể nghiên cứu thêm:

- personalized context;
- retrieval;
- web research;
- workflow automation;
- calendar và task integration;
- external app connections;
- tool use;
- agent orchestration;
- memory boundaries;
- permission và security model.

Đây là một track riêng, không được dùng để tuyên bố rằng foundation model hiện tại đã có các khả năng trên.

---

## 11. Review gate policy

Mỗi phase phải vượt qua một review gate trước khi được xem là hoàn thành.

### 11.1 Implementation evidence

Phase phải có implementation tối thiểu phù hợp với scope đã định nghĩa.

Ví dụ:

- public API;
- reusable module;
- executable script;
- config;
- output hoặc artifact cần thiết.

### 11.2 Automated tests

Phase phải có test cho:

- normal behavior;
- edge cases quan trọng;
- mathematical invariants;
- dimension hoặc shape constraints;
- invalid input;
- deterministic behavior nếu có seed;
- regression đã từng được phát hiện.

Test pass không tự động chứng minh implementation đúng hoàn toàn, nhưng phase không được xem là hoàn thành nếu test suite đang fail.

### 11.3 Executable evidence

Documentation phải cung cấp:

- command đã chạy;
- input;
- expected output hoặc expected behavior;
- kết quả pass/fail;
- failure explanation nếu có;
- commit hoặc PR chứa bằng chứng khi cần.

Một tuyên bố không có lệnh chạy hoặc test tương ứng chỉ được xem là ghi chú, không phải evidence.

### 11.4 Learning evidence

Mỗi phase phải giải thích:

- thành phần đang làm gì;
- đầu vào và đầu ra;
- công thức hoặc invariant chính;
- vì sao thành phần đó cần thiết;
- nó khác baseline trước đó như thế nào;
- nó còn hạn chế ở đâu.

### 11.5 Reproducibility evidence

Thí nghiệm phải ghi nhận các yếu tố liên quan:

- corpus path;
- corpus SHA-256;
- config;
- random seed;
- package environment;
- Git commit;
- checkpoint path;
- metric definitions.

### 11.6 Scope decision

Review gate phải đưa ra một trong các kết luận rõ ràng:

```text
Ready for next phase
```

hoặc:

```text
Blocked: fix current phase first
```

Không dùng các kết luận mơ hồ như:

```text
probably ready
mostly finished
seems acceptable
```

### 11.7 Không đồng nhất “gate pass” với “production-ready”

Một phase pass chỉ có nghĩa là phase đó đạt acceptance criteria trong scope hiện tại.

Ví dụ:

```text
Tokenizer gate passed
```

không có nghĩa:

```text
Tokenizer phù hợp với mọi ngôn ngữ hoặc production workload
```

Tương tự:

```text
Review Gate 0 completed
```

không có nghĩa:

```text
Transformer, MiniGPT hoặc production assistant đã hoàn thành
```

---

## 12. Reproducibility principles

### 12.1 Không phụ thuộc file ẩn

Mọi file bắt buộc để test hoặc chạy workflow nền tảng phải:

- được version-control;
- được tạo bởi script tái lập được;
- hoặc được mô tả rõ cách tải và kiểm tra integrity.

Không chấp nhận dependency chỉ tồn tại trên máy của một người phát triển.

### 12.2 Dữ liệu phải có provenance

Mỗi dataset hoặc fixture nên ghi:

- nguồn hoặc phương pháp tạo;
- mục đích;
- license hoặc usage assumption;
- encoding;
- normalization;
- kích thước;
- checksum;
- hạn chế;
- change policy.

### 12.3 Cố định random seed khi so sánh

Các experiment cần so sánh phải sử dụng seed được ghi nhận.

Seed giúp tái lập:

- initialization;
- batch sampling;
- generation sampling.

Seed không đảm bảo toàn bộ hệ thống luôn bit-identical trên mọi phần cứng hoặc phiên bản thư viện, nhưng là điều kiện tối thiểu cho comparison có kiểm soát.

### 12.4 Config tách khỏi implementation

Hyperparameters không nên bị phân tán tùy tiện trong source code.

Config nên ghi nhận:

- model type;
- dtype;
- initialization;
- corpus path;
- split ratios;
- batch size;
- seed;
- learning rate;
- training steps;
- evaluation interval;
- generation settings;
- checkpoint location.

### 12.5 Metrics phải được định nghĩa

Khi báo cáo kết quả, phải nêu rõ:

- metric được tính trên split nào;
- số prediction targets;
- mean hay sum;
- smoothing policy;
- checkpoint nào được đánh giá;
- dùng best-validation hay final-step model.

### 12.6 Checkpoint phải gắn với metadata

Checkpoint nên có hoặc liên kết được với:

- model type;
- parameter shape;
- vocabulary;
- config;
- seed;
- training step;
- best validation loss;
- corpus identity;
- serialization format.

### 12.7 Không thay đổi test để che lỗi

Khi test fail, thứ tự điều tra nên là:

```text
xác minh test contract
→ xác minh dữ liệu
→ xác minh implementation
→ xác minh config
→ chỉ thay đổi expected value khi contract thực sự thay đổi
```

Không được:

- giảm độ chặt của assertion chỉ để pass;
- thay corpus để tránh edge case;
- bỏ test phát hiện regression;
- cập nhật checksum mà không điều tra nguyên nhân thay đổi.

### 12.8 Clean working tree sau verification

Các lệnh inspect và test không nên bí mật sửa source-controlled files.

Sau verification:

```bash
git status --short
```

Working tree nên clean, trừ generated artifacts đã được dự kiến và ignore đúng policy.

---

## 13. Limitations

### Tiny corpus

Corpus hiện tại chỉ có:

```text
1677 character tokens
47 vocabulary characters
```

Kích thước này phù hợp để:

- test implementation;
- minh họa thuật toán;
- chạy thí nghiệm nhanh;
- kiểm tra overfitting;
- so sánh baseline nhỏ.

Kích thước này không phù hợp để:

- học kiến thức thế giới;
- sinh văn bản tổng quát;
- đánh giá năng lực ngôn ngữ thực tế;
- kết luận về khả năng scale;
- so sánh với foundation model lớn.

### Character-level tokenizer

Character tokenizer dễ hiểu và kiểm tra nhưng có hạn chế:

- sequence dài;
- không nắm trực tiếp cấu trúc từ;
- không xử lý unseen characters linh hoạt;
- kém hiệu quả hơn byte-level hoặc subword tokenization trong nhiều workload thực tế.

BPE, unigram subword hoặc byte-level tokenization có thể được nghiên cứu sau khi character tokenizer đã được hiểu chắc chắn.

### Bigram context

Count bigram và neural bigram chỉ dùng một token trước đó.

Chúng không thể học tốt:

- quan hệ dài hạn;
- cấu trúc câu phức tạp;
- phụ thuộc xa;
- chủ đề;
- semantic context sâu.

Các model này là learning baselines, không phải final architecture.

### Chưa có Transformer hoàn chỉnh

Repository hiện chưa có:

- causal self-attention hoàn chỉnh;
- multi-head attention;
- Transformer block hoàn chỉnh;
- stack decoder-only hoàn chỉnh;
- end-to-end MiniGPT training.

Các thành phần này thuộc roadmap.

### Metric trên tiny corpus

Loss hoặc perplexity thấp trên tiny corpus có thể chỉ cho thấy:

- mô hình nhớ dữ liệu;
- corpus có pattern đơn giản;
- model overfit;
- split quá nhỏ để đại diện.

Metric không nên được diễn giải thành chất lượng ngôn ngữ thực tế.

### Generated text

Văn bản sinh ra từ các baseline hiện tại chủ yếu dùng để:

- xác minh generation pipeline;
- quan sát distribution đã học;
- so sánh sampling strategy;
- phát hiện model collapse hoặc lỗi mapping.

Output không được kỳ vọng có chất lượng như một LLM production.

### Hardware và scalability

Scope hiện tại ưu tiên chạy cục bộ và mô hình nhỏ.

Dự án chưa chứng minh:

- GPU acceleration;
- multi-GPU training;
- distributed data parallelism;
- large-dataset streaming;
- high-throughput inference;
- fault-tolerant serving.

### Production assistant chưa tồn tại

Các chức năng như:

- deep research;
- web search;
- workflow automation;
- scheduling;
- application integration;
- personalized long-term memory;
- autonomous agents;

là tầm nhìn dài hạn, không phải chức năng hiện có của repository.

---

## 14. Contributing và next steps

### Nguyên tắc đóng góp

Mỗi pull request nên:

1. giải quyết một issue hoặc một scope rõ ràng;
2. tránh trộn documentation, refactor và feature lớn không liên quan;
3. bổ sung hoặc cập nhật test khi behavior thay đổi;
4. giữ deterministic behavior nếu có thể;
5. cập nhật documentation khi public workflow thay đổi;
6. ghi command verification;
7. không đưa generated artifact vào Git nếu chưa có policy;
8. không overclaim trạng thái của project;
9. giải thích ảnh hưởng tới các phase trước;
10. pass required CI trước khi merge.

### Quy trình đề xuất

```text
chọn issue
→ đọc acceptance criteria
→ xác định invariant
→ tạo branch
→ chạy baseline tests
→ triển khai thay đổi nhỏ nhất hợp lý
→ thêm hoặc sửa tests
→ chạy relevant scripts
→ cập nhật documentation
→ chạy full test suite
→ kiểm tra clean working tree
→ mở pull request
→ review evidence
→ merge khi gate đạt
```

### Branch naming

Ví dụ:

```text
docs/issue-11-readme-learning-map
feat/issue-20-block-context
test/issue-13-tokenizer-edge-cases
fix/issue-19-checkpoint-metadata
```

### Commit naming

Ví dụ:

```text
docs: expand README learning map
feat: add block context batches
test: cover unseen tokenizer characters
fix: preserve checkpoint metadata
```

### Pull request verification

PR description nên chứa:

```markdown
## Summary

Mô tả thay đổi và lý do.

## Scope

Nêu rõ phần nằm trong và ngoài PR.

## Verification

- [ ] `julia --project=. -e "using Pkg; Pkg.test()"`
- [ ] relevant inspection or training script
- [ ] documentation preview checked
- [ ] clean working tree confirmed

## Evidence

Ghi output, metric, screenshot hoặc report cần thiết.

## Limitations

Nêu những phần chưa giải quyết.

Closes #ISSUE_NUMBER
```

### Next steps ưu tiên

Theo learning roadmap, các bước tiếp theo nên tập trung vào:

1. hoàn thiện documentation và edge-case tests cho tokenizer/data pipeline;
2. chuẩn hóa count-model report;
3. chuẩn hóa neural-bigram checkpoint và experiment evidence;
4. xây dựng block-size data pipeline;
5. triển khai embedding + MLP context model;
6. thêm training curves và experiment logging;
7. triển khai causal self-attention từ đầu;
8. xây dựng Transformer block;
9. ghép decoder-only MiniGPT;
10. chạy comparison và ablation studies.

### Nguyên tắc cuối cùng

Dự án ưu tiên:

```text
understand before scaling
test before trusting
measure before claiming
record before comparing
```

Mục tiêu không chỉ là tạo ra một mô hình có thể chạy.

Mục tiêu là hiểu rõ:

- dữ liệu đi vào mô hình như thế nào;
- mô hình học điều gì;
- tại sao loss thay đổi;
- mỗi kiến trúc thêm khả năng gì;
- lỗi có thể xuất hiện ở đâu;
- kết quả có tái lập được hay không;
- giới hạn của mọi kết luận là gì.
