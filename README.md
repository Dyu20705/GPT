# MiniGPT

> Tự học Deep Learning và NLP từ những thành phần cơ bản.

## Development corpus

MiniGPT sử dụng `data/raw/tiny_corpus.txt` làm **deterministic test fixture** cho giai đoạn nền tảng. Corpus này được dùng để kiểm tra:

- character tokenizer và encode/decode round trip;
- phép chia train/validation/test;
- batch generation có thể tái lập;
- count-based và neural bigram baselines;
- các script kiểm tra dữ liệu cục bộ.

### Reproducibility contract

Corpus do dự án tự xây dựng, được version-control trực tiếp và có sẵn trong mọi clean clone. Dự án không yêu cầu tải dữ liệu từ nguồn ngoài hoặc sao chép file ẩn từ máy phát triển.

| Thuộc tính | Giá trị chuẩn |
| --- | --- |
| Path | `data/raw/tiny_corpus.txt` |
| Encoding | UTF-8 |
| Line endings | LF |
| File size | 2156 bytes |
| Character tokens | 1677 |
| Vocabulary size | 47 |
| SHA-256 | `11c4625ab69b40e072e5a0b5a26084f52dc410821b4cdb58be326b376502f9b4` |

### Verify the fixture

Từ thư mục gốc của repository, chạy:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. scripts/inspect_corpus.jl
julia --project=. -e "using Pkg; Pkg.test()"
```

`inspect_corpus.jl` phải báo đúng SHA-256 và các thống kê chuẩn trong bảng trên. Test suite cũng kiểm tra checksum để phát hiện mọi thay đổi ngoài ý muốn ở mức byte.

### Change policy

Mọi thay đổi đối với corpus được xem là thay đổi test fixture. Pull request tương ứng phải:

1. giải thích lý do thay đổi dữ liệu;
2. cập nhật SHA-256 và các thống kê liên quan;
3. chạy lại tokenizer, split, batch và baseline tests;
4. không chỉnh corpus chỉ để che một lỗi triển khai hoặc làm test pass.
