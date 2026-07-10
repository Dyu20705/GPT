# MiniGPT

> Tự học DL + NLP từ đầu

## Development corpus: tiny_corpus.txt

Corpus nhỏ, cố định, bước đầu cho:
-> tokenizer
-> deterministic split
-> batch generation
-> bigram/baseline tests
-> local inspection

Nội dung corpus tự tạo (1 dòng), sau đó copy paste XD

Corpus được đẩy trực tiếp lên repo, có sẵn khi clone

Vị trí: data/raw/tiny_corpus.txt

Xác minh: julia --project=. scripts/inspect_corpus.jl
Hash chuẩn: 11c4625ab69b40e072e5a0b5a26084f52dc410821b4cdb58be326b376502f9b4

Không thay đổi tùy ý corpus:
-> tương ứng với thay đổi test fixture
-> update hash
-> giải thích lý do
-> kiểm tra lại tokenization, splits, batches và baseline expectations
-> không được chỉnh chỉ để làm test pass

