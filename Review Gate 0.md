# Review Gate 0 — MiniGPT Foundation Audit

Status: **Blocked: fix foundation first**  
Priority: **P0**  
Scope: **Pre-planning review gate before creating a larger MVP/backlog/roadmap**  
Review date: **2026-07-07**  
Repository: `Dyu20705/GPT`

---

## 0. Executive conclusion

This review gate is completed as a **static repository audit plus evidence gap report**.

The current codebase already has a coherent MiniGPT learning foundation:

- Julia package entry point exists.
- Character tokenizer exists.
- Token-level split exists.
- Next-token batch generation exists.
- Count unigram and bigram models exist.
- Stable cross-entropy / perplexity utilities exist.
- Neural bigram model exists.
- Basic SGD training loop exists.
- Checkpoint save/load exists.
- Unit tests exist for tokenizer, split, batch, count models, neural bigram, training, generation, and checkpointing.
- Experiment notes exist for count models and neural bigram.

However, the gate should **not** be marked `Ready for next planning phase` yet.

Main blocker:

- `data/raw/tiny_corpus.txt` is required by tests and scripts, but `data/raw/*` is ignored by `.gitignore`.
- The review environment could not fetch `data/raw/tiny_corpus.txt` from the repository.
- Therefore a clean clone may not be able to run `Pkg.test()` or `scripts/inspect_corpus.jl` unless the corpus file is manually present on the local machine.

Because this issue is specifically a **review gate requiring fresh run evidence**, the correct conclusion is:

```text
Conclusion:
Blocked: fix foundation first
```

The foundation is conceptually strong, but the repo needs reproducible corpus provisioning and fresh command evidence before creating a large backlog for Transformer/GPT work.

---

## 1. Repository review

### 1.1 Observed repository structure

Observed files and directories from repository review:

```text
.
├── README.md
├── Project.toml
├── Manifest.toml
├── .gitignore
├── src/
│   ├── MiniGPT.jl
│   ├── tokenize.jl
│   ├── data.jl
│   ├── batch.jl
│   ├── count_models.jl
│   ├── objectives.jl
│   ├── neural_bigram.jl
│   ├── training.jl
│   └── checkpoint.jl
├── test/
│   ├── runtests.jl
│   ├── test_count_models.jl
│   └── test_neural_bigram.jl
├── scripts/
│   ├── inspect_corpus.jl
│   ├── inspect_count_models.jl
│   └── demo_batch.jl
├── configs/
│   └── neural_bigram.toml
├── experiments/
│   ├── phase1_count_models.md
│   └── phase2_neural_bigram.md
├── data/
│   ├── raw/
│   └── processed/
└── checkpoints/
```

Important note:

```text
data/raw/tiny_corpus.txt
```

is referenced by tests and scripts, but it appears to be ignored by `.gitignore` and was not fetchable during this review.

### 1.2 Entry point

Main package module:

```julia
module MiniGPT
```

Entry file:

```text
src/MiniGPT.jl
```

The package includes and exports these core modules:

```julia
include("tokenize.jl")
include("data.jl")
include("batch.jl")
include("count_models.jl")
include("objectives.jl")
include("neural_bigram.jl")
include("training.jl")
include("checkpoint.jl")
```

Public exports include:

```julia
CharacterTokenizer
TokenSplits
encode
decode
vocabulary
split_token_ids
next_token_example
get_batch
UnigramCountModel
BigramCountModel
fit_unigram
fit_bigram
evaluate
generate_ids
most_common_bigrams
NeuralBigramModel
NeuralLMReport
NeuralBigramTrainingState
logsumexp
cross_entropy
mean_cross_entropy
perplexity_from_loss
neural_bigram_gradient
train_step!
train!
save_checkpoint
load_checkpoint
vocab_size
```

Assessment:

- Package entry point is clean.
- Module boundary is understandable.
- The project is already separated into educational phases: tokenizer/data/batch → count models → neural bigram → training/checkpoint.

Status: **Done**

---

## 2. Dependency review

### 2.1 Package metadata

`Project.toml` declares:

```toml
name = "MiniGPT"
version = "0.1.0"
julia = "1.12"
```

`Manifest.toml` was generated with:

```toml
julia_version = "1.12.5"
```

This is consistent with the project direction, but it means contributors need Julia 1.12.x.

### 2.2 Dependency roles

Current dependency intent:

| Dependency | Likely role |
|---|---|
| `Flux` | Future neural-network abstraction / DL stack |
| `NNlib` | Neural network primitives |
| `Optimisers` | Future optimizer support |
| `Zygote` | Future automatic differentiation |
| `JLD2` | Intended checkpoint/storage dependency |
| `BenchmarkTools` | Benchmarking experiments |
| `DataFrames` | Experiment result tables |
| `Plots` | Visualization |
| `HTTP`, `Gumbo`, `Cascadia`, `JSON3` | Likely future dataset/web scraping/data ingestion tools |
| `ProgressMeter` | Training progress display |
| `Revise`, `Pluto` | Development / notebook workflow |
| `Random`, `LinearAlgebra`, `Serialization` | Used by current implementation |
| `Test`, `Unicode` | Test target/extras |

Dependency concern:

- `JLD2` is declared, but current checkpoint implementation uses Julia `Serialization`, not `JLD2`.
- `Flux`, `Optimisers`, and `Zygote` are declared, but the current neural bigram implementation uses a manual analytic gradient and manual SGD.
- This is acceptable for learning, but the README or experiment notes should explicitly explain which dependencies are current vs future-facing.

Status: **Partially Done**

---

## 3. Test suite review

### 3.1 Existing test coverage

`test/runtests.jl` covers:

- Package smoke check: `MiniGPT.hello()`.
- Tiny corpus fixture expected size/vocabulary.
- Unicode NFC normalization.
- No BOM.
- No CR characters.
- Train/validation/test split sizes.
- Train vocabulary coverage.
- CharacterTokenizer vocabulary size.
- Encode/decode round-trip.
- Token data pipeline.
- Next-token example shift invariant.
- Batch shape.
- Batch shift invariant.
- Seed reproducibility.
- Input validation errors.
- Includes count model tests.
- Includes neural bigram tests.

`test/test_count_models.jl` covers:

- Unigram counts.
- Additive smoothing.
- Bigram transition counts.
- Bigram row normalization.
- Backoff for zero outgoing transitions.
- Evaluation correctness.
- Infinite loss/perplexity for unseen events without smoothing.
- Reproducible generation.
- Most common bigrams.
- Input validation.

`test/test_neural_bigram.jl` covers:

- Stable `logsumexp`.
- Cross-entropy.
- Perplexity.
- NeuralBigramModel shape and seed reproducibility.
- Forward logits shape.
- Analytic gradient.
- SGD update.
- Finite-difference gradient check.
- Toy overfit.
- Greedy/sample generation.
- Checkpoint save/load.
- Training loop state.

Assessment:

- Test coverage is strong for a learning MVP.
- It covers both theory invariants and practical failure modes.
- The main risk is not test design; it is reproducibility if the corpus fixture is not committed/provisioned.

Status: **Partially Done**

---

## 4. Script / real-run review

### 4.1 `scripts/inspect_corpus.jl`

This script checks:

- Corpus path exists.
- Corpus is non-empty.
- Corpus can be opened.
- UTF-8 readability.
- Unicode NFC.
- BOM / replacement character / control character warnings.
- Character token count.
- Vocabulary size.
- Train/validation/test split counts.
- Token ID validity.
- Full vocabulary coverage.
- Encode/decode round-trip.
- SHA-256.
- Frequent/least frequent characters.
- Validation/test-only characters.
- Warnings.

This script is exactly the right kind of real-run evidence for Phase 0.

Status: **Done as implementation, Needs Investigation as fresh run evidence**

### 4.2 `scripts/inspect_count_models.jl`

This script checks:

- Corpus loading.
- Tokenization.
- Split.
- Unigram/bigram fitting.
- Count/probability invariants.
- Evaluation table across train/validation/test.
- Most common bigrams.
- Reproducible generation.
- Alpha smoothing sweep.

This is useful evidence for Phase 1 count models.

Status: **Done as implementation, Needs fresh run evidence**

### 4.3 `scripts/demo_batch.jl`

This script checks:

- Corpus loading.
- Tokenization.
- Split.
- Batch shape.
- Shift invariant.
- Same-seed reproducibility.
- Decoded examples.

This is useful for manually understanding next-token training data.

Status: **Done as implementation, Needs fresh run evidence**

---

## 5. Theory review

### 5.1 Character tokenizer

The character tokenizer maps every unique character in the training text to an integer ID.

In this project:

```julia
chars = sort!(collect(Set(text)))
stoi = Dict(ch => id for (id, ch) in enumerate(chars))
itos = Dict(id => ch for (id, ch) in enumerate(chars))
```

The learning purpose:

- It avoids complex subword tokenization.
- It makes the language modeling problem small enough to inspect by hand.
- It forces a clear understanding of vocabulary, token IDs, sequence modeling, and next-token prediction before moving to BPE/WordPiece or GPT tokenizers.

Important limitation:

- This tokenizer only knows characters seen during construction.
- Encoding text with unseen characters throws an error.
- That is acceptable for a tiny educational corpus, but future dataset work needs an explicit OOV policy or a tokenizer training protocol.

Status: **Done**

### 5.2 Round-trip invariant: `decode(encode(text)) == text`

This invariant means the tokenizer is lossless on the vocabulary it was built from.

Why it matters:

- If encode/decode changes text, the model is not learning the real corpus.
- It catches Unicode normalization issues.
- It catches missing characters.
- It catches broken `stoi` / `itos` mappings.
- It gives confidence that generated token IDs can be converted back to readable text.

Current implementation supports this invariant for known characters.

Status: **Done**

### 5.3 Train/validation/test split at token level

The split function divides token IDs sequentially:

```text
train = first 80%
validation = next 10%
test = remainder
```

Theory:

- Train is used to fit parameters.
- Validation is used for model selection and diagnostics.
- Test is reserved for final evaluation.
- Keeping the sequence order is appropriate for language modeling because token order is meaningful.

Risk:

- For tiny corpora, sequential split can create distribution artifacts.
- The current script checks validation/test-only characters to reduce this risk.

Status: **Done**

### 5.4 Batch next-token prediction

For language modeling, the input sequence is shifted by one token to create the target sequence.

Example:

```text
x = ids[start : start + context_length - 1]
y = ids[start + 1 : start + context_length]
```

Invariant:

```text
x[2:end] == y[1:end-1]
```

Meaning:

- For every token at time `t`, the model learns to predict token `t + 1`.
- This is the same basic objective used before scaling to Transformer/GPT.
- The current neural bigram uses `context_length = 1`, but the batch abstraction already prepares the mental model for longer contexts.

Status: **Done**

### 5.5 Unigram count model

Unigram assumes every token is independent:

```text
P(x1, x2, ..., xT) = Π P(xt)
```

Estimated probability:

```text
P(token) = (count(token) + alpha) / (N + alpha * V)
```

where:

- `N` = number of training tokens.
- `V` = vocabulary size.
- `alpha` = additive smoothing.

Learning purpose:

- Establishes the simplest frequency baseline.
- Shows what a model can learn from global token frequency alone.
- Provides a baseline that bigram and neural models should beat.

Status: **Done**

### 5.6 Bigram count model

Bigram assumes the next token depends only on the previous token:

```text
P(x1, x2, ..., xT) = P(x1) * Π P(xt | xt-1)
```

Estimated probability:

```text
P(next | previous) =
(count(previous -> next) + alpha) /
(count(previous outgoing transitions) + alpha * V)
```

Learning purpose:

- Introduces conditional probability.
- Shows how local context improves next-token prediction.
- Makes the transition from frequency counting to sequence modeling clear.

Important implementation detail:

- Count bigram stores probabilities as `probs[previous, next]`.
- Neural bigram stores logits as `logits_table[next, current]`.

Status: **Done**

### 5.7 Neural bigram model

The neural bigram model replaces count-derived probabilities with learnable logits:

```text
logits_table[next, current]
```

Then:

```text
P(next | current) = softmax(logits_table[:, current])
```

Difference from count bigram:

| Count bigram | Neural bigram |
|---|---|
| Directly estimates probabilities from counts | Learns logits through gradient descent |
| Uses explicit smoothing/backoff | Softmax gives every next token positive probability |
| Usually strong on tiny countable corpora | Teaches differentiable modeling and optimization |
| Not trained by loss minimization | Trained with cross-entropy + SGD |

Learning purpose:

- Introduces logits.
- Introduces softmax.
- Introduces differentiable loss.
- Introduces gradients.
- Introduces optimization.
- Introduces training state/checkpointing.
- Prepares the conceptual path to embeddings, MLPs, attention, and Transformer blocks.

Status: **Done for Phase 2 educational scope**

### 5.8 Cross-entropy and perplexity

Cross-entropy measures how surprised the model is by the correct next token.

For one prediction:

```text
loss = -log P(correct_next_token)
```

The implementation computes this from logits using stable log-sum-exp:

```text
cross_entropy(logits, target) = logsumexp(logits) - logits[target]
```

Mean cross-entropy is the average loss over many next-token predictions.

Perplexity is:

```text
perplexity = exp(mean_cross_entropy)
```

Interpretation:

- Lower cross-entropy is better.
- Lower perplexity is better.
- Perplexity can be read as the model’s effective average branching uncertainty.
- A perplexity of 1 means the model is perfectly confident and correct.
- Infinite perplexity means the model assigned zero probability to an observed target.

Status: **Done**

### 5.9 Training state and checkpoint

Training state tracks:

- `step`
- `tokens_seen`
- `best_validation_loss`
- `best_step`
- `best_logits_table`
- `seed`
- `history`

Checkpoint stores:

- version
- model type
- logits orientation
- logits table
- vocab size
- step
- optimizer state
- config
- best validation loss
- seed
- vocabulary

Learning purpose:

- Separates model parameters from training metadata.
- Preserves reproducibility information.
- Allows saving and reloading trained model state.
- Prepares for future larger training loops.

Concern:

- Current checkpoint uses Julia `Serialization`.
- `JLD2` is declared but not used here.
- For long-term reproducibility, choose and document one checkpoint format.

Status: **Partially Done**

---

## 6. Evidence review

### 6.1 Required command: package test

Required command:

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

Review result:

```text
Result:
NOT INDEPENDENTLY RUN IN THIS REVIEW ENVIRONMENT
```

Reason:

- ChatGPT review environment could inspect repository files through the GitHub connector.
- The environment could not clone GitHub directly.
- The environment did not provide a verified Julia 1.12 local runtime for running this repo.
- Existing experiment documentation says `Pkg.test()` passed Phase 0, Phase 1, and Phase 2 tests, but this review did not independently reproduce that result.

Required next action:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.test()"
```

Paste fresh output into the issue before closing the gate.

Status: **Needs Investigation**

### 6.2 Required command: corpus inspection

Required command:

```bash
julia --project=. scripts/inspect_corpus.jl
```

Review result:

```text
Result:
NOT INDEPENDENTLY RUN IN THIS REVIEW ENVIRONMENT
```

Important finding:

- The script expects `data/raw/tiny_corpus.txt`.
- `.gitignore` ignores `data/raw/*`.
- The corpus file was not fetchable from the repository during review.
- Therefore a clean clone may fail here unless the file is manually created or provisioned.

Required next action:

```bash
julia --project=. scripts/inspect_corpus.jl
```

Expected output must include:

- Corpus path.
- File size.
- SHA-256.
- UTF-8 readable.
- Unicode NFC.
- Line count.
- Token count.
- Vocabulary size.
- Train/validation/test token counts.
- Round-trip correct.
- Warnings or `Warnings: none`.

Status: **Needs Fix / Needs Investigation**

### 6.3 Manual smoke evidence

Required smoke command:

```julia
using MiniGPT

text = read("data/raw/tiny_corpus.txt", String)
tok = CharacterTokenizer(text)
ids = encode(tok, text)

@assert decode(tok, ids) == text
@show vocab_size(tok)
@show length(ids)
```

Review result:

```text
Result:
NOT INDEPENDENTLY RUN IN THIS REVIEW ENVIRONMENT
```

Additional concern:

- `vocab_size(tok)` may not be implemented for `CharacterTokenizer`.
- `vocab_size(model::NeuralBigramModel)` exists.
- `vocab_size` is exported, but this review did not find a `vocab_size(tokenizer::CharacterTokenizer)` method.
- If this smoke command fails at `vocab_size(tok)`, use `length(vocabulary(tok))` or add a tokenizer-specific `vocab_size` method.

Recommended corrected smoke command:

```julia
using MiniGPT

text = read("data/raw/tiny_corpus.txt", String)
tok = CharacterTokenizer(text)
ids = encode(tok, text)

@assert decode(tok, ids) == text
@show length(vocabulary(tok))
@show length(ids)
```

Status: **Needs Fix / Needs Investigation**

---

## 7. Existing experiment evidence

The repository contains experiment notes with prior results.

### 7.1 Count model experiment

`experiments/phase1_count_models.md` records:

- Corpus path.
- Corpus SHA-256.
- Vocabulary size 47.
- Train tokens 1341.
- Validation tokens 167.
- Test tokens 169.
- Unigram and bigram train/validation/test metrics.
- Smoothing sweep.
- Generated samples.
- Qualitative observations.
- Conclusion that smoothed bigram improves over unigram.

Key recorded metrics:

| Model | Validation PPL | Test PPL |
|---|---:|---:|
| Unigram | 23.750239 | 27.258243 |
| Bigram alpha=0.1 | 4.923512 | 5.759588 |

Assessment:

- Strong evidence that Phase 1 was run before.
- But it references a local absolute path and should be refreshed from clean repo/CI evidence.

Status: **Partially Done**

### 7.2 Neural bigram experiment

`experiments/phase2_neural_bigram.md` records:

- Julia 1.12.5.
- Corpus SHA-256.
- Vocabulary size 47.
- Config file path.
- Parameter matrix `(47, 47)`.
- Parameter count 2209.
- Initial NLL close to `log(47)`.
- SGD settings.
- Best validation NLL.
- Required sanity checks.
- Result table.
- Note that `Pkg.test()` passes Phase 0, Phase 1, and Phase 2.

Key recorded metrics:

| Model | Train NLL | Validation NLL | Test NLL | Test PPL |
|---|---:|---:|---:|---:|
| Unigram count | 3.189685 | 3.167593 | 3.305356 | 27.258243 |
| Bigram count, alpha=0.1 | 1.621591 | 1.594022 | 1.750866 | 5.759588 |
| Neural bigram | 2.623950 | 2.576862 | 2.773992 | 16.022470 |

Assessment:

- Good educational Phase 2 evidence.
- Neural bigram is not currently better than count bigram, but that is acceptable because its purpose is to teach differentiable training.
- The note says `f8d7843 plus current working-tree changes`, so the result should be regenerated from a clean committed state.

Status: **Partially Done**

---

## 8. Completion classification

| Area | Current implementation | Evidence | Status | Notes |
|---|---|---|---|---|
| Package setup | `Project.toml`, `Manifest.toml`, `src/MiniGPT.jl` exist | Static file review | Partially Done | Julia 1.12/1.12.5 pinned; dependency cleanup needed |
| README / docs | README is minimal; experiment docs exist | Static file review | Partially Done | README should explain phases, commands, corpus setup |
| Corpus fixture | Tests/scripts require `data/raw/tiny_corpus.txt` | `.gitignore` ignores `data/raw/*`; file not fetchable in review | Needs Fix | Reproducibility blocker |
| Character tokenizer | `CharacterTokenizer`, `encode`, `decode`, `vocabulary` | Source + tests | Done | Add `vocab_size(::CharacterTokenizer)` if smoke command requires it |
| Data split | `TokenSplits`, `split_token_ids` | Source + tests | Done | Sequential 80/10/remainder split |
| Batch generation | `next_token_example`, `get_batch` | Source + tests + demo script | Done | Shift invariant and seed reproducibility covered |
| Count unigram model | `fit_unigram`, evaluation, generation | Source + tests + phase1 notes | Done | Good frequency baseline |
| Count bigram model | `fit_bigram`, smoothing/backoff, evaluation, generation | Source + tests + phase1 notes | Done | Strong baseline on tiny corpus |
| Objectives / metrics | `logsumexp`, `cross_entropy`, `mean_cross_entropy`, `perplexity_from_loss` | Source + tests | Done | Stable implementation |
| Neural bigram model | Logit table model, softmax, gradient, generation | Source + tests + phase2 notes | Done | Educational objective achieved |
| Training loop | `train!`, `train_step!`, state tracking | Source + tests | Partially Done | Needs CLI/scripted full run evidence |
| Checkpointing | `save_checkpoint`, `load_checkpoint` | Source + tests | Partially Done | Uses `Serialization`; decide whether to use `JLD2` |
| Tests | Unit tests cover main invariants | Static review + experiment note | Partially Done | Fresh `Pkg.test()` evidence still required |
| Scripts / real run | `inspect_corpus`, `inspect_count_models`, `demo_batch` | Static review + experiment notes | Needs Investigation | Fresh script outputs still required |

---

## 9. Blockers and follow-up tasks

### Blocker 1 — Corpus fixture is not reproducible from clean clone

Problem:

```text
data/raw/tiny_corpus.txt
```

is required by tests/scripts, but `.gitignore` ignores `data/raw/*`.

Impact:

- `Pkg.test()` may fail.
- `scripts/inspect_corpus.jl` may fail.
- Manual smoke may fail.
- New contributors cannot reproduce the project from clone.

Recommended fix options:

Option A — Commit the tiny educational fixture:

```gitignore
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/raw/tiny_corpus.txt
!data/processed/.gitkeep
```

Option B — Keep raw data ignored, but add a reproducible setup script:

```text
scripts/setup_tiny_corpus.jl
```

and document:

```bash
julia --project=. scripts/setup_tiny_corpus.jl
```

Recommended for this project:

```text
Use Option A if tiny_corpus.txt is small and non-sensitive.
```

### Blocker 2 — Fresh command evidence missing

Required before closing Review Gate 0:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.test()"
julia --project=. scripts/inspect_corpus.jl
```

Paste output into this file or into the issue comment.

### Blocker 3 — Manual smoke command likely needs adjustment

The issue requested:

```julia
@show vocab_size(tok)
```

But review found `vocab_size` clearly implemented for `NeuralBigramModel`, not for `CharacterTokenizer`.

Fix either by adding:

```julia
vocab_size(tokenizer::CharacterTokenizer) = length(tokenizer.vocabulary)
```

or by changing smoke evidence to:

```julia
@show length(vocabulary(tok))
```

### Follow-up 4 — README should become a real phase map

README should include:

- Project goal.
- Julia version.
- Setup commands.
- Corpus setup.
- Test command.
- Script commands.
- Current phases completed.
- Current blockers.
- Next allowed phase after Review Gate.

### Follow-up 5 — Clarify checkpoint format

Current code uses `Serialization`.

Decide:

- Keep `Serialization` for internal educational checkpoints.
- Or migrate to `JLD2`.
- Or document why `JLD2` is a future dependency.

---

## 10. Acceptance criteria status

| Acceptance criterion | Status | Notes |
|---|---|---|
| Comment review tổng quan hiện trạng repo | Done | This file contains the review |
| Completion classification table | Done | See section 8 |
| Theory review for tokenizer/split/batch/bigram/neural/loss | Done | See section 5 |
| Evidence for `Pkg.test()` | Not Done | Needs fresh local/CI run |
| Evidence for `scripts/inspect_corpus.jl` | Not Done | Blocked until corpus fixture is reproducible |
| Manual smoke evidence | Not Done | Needs local/CI run; command may need `length(vocabulary(tok))` |
| Errors/follow-ups recorded | Done | See section 9 |
| Final conclusion | Done | Blocked: fix foundation first |

---

## 11. Final decision

```text
Conclusion:
Blocked: fix foundation first
```

Do not create a large Transformer/GPT backlog yet.

The correct next step is a small foundation-fix batch:

1. Make `data/raw/tiny_corpus.txt` reproducible from clean clone.
2. Re-run `Pkg.test()` from a clean environment.
3. Re-run `scripts/inspect_corpus.jl`.
4. Run manual smoke.
5. Paste command outputs.
6. Then update conclusion to:

```text
Conclusion:
Ready for next planning phase
```

Only after that should the project create the next MVP backlog for Transformer/GPT blocks.
