using MiniGPT

const CORPUS_RELATIVE_PATH = joinpath("data", "raw", "tiny_corpus.txt")
const CONTEXT_LENGTH = 16
const BATCH_SIZE = 4
const SEED = 42

function project_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function corpus_path()
    return abspath(joinpath(project_root(), CORPUS_RELATIVE_PATH))
end

function main()
    text = read(corpus_path(), String)
    tokenizer = CharacterTokenizer(text)
    ids = encode(tokenizer, text)
    splits = split_token_ids(ids)

    x, y = get_batch(
        splits.train;
        context_length = CONTEXT_LENGTH,
        batch_size = BATCH_SIZE,
        seed = SEED,
    )

    x_again, y_again = get_batch(
        splits.train;
        context_length = CONTEXT_LENGTH,
        batch_size = BATCH_SIZE,
        seed = SEED,
    )

    shift_invariant = x[2:end, :] == y[1:end-1, :]
    reproducible = x == x_again && y == y_again

    @assert size(x) == (CONTEXT_LENGTH, BATCH_SIZE)
    @assert size(y) == (CONTEXT_LENGTH, BATCH_SIZE)
    @assert shift_invariant
    @assert reproducible

    println("Corpus path: $(corpus_path())")
    println("Train tokens: $(length(splits.train))")
    println("Context length: $CONTEXT_LENGTH")
    println("Batch size: $BATCH_SIZE")
    println("shape(x): $(size(x))")
    println("shape(y): $(size(y))")
    println("Shift invariant: $shift_invariant")
    println("Same seed reproducible: $reproducible")
    println()

    for column in axes(x, 2)
        println("Column $column")
        println("x IDs: $(x[:, column])")
        println("y IDs: $(y[:, column])")
        println("x text: $(repr(decode(tokenizer, x[:, column])))")
        println("y text: $(repr(decode(tokenizer, y[:, column])))")
        println()
    end
end

main()
