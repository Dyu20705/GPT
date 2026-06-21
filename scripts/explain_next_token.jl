using MiniGPT

const TEXT = "hello"
const CONTEXT_LENGTH = 4

function main()
    tokenizer = CharacterTokenizer(TEXT)
    ids = encode(tokenizer, TEXT)
    x, y = next_token_example(ids, 1, CONTEXT_LENGTH)

    println("Text: $(repr(TEXT))")
    println("Token IDs: $ids")
    println()
    println("For context length = $CONTEXT_LENGTH:")
    println("x IDs: $x")
    println("y IDs: $y")
    println("x text: $(repr(decode(tokenizer, x)))")
    println("y text: $(repr(decode(tokenizer, y)))")
    println()
    println("The same window contains these next-token tasks:")

    for i in 1:CONTEXT_LENGTH
        context = decode(tokenizer, x[1:i])
        target = decode(tokenizer, y[i:i])
        println("$(repr(context)) -> $(repr(target))")
    end
end

main()
