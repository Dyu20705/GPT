function validate_absolute_path(path)
    if !isfile(path)
        println("Error: File does not exist at path: $path")
        return false
    end
    return true
end

function file_exists(path)
    return isfile(path) & !isdir(path) & isreadable(path) & filesize(path) > 0
end

function readOkUTF8(path)
    if !isfile(path)
        return
    end

    try
        open(path) do f
            for line in eachline(f)
                # Just read the lines to check for UTF-8 validity
            end
        end
        return true
    catch e
        if isa(e, UnicodeError)
            return false
        else
            rethrow(e)
        end
    end
    
end

function exact_numb_token_character_level(path)
    if !isfile(path)
        return
    end

    num_tokens = 0
    num_characters = 0

    open(path) do f
        for line in eachline(f)
            num_tokens += length(split(line))
            num_characters += length(line)
        end
    end

    return num_tokens, num_characters
end

function to_string(path)
    if !isfile(path)
        return
    end

    content = ""
    open(path) do f
        content = read(f, String)
    end
    return content
end

function contentToCharacterTokenArray(content)
    characterTokens = [c for c in content]
    return characterTokens
end

function toVocabulary(characterTokens)
    return list(sort(unique(characterTokens)))
end

function split_train_valid_test(contentToCharacterTokenArray)
    total_length = length(contentToCharacterTokenArray)
    train_end = Int(floor(0.8 * total_length))
    valid_end = Int(floor(0.9 * total_length))

    train_set = contentToCharacterTokenArray[1:train_end]
    valid_set = contentToCharacterTokenArray[train_end+1:valid_end]
    test_set = contentToCharacterTokenArray[valid_end+1:end]

    return train_set, valid_set, test_set
    
end

function encode(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    return encode
end

function decode(chars)
    itos = { i:ch for i,ch in enumerate(chars) }
    decode = lambda l: [itos[i] for i in l] # decoder: take a list of integers, output a string
    return decode
end

function valid_round_trip(corpus)
    e = encode(corpus)
    d = decode(e)
    return corpus == d
end
#exe script


