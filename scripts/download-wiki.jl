using Dates
using HTTP
using JSON3
using SHA

const DUMP_BASE_URL = "https://dumps.wikimedia.org/viwiki/latest"
const INDEX_URL = "$DUMP_BASE_URL/"
const SHA1SUMS_FILE = "viwiki-latest-sha1sums.txt"
const DEFAULT_OUTPUT_DIR = joinpath("data", "raw", "viwiki")

struct DumpFile
    name::String
    bytes::Int
    modified::String
end

function usage()
    println("""
    Download Vietnamese Wikipedia article dumps from Wikimedia.

    Usage:
      julia --project=. scripts/download-wiki.jl [options]

    Options:
      --target=shards       Download all pages-articles shard files. This is the default.
      --target=full         Download viwiki-latest-pages-articles.xml.bz2 as one file.
      --target=multistream  Download viwiki-latest-pages-articles-multistream.xml.bz2.
      --output-dir=PATH     Where dump files are stored. Default: $DEFAULT_OUTPUT_DIR
      --max-files=N         Limit number of files; useful for smoke tests.
      --smallest-first      Sort selected files by size before applying --max-files.
      --dry-run             Print selected files and verify dump index without downloading.
      --force               Re-download files even if they already exist.
      --help                Show this help.
    """)
end

function parse_args(args)
    options = Dict{String, Any}(
        "target" => "shards",
        "output_dir" => DEFAULT_OUTPUT_DIR,
        "max_files" => nothing,
        "smallest_first" => false,
        "dry_run" => false,
        "force" => false,
    )

    for arg in args
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif arg == "--dry-run"
            options["dry_run"] = true
        elseif arg == "--force"
            options["force"] = true
        elseif arg == "--smallest-first"
            options["smallest_first"] = true
        elseif startswith(arg, "--target=")
            options["target"] = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--output-dir=")
            options["output_dir"] = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--max-files=")
            options["max_files"] = parse(Int, split(arg, "=", limit = 2)[2])
        else
            throw(ArgumentError("unknown option: $arg"))
        end
    end

    target = options["target"]
    target in ("shards", "full", "multistream") ||
        throw(ArgumentError("target must be one of: shards, full, multistream"))

    return options
end

function fetch_text(url)
    response = HTTP.get(url; readtimeout = 60, retries = 3, status_exception = false)

    response.status == 200 ||
        throw(ErrorException("GET $url failed with HTTP status $(response.status)"))

    return String(response.body)
end

function parse_dump_index(html)
    files = DumpFile[]
    pattern = r"<a href=\"([^\"]+\.bz2)\">.*?</a>\s+([0-9]{2}-[A-Za-z]{3}-[0-9]{4}\s+[0-9:]+)\s+([0-9]+)"

    for match in eachmatch(pattern, html)
        name = String(match.captures[1])
        modified = String(match.captures[2])
        bytes = parse(Int, match.captures[3])
        push!(files, DumpFile(name, bytes, modified))
    end

    isempty(files) && throw(ErrorException("no .bz2 dump files found in $INDEX_URL"))
    return files
end

function select_files(files, target)
    if target == "full"
        selected = filter(file -> file.name == "viwiki-latest-pages-articles.xml.bz2", files)
    elseif target == "multistream"
        selected = filter(file -> file.name == "viwiki-latest-pages-articles-multistream.xml.bz2", files)
    else
        selected = filter(files) do file
            occursin(r"^viwiki-latest-pages-articles[0-9]+\.xml-p[0-9]+p[0-9]+\.bz2$", file.name)
        end
        sort!(selected, by = file -> file.name)
    end

    isempty(selected) && throw(ErrorException("no files selected for target=$target"))
    return selected
end

function parse_sha1sums(text)
    hashes = Dict{String, String}()

    for line in split(text, '\n')
        parts = split(strip(line))
        length(parts) == 2 || continue
        hash = lowercase(parts[1])
        name = parts[2]
        hashes[name] = hash

        latest_alias = replace(name, r"^viwiki-[0-9]{8}-" => "viwiki-latest-")
        hashes[latest_alias] = hash
    end

    return hashes
end

function human_size(bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    value = Float64(bytes)
    unit = 1

    while value >= 1024 && unit < length(units)
        value /= 1024
        unit += 1
    end

    return "$(round(value, digits = unit == 1 ? 0 : 2)) $(units[unit])"
end

function file_sha1(path)
    open(path, "r") do io
        return bytes2hex(sha1(io))
    end
end

function verify_sha1(path, expected)
    actual = file_sha1(path)

    actual == lowercase(expected) ||
        throw(ErrorException("SHA1 mismatch for $path: expected $expected, got $actual"))

    return true
end

function download_file(file, output_dir; force = false)
    mkpath(output_dir)

    url = "$DUMP_BASE_URL/$(file.name)"
    dest = joinpath(output_dir, file.name)
    part = "$dest.part"

    if isfile(dest) && !force
        existing_size = filesize(dest)

        if existing_size == file.bytes
            println("Already exists: $(file.name) ($(human_size(existing_size)))")
            return dest
        end

        println("Existing file has unexpected size; re-downloading: $(file.name)")
    end

    isfile(part) && rm(part; force = true)

    println("Downloading: $(file.name) ($(human_size(file.bytes)))")
    HTTP.download(url, part; readtimeout = 300, retries = 5)

    downloaded_size = filesize(part)
    downloaded_size == file.bytes ||
        throw(ErrorException("size mismatch for $(file.name): expected $(file.bytes), got $downloaded_size"))

    mv(part, dest; force = true)
    return dest
end

function write_metadata(output_dir, selected, target)
    metadata = Dict(
        "name" => "Vietnamese Wikipedia articles dump",
        "language" => "vi",
        "source" => "Wikimedia Dumps",
        "source_url" => INDEX_URL,
        "target" => target,
        "license" => "CC BY-SA 4.0 / GFDL, per Wikimedia project licensing",
        "retrieved_at" => string(Dates.now()),
        "files" => [
            Dict(
                "name" => file.name,
                "bytes" => file.bytes,
                "modified" => file.modified,
                "url" => "$DUMP_BASE_URL/$(file.name)",
            )
            for file in selected
        ],
        "processing" => [
            "downloaded official pages-articles XML dump",
            "no random API sampling",
            "no HTML parsing",
            "SHA1 verified when files are downloaded",
        ],
    )

    mkpath(output_dir)
    open(joinpath(output_dir, "metadata.json"), "w") do io
        JSON3.pretty(io, metadata)
    end
end

function main(args = ARGS)
    options = parse_args(args)
    target = options["target"]
    output_dir = options["output_dir"]
    max_files = options["max_files"]

    println("Fetching dump index: $INDEX_URL")
    files = parse_dump_index(fetch_text(INDEX_URL))
    selected = select_files(files, target)

    if max_files !== nothing
        options["smallest_first"] && sort!(selected, by = file -> file.bytes)
        max_files >= 1 || throw(ArgumentError("max-files must be at least 1"))
        selected = selected[1:min(max_files, length(selected))]
    end

    total_bytes = sum(file.bytes for file in selected)
    println("Selected target: $target")
    println("Selected files: $(length(selected))")
    println("Total compressed size: $(human_size(total_bytes))")

    for file in selected
        println("- $(file.name)  $(human_size(file.bytes))  $(file.modified)")
    end

    write_metadata(output_dir, selected, target)

    if options["dry_run"]
        println("Dry run only; no dump files downloaded.")
        println("Metadata written: $(joinpath(output_dir, "metadata.json"))")
        return nothing
    end

    sha1sums = parse_sha1sums(fetch_text("$DUMP_BASE_URL/$SHA1SUMS_FILE"))

    for file in selected
        expected_sha1 = get(sha1sums, file.name, nothing)
        expected_sha1 === nothing &&
            throw(ErrorException("missing SHA1 entry for $(file.name) in $SHA1SUMS_FILE"))

        path = download_file(file, output_dir; force = options["force"])
        verify_sha1(path, expected_sha1)
        println("Verified SHA1: $(file.name)")
    end

    println("Done. Dumps are in: $output_dir")
end

main()
