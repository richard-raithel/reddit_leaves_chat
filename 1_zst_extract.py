import zstandard as zstd

input_file = "../compressed_data/leaves_comments.zst"
output_file = "../data/leaves_comments.json"

with open(input_file, "rb") as compressed, open(output_file, "wb") as decompressed:
    dctx = zstd.ZstdDecompressor()
    dctx.copy_stream(compressed, decompressed)

print("Decompression complete!")
