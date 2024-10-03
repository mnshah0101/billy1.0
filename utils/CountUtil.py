import tiktoken



def count_tokens(string: str) -> int:
    encoding = tiktoken.get_encoding("o200k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens
