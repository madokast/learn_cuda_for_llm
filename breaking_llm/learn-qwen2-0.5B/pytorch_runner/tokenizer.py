import json
import regex as re # 必须使用 regex 模块而不是 python 内置的 re 模块
from typing import Dict, List, Tuple, Callable, Generator

TokenId = int
MAX_MERGE_PRIORITY = 10_000_000


class Tokenizer:
    def __init__(
        self,
        added_tokens: Dict[str, TokenId],  # 特殊字符，主要是 `<|xxx|>`
        normalizer: Callable[[str], str],  # 归一化，例如 é 可以是一个单字符，也可以是 e 加上一个组合音标，需要统一
        splitter: re.Pattern,  # 切分正则
        byte_map: List[str],  # 字节流映射到 utf8 字符
        merge_table: Dict[Tuple[str, str], int],  # 合并规则表，合并 tuple(str, str) -> 优先级（0 开始）
        vocab2id: Dict[str, TokenId],  # 词汇表，将合并的单词转为 token id
    ) -> None:
        self.added_tokens = added_tokens
        self.normalizer = normalizer
        self.splitter = splitter
        self.byte_map = byte_map
        self.merge_table = merge_table
        self.vocab2id = vocab2id

        self.id2vocab:Dict[TokenId, str] = {**{v: k for k, v in self.vocab2id.items()}, **{v: k for k, v in self.added_tokens.items()}}
        self.byte_reversed_map:Dict[str, int] = {v: k for k, v in enumerate(self.byte_map)}

    def encode(self, text: str) -> Generator[TokenId, None, None]:
        """将文本编码为 token id 列表"""
        # 处理特殊字符
        for add_token, token_id in self.added_tokens.items():
            idx = text.find(add_token)
            # 找到任意一个就转为 id，剩余串则递归处理
            if idx >= 0:
                left, right = text[:idx], text[idx + len(add_token):]
                yield from self.encode(left)
                yield token_id
                yield from self.encode(right)
                return
            
        # 到此处，说明 text 中没有特殊字符
        # 第一步，进行归一化
        text = self.normalizer(text)

        # 第二步，预分词
        # 2.1 正则切分
        words: List[str] = self.splitter.findall(text)

        for word in words:
            # 2.2 转为 utf8 字节流
            byte_word: bytes = word.encode("utf-8")
            # 2.3 每个字节映射到一个 utf8 字符
            unicode_word: List[str] = [self.byte_map[b] for b in byte_word]

            # 第三步，进行 merge
            tokens: List[str] = self.__merge(unicode_word)
            # 第四步，将 merge 后的 utf8 字符转为 token id
            yield from (self.vocab2id[t] for t in tokens)

    def decode(self, token_ids: List[TokenId]) -> Generator[str, None, None]:
        for token_id in token_ids:
            token = self.id2vocab[token_id]
            if token in self.added_tokens:
                yield token
            else:
                # 从 utf8 字符中获取字节
                byte_list = (self.byte_reversed_map[s] for s in token)
                yield bytes(byte_list).decode("utf-8")
        

    def __merge(self, tokens: List[str]) -> List[str]:
        def merge0(tokens: List[str]) -> List[str]:
            # 记录最优先合并的 token 对
            candidate_id, candidate_priority = None, MAX_MERGE_PRIORITY
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                priority = self.merge_table.get(pair, MAX_MERGE_PRIORITY)
                if priority < candidate_priority:
                    candidate_id = i
                    candidate_priority = priority
            if candidate_id is not None:
                merged = tokens[candidate_id] + tokens[candidate_id + 1]
                return tokens[:candidate_id] + [merged] + tokens[candidate_id + 2:]
            # 无法合并
            return tokens

        # 一直合并到无法合并为止
        while True:
            new_tokens = merge0(tokens)
            if len(new_tokens) == len(tokens): 
                return new_tokens
            tokens = new_tokens


# Copied from transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def load_tokenizer(tokenizer_json_path):

    with open(tokenizer_json_path) as f:
        config = json.load(f)

    added_tokens:Dict[str, TokenId] = {}
    for added_token in config["added_tokens"]:
        content = added_token["content"]
        token_id = TokenId(added_token["id"])
        added_tokens[content] = token_id

    import unicodedata
    normalizer = lambda text: unicodedata.normalize("NFC", text)

    splitter_re = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    splitter = re.compile(splitter_re)

    bytes_to_unicode_dict = bytes_to_unicode()
    byte_map = [bytes_to_unicode_dict[i] for i in range(256)]

    merge_table: Dict[Tuple[str, str], int] = {}
    for idx, (t1, t2) in enumerate(config['model']['merges']):
        merge_table[(t1, t2)] = idx

    vocab: Dict[str, TokenId] = {}
    for idx, token in enumerate(config['model']['vocab']):
        vocab[token] = idx

    return Tokenizer(added_tokens, normalizer, splitter, byte_map, merge_table, vocab)


if __name__ == '__main__':
    from pathlib import Path
    cur_dir = Path(__file__).parent
    tokenizer = load_tokenizer(cur_dir / "../qwen2-0.5B-Instruct/tokenizer.json")

    text = """<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"""

    token_ids = list(tokenizer.encode(text))
    print(token_ids)

    assert token_ids == [151644,    872,    198,   9707, 151645,    198, 151644,  77091,    198]

    decoded = tokenizer.decode(token_ids)
    assert "".join(decoded) == text

