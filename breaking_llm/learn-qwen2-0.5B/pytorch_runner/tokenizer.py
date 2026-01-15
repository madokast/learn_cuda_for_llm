import re
from typing import Dict, List, Tuple, Callable, Generator

TokenId = int
MAX_MERGE_PRIORITY = 10_000_000

class Tokenizer:
    def __init__(self) -> None:
        # 特殊字符，主要是 `<|xxx|>`
        self.added_tokens: Dict[str, TokenId] = {}
        # 归一化，例如 é 可以是一个单字符，也可以是 e 加上一个组合音标，需要统一
        self.normalizer:Callable[[str], str] = lambda x: x
        # 切分正则
        self.splitter:re.Pattern = re.compile(r"...")
        # 字节流映射到 utf8 字符
        self.byte_map: List[str] = [] # 256 长度数组完成映射
        # 合并规则表，合并 tuple(str, str) -> 优先级（0 开始）
        self.merge_table: Dict[Tuple[str, str], int] = {}
        # 词汇表，将合并的单词转为 token id
        self.vocab: Dict[str, TokenId] = {}

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
        words: List[str] = self.splitter.split(text)

        for word in words:
            # 2.2 转为 utf8 字节流
            byte_word: bytes = word.encode("utf-8")
            # 2.3 每个字节映射到一个 utf8 字符
            unicode_word: List[str] = [self.byte_map[b] for b in byte_word]

            # 第三步，进行 merge
            tokens: List[str] = self.__merge(unicode_word)
            # 第四步，将 merge 后的 utf8 字符转为 token id
            yield from (self.vocab[t] for t in tokens)

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
