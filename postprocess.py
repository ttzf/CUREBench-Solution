import re
import logging
import json

logger = logging.getLogger(__name__)

class PostProcessor:
    def __init__(self, model):
        self.model = model

    def process_multi_choice(self, response: str, current_choice: str = "") -> str:
        """
        处理选择题：优先使用已有选择，否则正则提取，最后尝试 fallback
        """
        # 1. 检查现有选择是否有效
        if self._is_valid_choice(current_choice):
            return current_choice

        # 2. 正则提取
        extracted = self._extract_choice_regex(response)
        if extracted:
            logger.info(f"Regex extracted choice: {extracted}")
            return extracted

        # 3. Fallback: 让模型重试 (需要调用方处理，或者这里直接调用模型)
        # 注意：为了避免死循环和过多的 API 调用，这里我们建议由调用方控制是否进行 fallback
        # 但根据需求，如果这里可以直接调用模型会更方便封装
        
        return "NOTAVALUE"

    def run_fallback_for_choice(self, question: str) -> str:
        """
        针对选择题的 Fallback 机制：重新询问模型
        """
        if not self.model:
            return "NOTAVALUE"
            
        logger.info("Running fallback for multiple choice...")
        prompt = f"Question: {question}\n\nPlease output ONLY the single letter (A, B, C, D, or E) corresponding to the correct answer. Do not output any other text."
        try:
            response, _ = self.model.inference(prompt)
            extracted = self._extract_choice_regex(response)
            if extracted:
                logger.info(f"Fallback successful: {extracted}")
                return extracted
        except Exception as e:
            logger.error(f"Fallback inference failed: {e}")
            
        return "NOTAVALUE"

    def process_open_ended(self, answer_text: str) -> str:
        """
        处理开放题：清洗多余解释，只保留最终答案句
        """
        if not answer_text:
            return ""

        # 清除常见的 Markdown 标记
        cleaned = answer_text.strip()
        
        # 策略 1: 如果包含 "The answer is" 或 "Final Answer:"，提取其后的内容
        patterns = [
            r"(?:The answer is|Final Answer[:]?|Conclusion[:]?)\s*(.*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
                break

        # 策略 2: 如果还是太长，尝试提取最后一句 (简单启发式)
        # 这里假设答案通常在最后。如果最后一句太短（如 "Thank you."），可能需要倒数第二句
        # 为简单起见，如果有多行，我们保留最后非空行
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        if lines:
            last_line = lines[-1]
            # 如果最后一行看起来是答案，就用它
            if len(lines) > 1 and len(last_line) < 200: 
                return last_line
        
        return cleaned

    def validate_open_ended_multi_choice(self, choice: str, open_ended_answer: str) -> str:
        """
        确保 choice 和 prediction (open_ended_answer) 一致
        目前主要检查 choice 是否存在。
        如果 choice 无效，尝试从 open_ended_answer 中再次提取。
        """
        if self._is_valid_choice(choice):
            return choice
            
        # 尝试从文本答案中提取
        extracted = self._extract_choice_regex(open_ended_answer)
        if extracted:
            logger.info(f"Recovered choice from open-ended answer: {extracted}")
            return extracted
            
        return "NOTAVALUE"

    def _is_valid_choice(self, choice: str) -> bool:
        return choice and choice.strip().upper() in ['A', 'B', 'C', 'D', 'E']

    def _extract_choice_regex(self, text: str) -> str:
        if not text:
            return ""
        
        text = text.strip()
        
        # 1. 直接就是单个字母
        if text.upper() in ['A', 'B', 'C', 'D', 'E']:
            return text.upper()
            
        # 2. 常见的 Answer 模式
        patterns = [
            r"(?:answer is|answer:|is)\s*([ABCDE])\b",
            r"^\s*([ABCDE])\s*[).]",  # A) or A. at start of line
            r"\(([ABCDE])\)",          # (A)
            r"\*\*([ABCDE])\*\*",      # **A**
            r"\[([ABCDE])\]"           # [A]
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
                
        return ""
