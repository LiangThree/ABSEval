"""
实现chat model的instruction构造方法
"""
from typing import List, Dict
from jinja2 import Template

from transformers import AutoTokenizer, AutoModel


class MetaChatCompleter:
    def __init__(self) -> None:
        pass

    @staticmethod
    def complete_chat_instruction(prompt: str, history: List[Dict[str, str]]) -> str:
        """
        用于构造chat模型的instruction指令
        history的格式是[{'user': 'xxx'}, {'bot': 'xxx'}]
        """
        pass


class ChatCompleterFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_chat_completer(model_repo_id: str, model: AutoModel = None,
                              tokenizer: AutoTokenizer = None) -> MetaChatCompleter:
        model_org, model_name = model_repo_id.split('/')
        if model_org == 'baichuan-inc':
            if 'baichuan2' in model_name:
                return Baichuan2ChatCompleter()
            else:
                return BaichuanChatCompleter()
        elif model_org == 'qwen':
            return QwenChatCompleter()
        elif 'llama3' in model_repo_id.lower():
            return Llama3ChatCompleter(tokenizer)
        elif model_org == 'meta' and 'llama3' not in model_repo_id.lower():
            return LlamaChatCompleter()
        elif model_org == '01ai':
            return YiChatCompleter()
        elif model_org == 'THUDM':
            if 'chatglm-6b' in model_name:
                return ChatglmChatCompleter()
            elif 'chatglm2' in model_name:
                return Chatglm2ChatCompleter()
            elif 'chatglm3' in model_name:
                return Chatglm3ChatCompleter()
        elif model_org == "mistralai":
            return MistralChatCompleter()
        elif model_org == "lmsys":
            return VicunaChatCompleter()
        elif model_org == "tiiuae":
            return FalconChatCompleter()
        elif model_org == "WizardLM":
            return WizardLMChatCompleter()
        raise ValueError(f'model_org: {model_org} not found')


class BaichuanChatCompleter(MetaChatCompleter):
    def __init__(self) -> None:
        super().__init__()
        self.user_token_str = "<reserved_102>"
        self.assistant_token_str = "<reserved_103>"

    def complete_chat_instruction(self, prompt: str, history: List[Dict[str, str]]) -> str:
        """
        用于构造chat模型的instruction指令
        history的格式是[{'role': 'user/assitant/system', 'content': 'xxx'}]}}]
        {system_content}<reserved_106>{user_content}<reserved_107>{assistant_content}...
        """
        total_input = ""
        # 剔除system的消息input
        if len(history) > 0 and history[0]['role'] == 'system':
            history = history[1:]

        # 构造历史消息input
        for message in history:
            role_str = self.user_token_str if message['role'] == 'user' else self.assistant_token_str
            total_input += f"{role_str} {message['content']}"

        # 构造用户输入prompt
        total_input += f"{self.user_token_str} {prompt}"

        # 在最后添加助手的标记，表示接下来的回复将属于助手
        total_input += self.assistant_token_str

        return total_input


class Baichuan2ChatCompleter(MetaChatCompleter):
    def __init__(self) -> None:
        super().__init__()
        self.user_token_str = "<reserved_106>"
        self.assistant_token_str = "<reserved_107>"

    def complete_chat_instruction(self, prompt: str, history: List[Dict[str, str]]) -> str:
        """
        用于构造chat模型的instruction指令
        history的格式是[{'role': 'user/assitant/system', 'content': 'xxx'}]}}]
        {system_content}<reserved_106>{user_content}<reserved_107>{assistant_content}...
        """
        total_input = ""
        # 构造system的消息input
        if len(history) > 0 and history[0]['role'] == 'system':
            total_input += f"{history[0]['content']}"
            history = history[1:]

        # 构造历史消息input
        for message in history:
            role_str = self.user_token_str if message['role'] == 'user' else self.assistant_token_str
            total_input += f"{role_str}{message['content']}"

        # 构造用户输入prompt
        total_input += f"{self.user_token_str}{prompt}"

        # 在最后添加助手的标记，表示接下来的回复将属于助手
        total_input += self.assistant_token_str

        return total_input


class YiChatCompleter(MetaChatCompleter):
    def __init__(self) -> None:
        super().__init__()
        self.user_token_str = "user"
        self.assistant_token_str = "assistant"
        self.system_token_str = "system"
        self.start_token_str = "<|im_start|>"
        self.end_token_str = "<|im_end|>"

    def complete_chat_instruction(self, prompt: str, history: List[Dict[str, str]]) -> str:
        """
        用于构造chat模型的instruction指令
        history的格式是[{'role': 'user/assitant/system', 'content': 'xxx'}]
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant
        """
        total_input = ""
        # 构造system的消息input
        if len(history) > 0 and history[0]['role'] == 'system':
            total_input += f"{self.start_token_str}{self.system_token_str}\n{history[0]['content']}{self.end_token_str}\n"
            history = history[1:]

        # 构造历史消息prompt
        for history_item in history:
            role_str = self.user_token_str if history_item['role'] == 'user' else self.assistant_token_str
            total_input += f"{self.start_token_str}{role_str}\n"
            total_input += f"{history_item['content']}{self.end_token_str}\n"

        # 构造用户输入的prompt
        total_input += f"{self.start_token_str}{self.user_token_str}\n"
        total_input += f"{prompt}{self.end_token_str}\n"

        # 构造助手的prompt提示词
        total_input += f"{self.start_token_str}{self.assistant_token_str}\n"

        return total_input

class Llama3ChatCompleter(MetaChatCompleter):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def complete_chat_instruction(self, prompt: str, history: List[Dict[str, str]]) -> str:
        history.append({'role': 'user', 'content': prompt})
        result = self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        return result

class LlamaChatCompleter(MetaChatCompleter):
    def __init__(self) -> None:
        super().__init__()
        self.bos_token = '<s>'
        self.ens_token = '</s>'
        self.DEFAULT_SYSTEM_PROMPT = \
            """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure \
that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""

    def complete_chat_instruction(self, prompt: str, history: List[Dict[str, str]]) -> str:
        """
        用于构造chat模型的instruction指令
        history的格式是[{'user': 'xxx'}, {'bot': 'xxx'}]
        """
        messages = history.copy()
        messages.append({'role': 'user', 'content': prompt})
        template = Template(self.chat_template)
        return template.render(messages=messages, bos_token=self.bos_token, eos_token=self.ens_token)

    @property
    def chat_template(self):
        DEFAULT_SYSTEM_PROMPT = self.DEFAULT_SYSTEM_PROMPT
        template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
            "{% set system_message = messages[0]['content'] %}"
            "{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}"
            "{% set loop_messages = messages %}"  # Or use the default system message if the flag is set
            "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = false %}"
            "{% endif %}"
            "{% for message in loop_messages %}"  # Loop over all non-system messages
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if loop.index0 == 0 and system_message != false %}"  # Embed system message in first message
            "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
            "{% else %}"
            "{% set content = message['content'] %}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way
            "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
            "{% elif message['role'] == 'system' %}"
            "{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' '  + content.strip() + ' ' + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        )
        template = template.replace("USE_DEFAULT_PROMPT", "true")
        default_message = DEFAULT_SYSTEM_PROMPT.replace("\n", "\\n").replace("'", "\\'")
        template = template.replace("DEFAULT_SYSTEM_MESSAGE", default_message)

        return template


class QwenChatCompleter(MetaChatCompleter):
    """
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    """

    def __init__(self):
        super().__init__()
        self.user_token_str = "user"
        self.assistant_token_str = "assistant"
        self.system_token_str = "system"
        self.start_token_str = "<|im_start|>"
        self.end_token_str = "<|im_end|>"

    def complete_chat_instruction(self, prompt: str, history: List[Dict[str, str]]) -> str:
        """
        用于构造chat模型的instruction指令
        history的格式是[{'role': 'user/assitant/system', 'content': 'xxx'}]
        """
        total_input = ""
        # 构造system的消息input
        if len(history) > 0 and history[0]['role'] == 'system':
            total_input += f"{self.start_token_str}{self.system_token_str}\n{history[0]['content']}{self.end_token_str}\n"
            history = history[1:]

        # 构造历史消息prompt
        for history_item in history:
            role_str = self.user_token_str if history_item['role'] == 'user' else self.assistant_token_str
            total_input += f"{self.start_token_str}{role_str}\n"
            total_input += f"{history_item['content']}{self.end_token_str}\n"

        # 构造用户输入的prompt
        total_input += f"{self.start_token_str}{self.user_token_str}\n"
        total_input += f"{prompt}{self.end_token_str}\n"

        # 构造助手的prompt提示词
        total_input += f"{self.start_token_str}{self.assistant_token_str}\n"

        return total_input


class Chatglm3ChatCompleter(MetaChatCompleter):
    """
    [gMASK]sop<|system|>
    system content<|user|>
    user content<|assistant|>
    assistant content<|user|>
    user content<|assistant|>
    """

    def __init__(self):
        super().__init__()
        self.user_token_str = "<|user|>"
        self.assistant_token_str = "<|assistant|>"
        self.system_token_str = "<|system|>"
        self.global_start_token_str = "[gMASK]sop"

    def complete_chat_instruction(self, prompt: str, history: List[Dict[str, str]]) -> str:
        """
        用于构造chat模型的instruction指令
        history的格式是[{'role': 'user/assitant/system', 'content': 'xxx'}]
        """
        total_input = ""
        total_input += self.global_start_token_str
        # 构造system的消息input
        if len(history) > 0 and history[0]['role'] == 'system':
            total_input += f"{self.system_token_str}\n"
            total_input += f"{history[0]['content']}"
            history = history[1:]

        # 构造历史消息prompt
        for history_item in history:
            role_str = self.user_token_str if history_item['role'] == 'user' else self.assistant_token_str
            total_input += f"{role_str}\n "
            total_input += f"{history_item['content']}"

        # 构造用户输入的prompt
        total_input += f"{self.user_token_str}\n "
        total_input += f"{prompt}"

        # 构造助手的prompt提示词
        total_input += f"{self.assistant_token_str}"

        return total_input


class Chatglm2ChatCompleter(MetaChatCompleter):
    def __init__(self) -> None:
        super().__init__()

    def complete_chat_instruction(self, prompt: str, history: List[Dict[str, str]]) -> str:
        """
        用于构造chatglm2模型的instruction指令
        history的格式是[{'user': 'xxx', 'bot': 'xxx'}]
        """
        formatted_history = []
        for i, message in enumerate(history):
            if 'user' in message and 'bot' in message:
                formatted_history.append((message['user'], message['bot']))

        # 构建聊天轮次的格式化字符串
        prompt_text = self.build_prompt(prompt, formatted_history)
        return prompt_text

    def build_prompt(self, query: str, history: List[Dict[str, str]] = None):
        """
        构建chatglm2模型所需的输入格式
        """
        if history is None:
            history = []
        prompt = ""
        for i, message in enumerate(history):
            if 'user' in message and 'bot' in message:
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, message['user'], message['bot'])
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        return prompt


class ChatglmChatCompleter(MetaChatCompleter):
    def __init__(self) -> None:
        super().__init__()

    def complete_chat_instruction(self, prompt: str, history: List[Dict[str, str]]) -> str:
        """
        用于构造Chatglm-6b模型的instruction指令
        history的格式是[{'user': 'xxx', 'bot': 'xxx'}]
        """
        if not history:
            return prompt
        else:
            return self.build_prompt(prompt, history)

    def build_prompt(self, query: str, history: List[Dict[str, str]] = None):
        """
        构建Chatglm-6b模型所需的输入格式
        """
        prompt = ""
        for i, message in enumerate(history):
            if 'user' in message and 'bot' in message:
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, message['user'], message['bot'])
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        return prompt


class MistralChatCompleter(MetaChatCompleter):
    def __init__(self) -> None:
        super().__init__()
        self.instruction_start = "<INST>"
        self.instruction_end = "</INST>"
        self.start_token = "<s>"
        self.end_token = "</s>"

    def complete_chat_instruction(self, prompt: str, history: List[Dict[str, str]]) -> str:
        """
        用于构造Instruction模型的instruction指令
        The very first instruction should begin with a begin of sentence id. The next instructions should not. The assistant generation will be ended by the end-of-sentence token id.
        text = "<s>[INST] What is your favourite condiment? [/INST]"
        "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
        "[INST] Do you have mayonnaise recipes? [/INST]"
        history的格式是[{'role': 'user/assitant/', 'content': 'xxx'}]
        """
        total_input = self.start_token
        # 构造历史消息input
        for message in history:
            if message["role"] == "user":
                total_input += f"{self.instruction_start} {message['content']} {self.instruction_end}"
            if message["role"] == "assistent":
                total_input += f"{message['content']}{self.end_token}"

        # 构造用户输入prompt
        total_input += f"{self.instruction_start} {prompt} {self.instruction_end}"

        return total_input


class VicunaChatCompleter(MetaChatCompleter):
    def __init__(self) -> None:
        super().__init__()
        self.user = "USER:"
        self.assistant = "ASSISTANT:"
        self.sys_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"

    def complete_chat_instruction(self, prompt: str, history: List[Dict[str, str]]) -> str:
        total_input = self.sys_prompt
        # 构造历史消息input
        for message in history:
            if message["role"] == "user":
                total_input += f"{self.user} {message['content']}\n"
            if message["role"] == "assistent":
                total_input += f"{self.assistant} {message['content']}\n"

        # 构造用户输入prompt
        total_input += f"{self.user} {prompt}\n{self.assistant}"
        return total_input


class FalconChatCompleter(MetaChatCompleter):
    def __init__(self) -> None:
        super().__init__()
        self.user = "USER:"
        self.assistant = "ASSISTANT:"

    def complete_chat_instruction(self, prompt: str, history: List[Dict[str, str]]) -> str:
        total_input = ""
        # 构造历史消息input
        for message in history:
            if message["role"] == "user":
                total_input += f"{self.user} {message['content']}\n"
            if message["role"] == "assistent":
                total_input += f"{self.assistant} {message['content']}\n"

        # 构造用户输入prompt
        total_input += f"{self.user} {prompt}\n{self.assistant}"
        return total_input


class WizardLMChatCompleter(MetaChatCompleter):
    def __init__(self) -> None:
        super().__init__()
        self.user = "USER:"
        self.assistant = "ASSISTANT:"
        self.sys_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "

    def complete_chat_instruction(self, prompt: str, history: List[Dict[str, str]]) -> str:
        total_input = self.sys_prompt
        # 构造历史消息input
        for message in history:
            if message["role"] == "user":
                total_input += f"{self.user} {message['content']} "
            if message["role"] == "assistent":
                total_input += f"{self.assistant} {message['content']} "

        # 构造用户输入prompt
        total_input += f"{self.user} {prompt} {self.assistant}"
        return total_input
