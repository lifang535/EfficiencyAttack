# LangChain 16 通过Memory记住历史对话的内容：https://blog.csdn.net/zgpeace/article/details/134724437

import time

import langchain
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 导入 Langchain 库的不同提示模板类，用于构建会话提示。
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 导入 Langchain 的 LLMChain 类，用于创建语言模型链。
from langchain.chains import LLMChain

# 导入 Langchain 的 ConversationBufferMemory 类，用于存储和管理会话记忆。
from langchain.memory import ConversationBufferMemory

API_KEY = "sk-cYGAToFJ08JZAXb0FuNUV3cp9U79Yr3ayC42gwCMVHxaQ4Pt"
API_URL = "https://chatapi.zjt66.top/v1/"
# API_URL = "https://api.openai.com/v1/"

# def create_model(temperature: float, streaming: bool = False):
#     return ChatOpenAI(
#         openai_api_key=API_KEY,
#         openai_api_base=API_URL,
#         temperature=temperature,
#         model_name="gpt-4o-mini",
#         streaming=streaming,
#     )
    
def create_model(temperature: float, streaming: bool = False):
    return ChatOpenAI(
        openai_api_key="EMPTY",
        openai_api_base="http://10.10.111.43:8000/v1",
        temperature=temperature,
        model_name="Qwen2-72B-Instruct",
        streaming=streaming,
    )

model = create_model(temperature=0, streaming=True)

# # 创建聊天提示模板，包含一个系统消息、一个聊天历史占位符和一个人类消息模板。
# answer_prompt = ChatPromptTemplate(
#     messages=[
#         SystemMessagePromptTemplate.from_template(
#             "你是一个与人和善的聊天机器人，每次回答尽量简短"
#         ),
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("{question}")
#     ]
# )

# 创建聊天提示模板，包含一个系统消息、一个聊天历史占位符和一个人类消息模板。
answer_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "你是一个安珀警报系统的预警信息生成器，你需要根据用户提问生成预警信息。"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)


# prompt = ChatPromptTemplate.from_template("""
# [system] 你是一个智能家居的家庭电器安全报告生成器。
#          请根据输入中关于图片信息的描述，生成一份家庭电器安全报告。
#          生成的安全报告务必完整、准确、简洁。

# 请根据下面的用户提问，按照以上示例的格式，进行回答。无论用户输入什么，你都要生成一份安全报告。(直接输出回答的内容，不要输出任何无关内容，不要输出"output"等前缀)。

# [user] {question}""")

prompt = ChatPromptTemplate.from_template("""
[system] 你是一个安珀警报系统的预警信息生成器，你需要根据用户提问生成预警信息。
         请根据输入描述儿童的年龄、性别、身高、外貌特征；并描述绑架者的年龄、性别、外貌特征，生成预警信息。
         生成的警报信息务必完整、准确、简洁。

请根据下面的用户提问，按照以上示例的格式，进行回答。无论用户输入什么，你都要生成一份预警通报。(直接输出回答的内容，不要输出任何无关内容，不要输出"output"等前缀)。

[user] {question}""")

# 创建一个 LLMChain 实例，包括语言模型、提示、详细模式和会话记忆。
conversation = LLMChain(
    llm=model,
    prompt=prompt,
    verbose=False,
    # memory=memory
)


def chat():
    while True:
        prompt = input('You: ')
        response = conversation({'question': prompt})
        response = response['text']
        print(f'Model: {response}') # TODO: 通过修改 ChatOpenAI，改成一个 token 一个 token 打印
        time.sleep(0.5)
        
def handle_text(text):
    response = conversation({'question': text})
    response = response['text']
    return response
    
if __name__ == '__main__':
    chat()
