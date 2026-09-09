from openai import OpenAI
import re


from openai import OpenAI

def localLLM(messages, model='mixtral:8x7b', temperature=0.3,
             base_url='http://localhost:11434/v1/', api_key='ollama'):  # 新增参数，用于传递 Ollama 选项
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    
    # 基础请求参数
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    

    
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content
    # pattern = r"(?<=\n</think>\n\n)[\s\S]*" 
    # only_response = re.search(pattern, response.choices[0].message.content)  #去除思考部分

    # return only_response.group().strip().lower().split(',')
    return response.choices[0].message.content
    
if __name__ == '__main__':
    # 测试示例：准备对话消息
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Why is the sky blue?"},
    ]
    
    # 设置 API 密钥
    api_key = "api_key"
    
    # 调用 localLLM 函数获取响应
    response = localLLM(messages)
    
    # 打印响应结果
    print(response)
    # analysis_prompt = """
        

    #     控制流节点：
    #     [{'id': '1', 'name': '注册账号并登录', 'description': '使用用户提供的邮箱地址进行注册，然后登录系统', 'type': '顺序', 'dependencies': []}, {'id': '2', 'name': '上传语音文件', 'description': '将用户录制的语音文件上传到系统中', 'type': '顺序', 'dependencies': ['1']}]

    #     每个节点对应的外部知识，用一个二维列表发送给你。二维列表里面的每个json列表代表一个节点的数个外部知识，每个节点只能在当前节点的选项中进行选择，json列表的字段包括：
    #     [[{'id': 2, 'service_name': '用户注册服务', 'api_url': 'https://api.example.com/auth/register', 'description': '支持新用户注册，检查邮箱和用户名是否重复，返回用户 ID。'}, {'id': 9, 'service_name': '邮件发送服务', 'api_url': 'https://api.example.com/mail/send', 'description': '支持发送邮件，支持 HTML 格式和附件，返回发送状态。'}, {'id': 1, 'service_name': '用户认证服务', 'api_url': 'https://api.example.com/auth/login', 'description': '提供用户登录验证，支持用户名和密码校验，返回 JWT token。'}], [{'id': 15, 'service_name': '语音转写服务', 'api_url': 'https://api.example.com/speech/transcribe', 'description': '支持语音文件转文字，返回转写文本和时间戳。'}, {'id': 3, 'service_name': '文件上传服务', 'api_url': 'https://api.example.com/file/upload', 'description': '支持用户上传文件，自动存储到对象存储系统，返回文件 URL。'}, {'id': 9, 'service_name': '邮件发送服务', 'api_url': 'https://api.example.com/mail/send', 'description': '支持发送邮件，支持 HTML 格式和附件，返回发送状态。'}]]

    #     """
    # D_SYSTEM_PROMPT = """
    #     你是一个智能工作流执行规划器。
    #     你的任务是：
    #     1. 接收用户输入和之前的大模型生成的控制流结构，每个控制流节点的外部知识。
    #     2.控制流不可改变，你要做的是为每个节点选择最合适的服务。对每个节点，我将分别给你数个选项。
    #     3. 输出整体的工作流
    #     4. JSON 结构必须包含：
    #     - 节点ID
    #         - 节点名称
    #         - 节点描述
    #         - 节点类型（顺序/并行/条件/循环）
    #         - 依赖节点
    #         - 服务选项（每个节点可调用的 API 列表，每个 API 包含：
    #             - service_name
    #             - api_url
    #             - description
    #             - input_data: 节点输入数据
    #             - output_data: 节点输出数据
    #         )
    #     5.每个节点的输入必须是前面步骤已经获取到的。
    #     """   
    # messages=[
    #         {"role": "user", "content": analysis_prompt},
    #         {"role": "system", "content": D_SYSTEM_PROMPT}
    #     ]
    # response = localLLM(messages)
    # print(response) 