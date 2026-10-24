# processor/query_processor/prompt/item_name_confirm.py

ITEM_NAME_EXTRACT_SYSTEM_PROMPT = "你是一个专业的河南旅游助手，擅长理解用户意图、从对话中提取景点/目的地等关键信息。"

ITEM_NAME_EXTRACT_TEMPLATE = """
历史会话：
{history_text}

当前用户问题：
{query}

请根据历史会话和当前问题，提取用户正在询问的河南旅游景点/目的地名称（item_names）。
1. 如果用户明确提到了景点/目的地名称，请提取出来。可能有一个或多个，但不能重复。
2. 如果用户使用了代词或指代（如"这个景区""那座山""它"），请结合历史会话指代消解，确定景点名称。
3. 如果无法确定景点名称，item_names 返回空列表。
4. 请重新改写用户的问题（rewritten_query），使其成为包含景点名称的独立完整问题。

请直接返回JSON格式结果，格式如下：
{{
    "item_names": ["少林寺", "龙门石窟"],
    "rewritten_query": "关于少林寺和龙门石窟，..."
}}
"""