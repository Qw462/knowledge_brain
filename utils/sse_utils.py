import asyncio
import json
from typing import Dict, Optional, Any, AsyncGenerator
import queue

from fastapi import Request


class SSEEvent:
    READY="ready"        # 连接建立
    PROGRESS="progress"  # 任务节点进度
    DELTA="delta"        # LLM流失输出增量
    FINAL="final"        # 最终完整答案
    ERROR="error"        # 错误信息
    CLOSE="close"        # 关闭连接信号


# 全局sse会话队列存储
# Key :session_id , Value:queue.Queue
_session_stream:Dict[str, "queue.Queue"]={}


def get_sse_queue(session_id:str) ->Optional["queue.Queue"]:
    """获取指定的session的队列"""
    return _session_stream.get(session_id)
def create_sse_queue(session_id:str) -> "queue.Queue":
    """创建并注册一个新的SSE队列"""
    q=queue.Queue()
    _session_stream[session_id]=q
    return q

def remove_sse_queue(session_id:str):
    """移除指定的session队列"""
    _session_stream.pop(session_id,None)

def _sse_pack(event:str,data:Dict[str,Any])->str:
    """打包sse消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

def push_to_session(session_id:str,event:str,data:Dict[str,Any]):
    """通过session_id推送事件"""
    stream_queue=get_sse_queue(session_id)
    if stream_queue:
        stream_queue.put({"event":event,"data":data})


async def sse_generator(session_id:str,request:Request)->AsyncGenerator[str,None]:
    """SSE事件生成器,用于FastAPI的StreamingResponse"""
    stream_queue=get_sse_queue(session_id)
    if stream_queue is None:
        #如果没有对应的队列，直接返回
        return
    loop=asyncio.get_running_loop()
    try:
        #发送连接建立信号
        yield _sse_pack("ready",{})
        while True:
            #若客户端断开尽快退出
            if await request.is_disconnected():
                break

            try:
                # 使用 run_in_executor 避免阻塞 async 事件循环
                msg=await loop.run_in_executor(None,stream_queue.get,True,1.0)
            except queue.Empty:
                continue
            event=msg.get("event")
            data=msg.get("data")

            #特殊关闭事件
            if event=="__close__":
                break

            yield _sse_pack(event,data)
    except (asyncio.CancelledError,ConnectionResetError,BrokenPipeError):
        return
    finally:
        remove_sse_queue(session_id)














