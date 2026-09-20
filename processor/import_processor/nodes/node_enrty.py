# processor/import_processor/nodes/node_entry.py
import logging
import json
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, ValidationError
from processor.import_processor.state import ImportGraphState

from pathlib import Path

class NodeEntry(BaseNode):
    """
    入口节点：任务分发
    """

    name = "node_entry"

    def process(self, state: ImportGraphState):
        #1 从state中获取文件的绝对路径
        import_file_path=state.get("import_file_path")
        #判断路径是否为空
        if not import_file_path:
            raise StateFieldError(
                field_name="import_file_path",
                message="文件路径不能为空",
                expected_type=str
            )

        #2 将import_file_path转换为path对象
        import_file_path_obj=Path(import_file_path)

        if not import_file_path_obj.exists():
            raise FileProcessingError(message=f"文件{import_file_path_obj}不存在")

        #3 判断文件类型
        if import_file_path_obj.suffix==".pdf":
            state["is_pdf_read_enabled"]=True
            state["pdf_path"]=import_file_path
        elif import_file_path_obj.suffix==".md":
            state["is_md_read_enabled"]=True
            state["md_path"]=import_file_path
        else:
            raise ValidationError(message=f"不支持文件类型{import_file_path_obj.suffix}")

        #4 获取文件名作为标题
        state["file_title"]=import_file_path_obj.stem

        return state

if __name__ == "__main__":
    #激活日志
    setup_logging()

    #初始化图状态
    init_state={
        "import_file_path":r"E:\BaiduNetdiskDownload\项目文件\2.资料\04-设备手册汇总\doc\hak180产品安全手册.pdf"
    }

    #创建节点对象
    node_entry=NodeEntry()
    #执行节点单元测试
    result=node_entry(init_state)

    #将返回的状态图进行json序列化
    json_state=json.dumps(result,ensure_ascii=False,indent=4)

    #输出
    logging.getLogger().info(json_state)