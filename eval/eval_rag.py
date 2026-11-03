import sys
import types

# 伪造缺失的vertexai模块，骗过ragas的预导入
mock_module = types.ModuleType("langchain_community.chat_models.vertexai")
mock_module.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = mock_module

"""
RAG 评估程序
基于 ragas 框架，对 RAG 系统进行五维指标评估
指标：Faithfulness、Answer Relevancy、Context Precision、Context Recall、Answer Correctness
"""

import os
import sys
import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from utils.llm_utils import get_llm_client
from utils.embedding_utils import get_bge_m3_ef
from processor.query_processor.main_graph import KBQueryWorkflow
from tool.logger import logger

# ======================== 路径配置 ========================
# 获取项目根目录，确保模块导入正确
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 评估数据目录
EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def step_1_load_test_set() -> pd.DataFrame:
    """
    步骤1：加载评估测试集
    从 eval/qa.csv 读取测试数据，包含 question 和 ground_truth 两列
    :return: 测试集 DataFrame
    """
    csv_path = os.path.join(EVAL_DIR, "qa.csv")
    # 读取 CSV，qa.csv 使用三引号包裹字段，需指定双引号为引用符
    df = pd.read_csv(csv_path, quotechar='"')
    logger.info(f"【步骤1】成功加载测试集，共 {len(df)} 条数据")
    return df


def step_2_run_rag_pipeline(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    步骤2：运行 RAG 流程，获取每个问题的回答和上下文证据
    调用 KBQueryWorkflow.run() 执行 RAG 工作流，
    从 state 中提取 answer（生成的回答）和 reranked_docs（重排序后的上下文证据）
    :param test_df: 测试集 DataFrame
    :return: 包含 question、ground_truth、answer、contexts 的 DataFrame
    """
    # 初始化 RAG 工作流
    workflow = KBQueryWorkflow()

    answers = []
    contexts_list = []

    for idx, row in test_df.iterrows():
        question = row["question"]
        logger.info(f"【步骤2】正在处理第 {idx + 1}/{len(test_df)} 个问题: {question}")

        # 构造初始状态，传入用户问题
        initial_state = {
            "session_id": f"eval_session_{idx}",
            "original_query": question,
        }

        # 调用 RAG 流程的 run 方法，获取完整状态
        result_state = workflow.run(initial_state)

        # 从 state 中提取生成的回答
        answer = result_state.get("answer", "")
        # 从 state 中提取重排序后的上下文证据（reranked_docs）
        reranked_docs = result_state.get("reranked_docs", [])

        answers.append(answer)
        # 从 reranked_docs 字典列表中提取 content 字段，转为字符串列表
        # 每个  doc 结构为：{chunk_id, title, content, url, source, score}
        context_texts = [doc.get("content", "") for doc in reranked_docs] if reranked_docs else [""]
        contexts_list.append(context_texts)

        logger.info(f"【步骤2】第 {idx + 1} 个问题处理完成，获取到 {len(reranked_docs)} 条上下文")

    # 将 RAG 结果合并到 DataFrame
    test_df = test_df.copy()
    test_df["answer"] = answers
    test_df["contexts"] = contexts_list

    logger.info(f"【步骤2】RAG 流程全部完成，共处理 {len(test_df)} 条数据")
    return test_df


def step_3_setup_ragas_models():
    """
    步骤3：配置 Ragas 评估所需的 LLM 和 Embedding 模型
    LLM 模型从 utils/llm_utils.py 的 get_llm_client 获取
    Embedding 模型基于本地 BGE-M3 模型加载
    :return: (ragas_llm, ragas_embeddings) 元组
    """
    # 获取项目已有的 LangChain LLM 客户端
    llm_client = get_llm_client()
    # 包装为 Ragas 兼容的 LLM 对象
    ragas_llm = LangchainLLMWrapper(llm_client)
    logger.info("【步骤3】LLM 模型配置完成")

    # 使用 langchain-community 的 HuggingFaceEmbeddings 加载本地 BGE-M3 模型
    # 因为项目的 BGEM3EmbeddingFunction 返回 dense+sparse 字典结构，
    # 与 Ragas 所需的 List[List[float]] 格式不兼容，所以使用 HuggingFaceEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from config.embedding_config import embedding_config

    langchain_embeddings = HuggingFaceEmbeddings(
        model_name=embedding_config.bge_m3_path,
        model_kwargs={"device": embedding_config.bge_device},
    )
    # 包装为 Ragas 兼容的 Embeddings 对象
    ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
    logger.info("【步骤3】Embedding 模型配置完成")

    return ragas_llm, ragas_embeddings


def step_4_run_evaluation(rag_df: pd.DataFrame, ragas_llm, ragas_embeddings):
    """
    步骤4：执行 Ragas 五维指标评估
    五个指标分别为：
      - Faithfulness（忠实度）：答案是否基于上下文中的事实
      - Answer Relevancy（答案相关性）：答案与问题的相关程度
      - Context Precision（上下文精确度）：上下文中相关信息的排序质量
      - Context Recall（上下文召回率）：上下文对标准答案的覆盖程度
      - Answer Correctness（答案正确性）：答案与标准答案的整体正确性
    :param rag_df: 包含 question、answer、contexts、ground_truth 的 DataFrame
    :param ragas_llm: Ragas 兼容的 LLM 对象
    :param ragas_embeddings: Ragas 兼容的 Embeddings 对象
    :return: 包含各指标得分的 DataFrame
    """
    # 构造 HuggingFace Dataset，Ragas evaluate 要求此格式
    # contexts 字段要求为 List[List[str]] 格式
    eval_dataset = Dataset.from_dict({
        "question": rag_df["question"].tolist(),
        "answer": rag_df["answer"].tolist(),
        "contexts": rag_df["contexts"].tolist(),
        "ground_truth": rag_df["ground_truth"].tolist(),
    })

    # 定义五个评估指标
    metrics = [
        Faithfulness(),           # 忠实度
        AnswerRelevancy(),        # 答案相关性
        ContextPrecision(),       # 上下文精确度
        ContextRecall(),          # 上下文召回率
        AnswerCorrectness(),      # 答案正确性
    ]

    logger.info("【步骤4】开始执行 Ragas 评估，共 5 个指标...")

    # 执行评估，传入数据集、指标列表、LLM 和 Embedding 模型
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    logger.info("【步骤4】Ragas 评估完成")
    return result


def step_5_save_results(rag_df: pd.DataFrame, eval_result):
    """
    步骤5：保存评估结果到 CSV 文件
    输出文件为 eval/qa_result.csv，编码为 UTF-8 BOM
    包含 9 列：question、context、answer、ground_truth、
               Faithfulness、Answer Relevancy、Context Precision、
               Context Recall、Answer Correctness
    :param rag_df: RAG 流程产出的 DataFrame
    :param eval_result: Ragas 评估结果对象
    """
    # 将评估结果转为 pandas DataFrame
    result_df = eval_result.to_pandas()

    # 构建输出 DataFrame，合并原始数据与评估指标得分
    output_df = pd.DataFrame({
        "question": rag_df["question"],
        "context": ["\n".join(ctx) for ctx in rag_df["contexts"]],  # 上下文列表拼接为字符串
        "answer": rag_df["answer"],
        "ground_truth": rag_df["ground_truth"],
        "Faithfulness": result_df["faithfulness"].tolist(),
        "Answer Relevancy": result_df["answer_relevancy"].tolist(),
        "Context Precision": result_df["context_precision"].tolist(),
        "Context Recall": result_df["context_recall"].tolist(),
        "Answer Correctness": result_df["answer_correctness"].tolist(),
    })

    # 保存为 CSV，使用 UTF-8 BOM 编码（便于 Excel 正确识别中文）
    output_path = os.path.join(EVAL_DIR, "qa_result.csv")
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info(f"【步骤5】评估结果已保存到: {output_path}")
    logger.info(f"【步骤5】各指标平均分：")
    for col in ["Faithfulness", "Answer Relevancy", "Context Precision",
                 "Context Recall", "Answer Correctness"]:
        logger.info(f"  {col}: {output_df[col].mean():.4f}")


def main():
    """主函数：串联所有评估步骤"""
    logger.info("=" * 60)
    logger.info("RAG 评估程序启动")
    logger.info("=" * 60)

    # 步骤1：加载测试集
    test_df = step_1_load_test_set()

    # 步骤2：运行 RAG 流程，获取回答和上下文
    rag_df = step_2_run_rag_pipeline(test_df)

    # 步骤3：配置 Ragas 评估所需的 LLM 和 Embedding 模型
    ragas_llm, ragas_embeddings = step_3_setup_ragas_models()

    # 步骤4：执行五维指标评估
    eval_result = step_4_run_evaluation(rag_df, ragas_llm, ragas_embeddings)

    # 步骤5：保存评估结果
    step_5_save_results(rag_df, eval_result)

    logger.info("=" * 60)
    logger.info("RAG 评估程序全部完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()