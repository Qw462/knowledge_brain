## 生成测试集

【核心目标】
    根据 eval/hak180产品安全手册_new.md 生成一个用于 ragas测试的评估的测试集
【要求】
    1 生成测试集格式 csv 文件，文件名称 qa_test.csv
    2 测试集包含列为： question,  ground_truth
    3 请使用“ HAK 180 烫金机 ”  作为每个问题的前缀
    4 把生成文件保存到 eval 目录下
    5 生成5对问题和答案
## 生成评估程序

【核心目标】
    生成一个基于 ragas的python的评估程序，用于评估测试集
【程序要求】
    【输入要求】
        要读取eval/qa.csv，作为评估测试集。
    【过程要求】
        1 rag的流程入口 在 processor\query_process\main_graph.py 中的run方法
        2 其中要求的上下文证据要从state中的reranked_docs中提取
        3 ragas需要的llm模型和embedding模型工具 
           分别从utils\llm_client_util.py 
           和 utils\embedding_utils.py 获得 
        4 评估要求的5个指标：Faithfulness 、
            Answer Relevancy、Context Precision、
            Context Recall、Answer Correctness 
    【输出要求】
        1 评估输出文件为 qa_result.csv ，
                    保存在eval 目录下 ，编码集 utf-8 bom
        2 qa_result.csv 包括9列：question、 
             context、 answer、ground_truth、Faithfulness 、
             Answer Relevancy、Context Precision、
             Context Recall、Answer Correctness
     【代码风格要求】
         1 每个函数都要有注释，核心名词注释，核心步骤注释
         2 每个核心步骤的函数名要求用step_1_xx step_2_xx来命名。