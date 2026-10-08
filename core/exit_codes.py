"""进程退出码契约：CI 据此区分「模型退化」与「环境故障」。"""
EXIT_OK = 0                     # 门禁通过
EXIT_GATE_FAILED = 1            # 质量不合格 —— 阻断发布
EXIT_INFRA_ERROR = 2            # 环境/网络/key/限流 —— 告警，不判断模型质量
EXIT_DATASET_INVALID = 3        # 数据集不合法 —— 数据负责人不通过
EXIT_CONFIG_ERROR = 4           # 配置/门禁文件/自评护栏 —— 退回配置责任人