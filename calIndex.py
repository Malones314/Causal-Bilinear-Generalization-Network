# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score, f1_score, roc_curve, roc_auc_score, auc
# import numpy as np
# def cal_index(y_true, y_pred_labels, y_pred_probs):
#     """
#     Calculate Accuracy, Recall, Precision, F1-Score, and AUC.
#     This corrected version removes the flawed `labels` parameter to ensure accurate metric calculation.
#     """
#     acc_ = accuracy_score(y_true, y_pred_labels)
#
#     # 添加 `zero_division=0` 防止因分母为零发出警告
#     prec_ = precision_score(y_true, y_pred_labels, average='macro', zero_division=0)
#     recall_ = recall_score(y_true, y_pred_labels, average='macro', zero_division=0)
#     F1_score_ = f1_score(y_true, y_pred_labels, average='macro', zero_division=0)
#
#     # 计算 AUC (此部分逻辑正确)
#     n_classes = y_pred_probs.shape[1]
#     try:
#         # 确保 y_true 中有多个类别才能计算AUC
#         if len(np.unique(y_true)) < 2:
#             raise ValueError("Only one class present in y_true. ROC AUC score is not defined.")
#
#         if n_classes > 2:
#             auc_ = roc_auc_score(y_true, y_pred_probs, multi_class='ovo', average='macro')
#         else:
#             # 对于二分类，标准做法是使用正类（标签为1）的概率
#             auc_ = roc_auc_score(y_true, y_pred_probs[:, 1])
#
#     except ValueError as e:
#         # 打印错误信息以帮助调试，然后返回一个表示失败的值
#         # print(f"AUC calculation failed: {e}")
#         auc_ = -1.0
#
#     return acc_, auc_, prec_, recall_, F1_score_

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def cal_index(y_true, y_pred_labels, y_pred_probs):
    """
    计算指标，自动适配二分类和多分类。
    """
    # 1. 计算 Accuracy
    acc_ = accuracy_score(y_true, y_pred_labels)

    # 获取类别数量
    n_classes = y_pred_probs.shape[1]

    # 2. 区分二分类和多分类的计算方式
    if n_classes == 2:
        # 【二分类关键修改】：使用 'binary' 模式，直接反映“故障类（Label=1）”的性能
        # 这样如果模型全预测为0，Recall和F1会直接变成0，而不是误导性的0.5
        prec_ = precision_score(y_true, y_pred_labels, pos_label=1, average='binary', zero_division=0)
        recall_ = recall_score(y_true, y_pred_labels, pos_label=1, average='binary', zero_division=0)
        F1_score_ = f1_score(y_true, y_pred_labels, pos_label=1, average='binary', zero_division=0)
    else:
        # 多分类保持 macro
        prec_ = precision_score(y_true, y_pred_labels, average='macro', zero_division=0)
        recall_ = recall_score(y_true, y_pred_labels, average='macro', zero_division=0)
        F1_score_ = f1_score(y_true, y_pred_labels, average='macro', zero_division=0)

    # 3. 计算 AUC (更安全的处理)
    auc_ = 0.5 # 默认值，相当于随机猜测
    try:
        # 必须包含至少两个类别才能计算 ROC AUC
        if len(np.unique(y_true)) < 2:
            # 这种情况在 Batch 较小或模型坍塌时常见
            # 返回 0.5 (中性) 或 None (不参与统计)，不要返回 -1
            auc_ = 0.5
        else:
            if n_classes > 2:
                auc_ = roc_auc_score(y_true, y_pred_probs, multi_class='ovo', average='macro')
            else:
                # 二分类：取正类的概率
                auc_ = roc_auc_score(y_true, y_pred_probs[:, 1])
    except Exception as e:
        print(f"[Warning] AUC calculation error: {e}")
        auc_ = 0.5

    return acc_, auc_, prec_, recall_, F1_score_