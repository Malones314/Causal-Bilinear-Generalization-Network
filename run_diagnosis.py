print("DEBUG: 进入测试程序", flush=True)
import os
import sys
import argparse
import json

print("DEBUG: Python后端进程已启动，准备导入基础库...", flush=True)

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import types



# 确保脚本能找到你的自定义模块
# 假设 run_diagnosis.py 和 CBGN.py 在同级或可通过 sys.path 访问
try:
    from CBGN import CBGN
except ImportError:
    # 如果运行路径不对，尝试将当前目录加入环境变量
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from CBGN import CBGN

# ================= 配置区域 (对齐你的 wav_process_section.py) =================
CONFIG = {
    'n_mels': 128,  
    'frames': 157,  
    'n_fft': 1024,  
    'hop_length': 210,  
    'dtype_feature': np.float32
}

def dict_to_namespace(d):
    """将字典递归转换为 SimpleNamespace 对象，适配旧配置加载"""
    if not isinstance(d, dict):
        return d
    n = types.SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(n, k, dict_to_namespace(v))
        else:
            setattr(n, k, v)
    return n

def adjust_audio_length(y, target_length):
    """动态音频长度调整策略，不足则填充微弱噪声"""
    if len(y) < target_length:
        noise = np.random.normal(0, 0.001, target_length - len(y))
        return np.concatenate([y, noise])
    return y[:target_length]

def wav_to_mel1d(file_name):
    """提取Log-Mel特征并转化为模型需要的维度"""
    try:
        y, sr = librosa.load(file_name, sr=None, mono=True)
    except Exception:
        return None # 无法读取或非音频文件则静默跳过

    try:
        required_samples = (CONFIG['frames'] - 1) * CONFIG['hop_length'] + CONFIG['n_fft']
        y = adjust_audio_length(y, required_samples)

        mel = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length'],
            n_mels=CONFIG['n_mels'],
            power=2.0
        )

        log_mel = np.log1p(mel + sys.float_info.epsilon)
        log_mel = log_mel.astype(CONFIG['dtype_feature'])

        current_frames = log_mel.shape[1]
        if current_frames < CONFIG['frames']:
            log_mel = np.pad(log_mel, ((0, 0), (0, CONFIG['frames'] - current_frames)),
                             mode='constant', constant_values=-80)
        elif current_frames > CONFIG['frames']:
            start = (current_frames - CONFIG['frames']) // 2
            log_mel = log_mel[:, start:start + CONFIG['frames']]

        log_mel = log_mel.T # 转置为 (157, 128)
        if log_mel.shape != (CONFIG['frames'], CONFIG['n_mels']):
            return None
            
        return log_mel
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="CBGN Inference Script for Qt UI")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the .pth model file")
    parser.add_argument('--folder_path', type=str, required=True, help="Path to the folder containing audio files")
    parser.add_argument('--continuous', action='store_true', help="Run in continuous monitoring mode")
    args = parser.parse_args()

    # 1. 检查路径有效性
    if not os.path.exists(args.model_path):
        print(json.dumps({"error": f"模型文件不存在: {args.model_path}"}))
        sys.exit(1)
    if not os.path.exists(args.folder_path):
        print(json.dumps({"error": f"测试文件夹不存在: {args.folder_path}"}))
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载模型与配置
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        raw_configs = checkpoint['configs']
        configs = dict_to_namespace(raw_configs) if isinstance(raw_configs, dict) else raw_configs
        
        # 强制开启必要标记并实例化模型
        configs.use_attention = True
        model = CBGN(configs, seed=checkpoint.get('seed', 0), class_weights=None).to(device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            raise ValueError("无法在检查点中找到 model_state_dict，可能是不兼容的旧版模型。")
        
        model.eval()
    except Exception as e:
        print(json.dumps({"error": f"模型加载失败: {str(e)}"}))
        sys.exit(1)

    # 3. 扫描文件夹并获取所有有效的音频文件列表
    valid_extensions = ('.wav', '.flac', '.mp3')

    if args.continuous:
        import time
        processed_files = set()  # 用来记录已经处理过的文件
        print("DEBUG: 已进入持续监控模式，正在监听文件夹...", flush=True)

        while True:
            # 获取当前文件夹内的所有音频文件
            current_files = set(f for f in os.listdir(args.folder_path) if f.lower().endswith(valid_extensions))
            # 找出还没处理过的新文件
            new_files = current_files - processed_files

            for filename in new_files:
                file_path = os.path.join(args.folder_path, filename)
                feat = wav_to_mel1d(file_path)

                if feat is None:
                    res = {"filename": filename, "status": "failed", "message": "特征提取失败"}
                else:
                    x_tensor = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).to(device)

                    # 【关键修复1】：使用 torch.no_grad() 关闭梯度追踪，防止持续监控导致显存爆炸
                    with torch.no_grad():
                        f_map_h, _ = model.encoder_h(x_tensor)
                        f_map_h_att = model.attention_cbam(f_map_h)
                        z_h_flat = torch.flatten(f_map_h_att, 1)
                        z_h = model.bottleneck_h(z_h_flat)
                        logits = model.classifier_h(z_h)
                        probs = F.softmax(logits, dim=1)

                    # 【关键修复2】：增加 .detach() 安全断开计算图后，再转换为 numpy 数组
                    prob_arr = probs.detach().cpu().numpy()[0]
                    pred_class = int(np.argmax(prob_arr))
                    confidence = float(prob_arr[pred_class])

                    status_label = "正常" if pred_class == 0 else f"异常 (类别 {pred_class})"
                    res = {
                        "filename": filename,
                        "status": "success",
                        "prediction": status_label,
                        "confidence": round(confidence, 4)
                    }

                # 【关键】立刻输出这一条文件的结果（注意键名改为了 continuous_result）
                print(json.dumps({"continuous_result": res}, ensure_ascii=False), flush=True)
                # 记录为已处理
                processed_files.add(filename)

            # 休眠1秒，避免 CPU 占用过高
            time.sleep(1)
    else:

        audio_files = [f for f in os.listdir(args.folder_path) if f.lower().endswith(valid_extensions)]
        total_files = len(audio_files)

        if total_files == 0:
            print(json.dumps({"error": "文件夹中未找到支持的音频文件"}))
            sys.exit(1)

        results = []

        with torch.no_grad():
            for i, filename in enumerate(os.listdir(args.folder_path)):
                if not filename.lower().endswith(valid_extensions):
                    continue # 过滤非音频文件

                current_index = i + 1
                print(f"PROGRESS:{current_index}:{total_files}:{filename}", flush=True)

                file_path = os.path.join(args.folder_path, filename)
                feat = wav_to_mel1d(file_path)

                if feat is None:
                    results.append({"filename": filename, "status": "failed", "message": "特征提取失败"})
                    continue

                # 转换为模型所需的张量形状 (Batch, Channels, Height, Width) -> (1, 1, 157, 128)
                x_tensor = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).to(device)

                # 执行单样本推理 (提取自 CBGN.py 的 test_model 逻辑)
                f_map_h, _ = model.encoder_h(x_tensor)
                f_map_h_att = model.attention_cbam(f_map_h)
                z_h_flat = torch.flatten(f_map_h_att, 1)
                z_h = model.bottleneck_h(z_h_flat)
                logits = model.classifier_h(z_h)
                probs = F.softmax(logits, dim=1)

                prob_arr = probs.cpu().numpy()[0]
                pred_class = int(np.argmax(prob_arr))
                confidence = float(prob_arr[pred_class])

                # 根据你的 num_classes 映射诊断结果 (0通常为正常，1通常为异常)
                status_label = "正常" if pred_class == 0 else f"异常 (类别 {pred_class})"

                results.append({
                    "filename": filename,
                    "status": "success",
                    "prediction": status_label,
                    "confidence": round(confidence, 4)
                })

        # 4. 以 JSON 格式输出最终结果供 Qt 解析
        print(json.dumps({"results": results}, ensure_ascii=False))

if __name__ == '__main__':
    main()