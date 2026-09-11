"""
PPO推理脚本 - 芯片布局优化

使用训练好的模型找到最优布局
"""

import torch
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from .env import (
        DEFAULT_FOOTPRINT_MODE,
        FOOTPRINT_MODES,
        ChipletPlacementEnv,
        create_env_from_json,
    )
    from .train import ActorCritic
    from .chiplet_model import get_adjacency_info
    from .unit import (
        calculate_wirelength,
        calculate_manhattan_wirelength,
        calculate_layout_utilization,
        export_layout_result_json,
        visualize_layout_with_bridges,
    )
except ImportError:
    from env import (
        DEFAULT_FOOTPRINT_MODE,
        FOOTPRINT_MODES,
        ChipletPlacementEnv,
        create_env_from_json,
    )
    from train import ActorCritic
    from chiplet_model import get_adjacency_info
    from unit import (
        calculate_wirelength,
        calculate_manhattan_wirelength,
        calculate_layout_utilization,
        export_layout_result_json,
        visualize_layout_with_bridges,
    )


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _align_default_canvas_to_thermal(args) -> None:
    intp_size = float(args.thermal_intp_size) if args.thermal_intp_size is not None else 50.0
    if args.max_width is None:
        args.max_width = intp_size
    if args.max_height is None:
        args.max_height = intp_size


def load_model(model_path: str, env: ChipletPlacementEnv) -> ActorCritic:
    """加载训练好的模型"""
    model = ActorCritic(env.observation_dim, env.action_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ 模型已从 {model_path} 加载")
    return model


def run_inference(
    model: ActorCritic, 
    env: ChipletPlacementEnv, 
    deterministic: bool = False,
    seed: int = None
) -> Tuple[Dict, float, bool]:
    """
    运行单次推理
    
    Args:
        model: 训练好的模型
        env: 环境
        deterministic: 是否使用确定性策略（选择概率最大的动作）
        seed: 随机种子，用于复现结果
        
    Returns:
        (layout, total_reward, success)
    """
    # 设置随机种子
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    obs = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    max_steps = env.num_chiplets * 10
    
    while not done and step_count < max_steps:
        step_count += 1
        
        # 获取有效动作
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            break
        
        # 模型推理
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_logits, _ = model(obs_tensor)
            
            # 掩码无效动作
            mask = torch.ones(action_logits.shape[-1], device=device) * float('-inf')
            mask[valid_actions] = 0
            masked_logits = action_logits + mask
            
            if deterministic:
                # 选择概率最大的动作
                action = torch.argmax(masked_logits, dim=-1).item()
            else:
                # 采样动作
                probs = torch.softmax(masked_logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if "error" in info:
            break
    
    success = len(env.state.layout) == env.num_chiplets
    return env.state.layout, total_reward, success


def visualize_layout(
    layout: Dict, 
    problem, 
    save_path: str = None,
    title: str = "芯片布局"
):
    """可视化布局"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib 未安装，跳过可视化")
        return

    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    if not layout:
        print("布局为空，无法可视化")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 计算边界
    x_coords = [c.x for c in layout.values()] + [c.x + c.width for c in layout.values()]
    y_coords = [c.y for c in layout.values()] + [c.y + c.height for c in layout.values()]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # 绘制芯片
    colors = plt.cm.tab20(np.linspace(0, 1, len(layout)))
    
    for idx, (chip_id, chip) in enumerate(layout.items()):
        rect = patches.Rectangle(
            (chip.x, chip.y), chip.width, chip.height,
            linewidth=2, edgecolor='black', facecolor=colors[idx], alpha=0.6
        )
        ax.add_patch(rect)
        
        # 添加芯片ID标签
        cx = chip.x + chip.width / 2
        cy = chip.y + chip.height / 2
        ax.text(cx, cy, chip_id, ha='center', va='center', 
                fontsize=12, fontweight='bold')
    
    # 绘制邻接关系
    for chip1_id, chip2_id in problem.connection_graph.edges():
        if chip1_id in layout and chip2_id in layout:
            chip1 = layout[chip1_id]
            chip2 = layout[chip2_id]
            
            cx1 = chip1.x + chip1.width / 2
            cy1 = chip1.y + chip1.height / 2
            cx2 = chip2.x + chip2.width / 2
            cy2 = chip2.y + chip2.height / 2
            
            ax.plot([cx1, cx2], [cy1, cy2], 'r--', linewidth=1, alpha=0.5)
    
    # 设置坐标轴
    margin = 5
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Layout image saved to {save_path}")
    
    plt.show()


def save_layout_json(layout: Dict, save_path: str, metadata: Dict = None):
    """保存布局到JSON文件"""
    chiplets = {}
    for chip_id, chip in layout.items():
        chiplets[chip_id] = {
            "x": float(chip.x),
            "y": float(chip.y),
            "width": float(chip.width),
            "height": float(chip.height)
        }
    layout_data = {
        "metadata": metadata or {},
        "chiplets": chiplets,
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(layout_data, f, indent=2, ensure_ascii=False)
    
    print(f"Layout saved to {save_path}")


def calculate_metrics(layout: Dict, problem) -> Dict:
    """计算布局指标"""
    if not layout:
        return {}
    
    # 边界框
    x_coords = [c.x for c in layout.values()] + [c.x + c.width for c in layout.values()]
    y_coords = [c.y for c in layout.values()] + [c.y + c.height for c in layout.values()]
    
    bbox_width = max(x_coords) - min(x_coords)
    bbox_height = max(y_coords) - min(y_coords)
    bbox_area = bbox_width * bbox_height
    
    # 芯片总面积
    total_chip_area = sum(c.width * c.height for c in layout.values())
    
    # 利用率
    utilization = total_chip_area / bbox_area if bbox_area > 0 else 0
    
    # 检查邻接约束满足情况
    satisfied_adjacency = 0
    total_adjacency = 0
    
    # 遍历连接图中的所有边
    for chip1_id, chip2_id in problem.connection_graph.edges():
        total_adjacency += 1
        if chip1_id in layout and chip2_id in layout:
            chip1 = layout[chip1_id]
            chip2 = layout[chip2_id]
            is_adj, overlap_len, _ = get_adjacency_info(chip1, chip2)
            if is_adj and overlap_len >= 1.0:  # 假设min_overlap=1.0
                satisfied_adjacency += 1
    
    return {
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_area,
        "total_chip_area": total_chip_area,
        "utilization": utilization,
        "satisfied_adjacency": satisfied_adjacency,
        "total_adjacency": total_adjacency,
        "adjacency_rate": satisfied_adjacency / total_adjacency if total_adjacency > 0 else 0
    }


def calculate_footprint_metrics(env: ChipletPlacementEnv, layout: Dict) -> Dict:
    """Occupied-envelope (bump-expanded) bbox and utilization.

    Uses env._occupied_layout_stats(), the same definition as the observation and
    the per-step reward, so exported numbers match what the policy optimises.
    """
    if not layout:
        return {}
    bbox_w, bbox_h, bbox_area, occupied_area = env._occupied_layout_stats(layout)
    return {
        "footprint_bbox_width": bbox_w,
        "footprint_bbox_height": bbox_h,
        "footprint_bbox_area": bbox_area,
        "footprint_occupied_area": occupied_area,
        "footprint_utilization": occupied_area / bbox_area if bbox_area > 0 else 0.0,
    }


def calculate_rlplanner_metrics(env: ChipletPlacementEnv, layout: Dict) -> Dict:
    """计算导出用的温度/线长指标。"""
    try:
        rl_reward, metrics = env._calculate_rlplanner_terminal_reward(layout)
        metrics = dict(metrics)
        metrics["rlplanner_reward"] = rl_reward
        return metrics
    except Exception as exc:
        return {"rlplanner_reward_error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser(description="PPO芯片布局推理")
    parser.add_argument("json_path", type=str, help="输入JSON文件路径")
    parser.add_argument("--model", type=str, default="checkpoints/ppo_model.pt",
                        help="模型文件路径 (默认: checkpoints/ppo_model.pt)")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="运行次数，选择最优结果 (默认: 0)")
    parser.add_argument("--deterministic", action="store_true",
                        help="使用确定性策略")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子，用于复现结果（默认每次运行使用不同种子）")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="输出目录 (默认: results)")
    parser.add_argument("--grid_resolution", type=str, default=None,
                        help="网格分辨率，数字或 auto (默认: auto，按用例尺寸自动推断)")
    parser.add_argument("--max_width", type=float, default=None,
                        help="最大宽度 (默认: thermal_intp_size)")
    parser.add_argument("--max_height", type=float, default=None,
                        help="最大高度 (默认: thermal_intp_size)")
    parser.add_argument("--thermal_intp_size", type=float, default=None,
                        help="fastTM interposer size in mm (默认: 50.0)")
    
    parser.add_argument("--placement_area_weight", type=float, default=1.0, help="w1: chiplet area weight")
    parser.add_argument("--placement_pin_weight", type=float, default=1.0, help="w2: total incident pin count weight")
    parser.add_argument(
        "--max_auto_grid_resolution",
        type=int,
        default=100,
        help="auto 网格每个轴的最大分辨率（默认 100）",
    )
    parser.add_argument(
        "--footprint_mode",
        choices=FOOTPRINT_MODES,
        default=DEFAULT_FOOTPRINT_MODE,
        help="占据包络来源：json_first=优先用 footprint_w/h，再退回 hubump，最后按连线重算（默认）；"
             "json_only=必须由 JSON 声明，否则报错；recompute=始终按连线重算",
    )
    parser.add_argument(
        "--verify_footprint",
        action="store_true",
        help="声明包络与按连线重算值不一致时直接报错（默认只告警）",
    )
    args = parser.parse_args()
    # Use the same score weights as training.
    _align_default_canvas_to_thermal(args)
    
    print("=" * 70)
    print("PPO芯片布局推理")
    print("=" * 70)
    print(f"输入文件: {args.json_path}")
    print(f"模型文件: {args.model}")
    print(f"运行次数: {args.num_runs}")
    print(f"策略模式: {'确定性' if args.deterministic else '随机'}")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 创建环境
    env = create_env_from_json(
        args.json_path,
        grid_resolution=args.grid_resolution,
        max_width=args.max_width,
        max_height=args.max_height,
        placement_area_weight=args.placement_area_weight,
        placement_pin_weight=args.placement_pin_weight,
        max_auto_grid_resolution=args.max_auto_grid_resolution,
        thermal_intp_size=args.thermal_intp_size,
        min_overlap=0.5,
        placement_reward=1.0,
        adjacency_reward=1.0,
        extra_adjacency_reward=100,
        compact=3,
        min_wirelength_reward_scale=0,
        terminal_util_reward_scale=100.0,
        footprint_mode=args.footprint_mode,
        verify_footprint=args.verify_footprint,


    )
    
    print(f"\n环境信息:")
    print(f"  芯片数量: {env.num_chiplets}")
    print(f"  放置顺序: {env.placement_order}")
    print(f"  网格分辨率: {env.grid_resolution}")
    print(f"  占据包络来源: {env.footprint_mode}"
          f"{' (verify)' if env.verify_footprint else ''}")
    if env.footprint_warnings:
        print(f"  包络告警: {len(env.footprint_warnings)} 条（声明值与连线重算不一致）")
    if getattr(env, "grid_auto_info", None):
        print(f"  网格模式: {env.grid_auto_info.get('mode')}")
        if "geometry_step" in env.grid_auto_info:
            print(f"  几何步长: {env.grid_auto_info['geometry_step']}")
        if "warning" in env.grid_auto_info:
            print(f"  网格警告: {env.grid_auto_info['warning']}")
    
    # 加载模型
    model = load_model(args.model, env)
    
    # 运行多次推理
    print(f"\n开始推理...")
    print("-" * 70)
    
    best_layout = None
    best_reward = float('-inf')
    best_metrics = None
    success_count = 0
    max_utilization = 0.0
    
    for run in range(args.num_runs):
        # 为每次运行设置不同的种子：
        # - 如果用户提供了 --seed，则使用 args.seed + run（可复现多次运行）
        # - 如果未提供 --seed，则不设置种子（使用系统随机，每次运行有不同结果）
        if args.seed is not None:
            run_seed = args.seed + run
        else:
            run_seed = None

        layout, reward, success = run_inference(model, env, args.deterministic, seed=run_seed)
         
        
        if success:
            success_count += 1
            metrics = calculate_metrics(layout, env.problem)
            
 
            print(f"运行 {run+1}/{args.num_runs}: "
                  f"奖励={reward:.2f}, "
                  f"利用率={metrics['utilization']*100:.1f}%")
            
        
            # if metrics['utilization'] > max_utilization:
            #     max_utilization = metrics['utilization']
            #     best_layout = layout
            #     best_reward = reward
            #     best_metrics = metrics

            
            if reward > best_reward:
                best_reward = reward
                best_layout = layout
                best_metrics = metrics
        else:
            print(f"运行 {run+1}/{args.num_runs}: 失败")
    
    print("-" * 70)
    print(f"成功率: {success_count}/{args.num_runs} ({success_count/args.num_runs*100:.1f}%)")
    
    if best_layout is None:
        print("\n❌ 所有运行都失败了！")
        return
    
    # 显示最优结果
    print("\n" + "=" * 70)
    print("最优布局指标:")
    print("=" * 70)
    print(f"  总奖励: {best_reward:.2f}")
    print(f"  边界框: {best_metrics['bbox_width']:.2f} x {best_metrics['bbox_height']:.2f}")
    print(f"  边界框面积: {best_metrics['bbox_area']:.2f}")
    print(f"  芯片总面积: {best_metrics['total_chip_area']:.2f}")
    print(f"  利用率(硅体): {best_metrics['utilization']*100:.2f}%")
    footprint_metrics = calculate_footprint_metrics(env, best_layout)
    best_metrics.update(footprint_metrics)
    if footprint_metrics:
        print(f"  占据包络边界框: {footprint_metrics['footprint_bbox_width']:.2f} x "
              f"{footprint_metrics['footprint_bbox_height']:.2f}")
        print(f"  占据包络面积: {footprint_metrics['footprint_bbox_area']:.2f}")
        print(f"  占据包络面积(芯片合计): {footprint_metrics['footprint_occupied_area']:.2f}")
        print(f"  利用率(含 bump 包络): {footprint_metrics['footprint_utilization']*100:.2f}%")
    rlplanner_metrics = calculate_rlplanner_metrics(env, best_layout)
    best_metrics.update(rlplanner_metrics)
    print(f"  温度: {best_metrics.get('rlplanner_temperature', float('nan'))}")
    print(f"  CPLEX分配总线长: {best_metrics.get('rlplanner_total_wirelength', float('nan'))}")
    print(f"  CPLEX分配平均线长: {best_metrics.get('rlplanner_avg_wirelength', float('nan'))}")
    if best_metrics.get("rlplanner_reward_error"):
        print(f"  温度/线长计算警告: {best_metrics['rlplanner_reward_error']}")
    print("=" * 70)
    
    # 保存结果
    json_name = Path(args.json_path).stem
    layout_json_path = output_dir / f"{json_name}_layout.json"
    layout_img_path = output_dir / f"{json_name}_layout.png"
    
    export_layout_result_json(
        best_layout,
        env.problem,
        str(layout_json_path),
        metrics=best_metrics,
    )
    
    # 使用unit.py的可视化函数
    visualize_layout_with_bridges(
        best_layout,  # 使用最优布局
        env.problem,  # 问题定义
        output_file=str(layout_img_path),
        show_bridges=False,
        show_coordinates=True
    )
    
    print(f"\nInference complete.")


if __name__ == "__main__":
    main()
