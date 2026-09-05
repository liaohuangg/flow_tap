import sys
from pathlib import Path

_DIFFUSION_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DIFFUSION_DIR.parent
for _path in (_REPO_ROOT, _DIFFUSION_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import utils
import torch
import hydra
import models
from omegaconf import OmegaConf, open_dict
import legalization
import analysis_utils
import common
import os
import time
import wandb
import pickle
import policies
from train_graph_thermal import (
    _build_thermal_model_from_ckpt,
    _load_thermal_checkpoint,
    _thermal_output_to_grid_and_avg,
    _thermal_rasterize,
    _denorm_temp_k,
)

CORE_METRIC_KEYS = (
    "idx",
    "thermal_max_c",
    "thermal_mean_c",
    "thermal_avg_head_c",
    "hpwl_ratio",
    "hpwl_rescaled",
    "legality_2",
    "macro_legality",
    "bbox_area_ratio",
    "model_time",
    "generation_time",
    "eval_time",
)


def _core_metrics(metrics):
    return {key: metrics[key] for key in CORE_METRIC_KEYS if key in metrics}


def cost(output_metrics):
    """
    Returns dict with cost function(s) for hyperparam sweep
    """
    legality_target = 0.995
    macro_legality_target = 0.998
    legality_temp = 0.001
    hpwl = torch.tensor(output_metrics["hpwl_rescaled"]).mean()
    
    legality = torch.tensor(output_metrics["legality_2"]).mean()
    legality_cost_factor = 1 + 10 * torch.nn.functional.relu((legality_target - legality)/legality_temp)
    
    macro_legality = torch.tensor(output_metrics["macro_legality"]).mean()
    macro_legality_cost_factor = 1 + 10 * torch.nn.functional.relu((macro_legality_target - macro_legality)/legality_temp)
    
    full_cost = (legality_cost_factor * hpwl).item()
    macro_cost = (macro_legality_cost_factor * hpwl).item()
    costs = {
        "cost": full_cost,
        "macro_cost": macro_cost,
    }
    return costs


class ThermalEvaluator:
    def __init__(self, thermal_cfg, device):
        self.device = device
        self.ckpt_path = thermal_cfg.get("ckpt", "none")
        if self.ckpt_path in (None, "", "none"):
            raise ValueError("eval_thermal.py requires +thermal.ckpt=<thermal guidance checkpoint>")
        self.grid_size = int(thermal_cfg.get("grid_size", 128))
        self.rect_sharpness = float(thermal_cfg.get("rect_sharpness", 80.0))
        ckpt = _load_thermal_checkpoint(self.ckpt_path)
        self.stats = ckpt.get("stats") if isinstance(ckpt.get("stats"), dict) else None
        self.model = _build_thermal_model_from_ckpt(ckpt, device)

    @torch.no_grad()
    def __call__(self, x_sample, cond):
        x_batch = x_sample.unsqueeze(0).to(self.device)
        power_grid, layout_grid, total_power = _thermal_rasterize(
            x_batch,
            cond,
            grid_size=self.grid_size,
            rect_sharpness=self.rect_sharpness,
            stats=self.stats,
        )
        temp, avg_temp = _thermal_output_to_grid_and_avg(self.model(power_grid, layout_grid, total_power))
        has_temp_stats = self.stats is not None and "temp_min" in self.stats and "temp_max" in self.stats
        if has_temp_stats:
            temp = _denorm_temp_k(temp, self.stats)
            if avg_temp is not None:
                avg_temp = _denorm_temp_k(avg_temp, self.stats)
        metrics = {
            "thermal_max_k": temp.max().detach().cpu().item(),
            "thermal_mean_k": temp.mean().detach().cpu().item(),
        }
        if avg_temp is not None:
            metrics["thermal_avg_head_k"] = avg_temp.mean().detach().cpu().item()
        if has_temp_stats:
            metrics.update(
                {
                    "thermal_max_c": (temp.max() - 273.15).detach().cpu().item(),
                    "thermal_mean_c": (temp.mean() - 273.15).detach().cpu().item(),
                }
            )
            if avg_temp is not None:
                metrics["thermal_avg_head_c"] = (avg_temp.mean() - 273.15).detach().cpu().item()
        return metrics


def save_outputs_with_thermal(
    x_in,
    cond,
    model,
    save_folder,
    thermal_evaluator,
    output_number_offset=0,
    policy="open_loop",
    policy_kwargs={},
    preprocess_fn=None,
    postprocess_fn=None,
    legalization_fn=None,
):
    idx = cond.file_idx if "file_idx" in cond else output_number_offset
    placed_stem = utils.output_case_stem(cond, idx, "placed")
    sample_stem = utils.output_case_stem(cond, idx, "sample")
    x_in = torch.unsqueeze(x_in, dim=0).to(model.device)
    original_device = cond.x.device
    cond.to(model.device)
    metrics = {}
    metrics_special = {}

    t0 = time.time()
    x_preprocessed, cond_preprocessed = preprocess_fn(x_in, cond) if preprocess_fn is not None else (x_in, cond)

    t1 = time.time()
    if cond_preprocessed.num_nodes == 0:
        sample = torch.zeros_like(x_preprocessed)
    else:
        if policy == "open_loop":
            sample, _, policy_metrics_special = policies.open_loop(
                1,
                model,
                x_preprocessed,
                cond_preprocessed,
                intermediate_every=0,
                save_videos=policy_kwargs["save_videos"],
            )
            metrics_special.update(policy_metrics_special)
        elif policy == "open_loop_clustered":
            sample, _ = policies.open_loop_clustered(1, model, x_preprocessed, cond_preprocessed, intermediate_every=0)
        elif policy == "iterative_clustering":
            sample, policy_metrics, policy_metrics_special = policies.iterative_clustering(
                1, model, x_preprocessed, cond_preprocessed, **policy_kwargs
            )
            metrics.update(policy_metrics)
            metrics_special.update(policy_metrics_special)
        elif policy == "random":
            sample = policies.random(1, x_preprocessed, cond_preprocessed)
        else:
            raise NotImplementedError
    t2 = time.time()

    image = utils.visualize_placement(sample[0], cond_preprocessed, plot_pins=True, plot_edges=False, img_size=(2048, 2048))

    if legalization_fn is not None:
        sample, legalization_metrics, legalization_metrics_special = legalization_fn(sample, cond_preprocessed)
        metrics.update(legalization_metrics)
        metrics_special.update(legalization_metrics_special)
        image_legalized = utils.visualize_placement(
            sample[0], cond_preprocessed, plot_pins=True, plot_edges=False, img_size=(2048, 2048)
        )
    else:
        image_legalized = image
    utils.debug_plot_img(image_legalized, os.path.join(save_folder, placed_stem))

    sample_unprocessed = sample.detach().clone()
    sample, cond_postprocessed = postprocess_fn(sample, cond_preprocessed)

    sample = sample.squeeze(dim=0).detach().to(device=cond.x.device)
    sample = utils.postprocess_placement(sample, cond_postprocessed).cpu().numpy()
    save_file = os.path.join(save_folder, f"{sample_stem}.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(sample, f)
    utils.save_placement_json(sample, cond_postprocessed, save_folder, idx)
    t3 = time.time()

    hpwl_normalized, hpwl_rescaled = utils.hpwl_fast(sample_unprocessed[0], cond_preprocessed, normalized_hpwl=False)
    macro_hpwl_normalized, macro_hpwl_rescaled = utils.macro_hpwl(sample_unprocessed[0], cond_preprocessed, normalized_hpwl=False)
    legality = utils.check_legality_new(sample_unprocessed[0], x_in[0], cond_preprocessed, cond_preprocessed.is_ports, score=True)
    if "is_macros" in cond:
        macro_legality = utils.check_legality_new(
            sample_unprocessed[0], x_in[0], cond_preprocessed, (~cond_preprocessed.is_macros) | cond_preprocessed.is_ports, score=True
        )
    else:
        macro_legality = 0.0
    original_hpwl_normalized = utils.hpwl_fast(x_preprocessed, cond_preprocessed, normalized_hpwl=True)
    bbox_metric_values = utils.bbox_metrics(sample_unprocessed[0], cond_preprocessed, reference_x=x_preprocessed[0])

    thermal_metrics = thermal_evaluator(sample_unprocessed[0], cond_preprocessed)
    t4 = time.time()

    cond.to(original_device)

    all_metrics = {
        **thermal_metrics,
        "idx": idx,
        "hpwl_normalized": hpwl_normalized,
        "hpwl_rescaled": hpwl_rescaled,
        "macro_hpwl_normalized": macro_hpwl_normalized,
        "macro_hpwl_rescaled": macro_hpwl_rescaled,
        "legality_2": legality,
        "macro_legality": macro_legality,
        "original_hpwl_normalized": original_hpwl_normalized,
        "hpwl_ratio": hpwl_normalized / max(1e-12, original_hpwl_normalized),
        **bbox_metric_values,
        "model_time": t2 - t1,
        "generation_time": t3 - t0,
        "eval_time": t4 - t3,
        "model_vertices": cond_preprocessed.num_nodes,
        "model_edges": cond_preprocessed.num_edges,
    }
    return _core_metrics(all_metrics), metrics_special, image, image_legalized

@hydra.main(version_base=None, config_path="configs", config_name="config_eval")
def main(cfg):
    # Preliminaries
    OmegaConf.set_struct(cfg, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg.seed)
    thermal_cfg = dict(cfg.get("thermal", {}) or {})

    # Prepare legalization function
    if cfg.legalization.mode in [None, "none", "None", ""]:
        legalize_fn = None
    elif cfg.legalization.mode in ["scheduled", "standard"]:
        def legalize_fn(x, cond):
            return legalization.legalize(
                x, 
                cond,
                thermal_cfg=thermal_cfg,
                **cfg.legalization,
                )
    elif cfg.legalization.mode == "opt":
        def legalize_fn(x, cond):
            return legalization.legalize_opt(
                x, 
                cond,
                **cfg.legalization,
                )
    else:
        raise NotImplementedError(f"Unknown legalization mode: {cfg.legalization.mode}")
    # Prepare pre and post processing functions. Note that postprocess fns are applied in reverse order
    preprocess_fns = []
    postprocess_fns = []
    if cfg.cluster.is_cluster:
        def cluster_preprocess_fn(x, cond):
            cluster_cond, cluster_x = utils.cluster(cond, cfg.cluster.num_clusters, verbose=cfg.cluster.verbose, placements=x)
            return cluster_x, cluster_cond
        def cluster_postprocess_fn(x, cond):
            return utils.uncluster(cond, x, return_cond=True)
        preprocess_fns.append(cluster_preprocess_fn)
        postprocess_fns.append(cluster_postprocess_fn)
    elif cfg.cluster.cached_clusters:
        def cluster_postprocess_fn(x, cond):
            return utils.uncluster(cond, x, return_cond=True)
        postprocess_fns.append(cluster_postprocess_fn)
    if cfg.sc_halo != 1.0:
        def resize_standard_cells(x, cond):
            _, _, sc_mask = analysis_utils.get_masks(x, cond)
            is_resize = sc_mask.float()
            size_multiplier = (is_resize * cfg.sc_halo) + ((1-is_resize))
            cond.x = cond.x * size_multiplier.unsqueeze(dim=-1)
            return x, cond
        preprocess_fns.append(resize_standard_cells)
    if cfg.edge_dropout > 0.0: # used for debugging
        def edge_dropout(x, cond):
            x, cond = utils.edge_dropout(x, cond, cfg.edge_dropout)
            return x, cond
        preprocess_fns.append(edge_dropout)
    if cfg.macros_only:
        if cfg.cached_macros:
            postprocess_fns.append(utils.add_non_macros)
        else:
            preprocess_fns.append(utils.remove_non_macros)
            postprocess_fns.append(utils.add_non_macros)
    def preprocess_fn(x, cond):
        for preprocess_step in preprocess_fns:
            x, cond = preprocess_step(x, cond)
        return x, cond
    def postprocess_fn(x, cond):
        for i, postprocess_step in enumerate(reversed(postprocess_fns)):
            x, cond = postprocess_step(x, cond)    
        return x, cond

    # Preparing dataset
    train_set, val_set = utils.load_graph_data_with_config(cfg.task, train_data_limit = cfg.train_data_limit, val_data_limit = cfg.val_data_limit)
    sample_shape = val_set[0][0].shape
    dataloader = utils.GraphDataLoader(
        train_set, 
        val_set, 
        cfg.val_batch_size, 
        cfg.val_batch_size, 
        device,
        preprocess_fn = preprocess_fn,
        val_shuffle = False, # Don't shuffle validation set
        )
    with open_dict(cfg):
        if cfg.family in ["cond_diffusion", "continuous_diffusion", "flow_matching", "guided_diffusion", "skip_diffusion", "skip_guided_diffusion", "no_model"]:
            cfg.model.update({
                "num_classes": cfg.num_classes,
                "input_shape": tuple(sample_shape),
                "device": device,
            })
        else:
            raise NotImplementedError

    # Preparing model
    model_types = {
        "cond_diffusion": models.CondDiffusionModel,
        "continuous_diffusion": models.ContinuousDiffusionModel, 
        "flow_matching": models.FlowMatchingModel,
        "guided_diffusion": models.GuidedDiffusionModel,
        "skip_diffusion": models.SkipDiffusionModel,
        "skip_guided_diffusion": models.SkipGuidedDiffusionModel,
        "no_model": models.NoModel,
    }
    if cfg.implementation == "custom":
        model = model_types[cfg.family](**cfg.model).to(device)
    else:
        raise NotImplementedError

    # Prepare logger
    num_params = sum([param.numel() for param in model.parameters()])
    with open_dict(cfg):  # for eval/debugging
        cfg.update({
            "num_params": num_params,
            "train_dataset": dataloader.get_train_size(),
            "val_dataset": dataloader.get_val_size(),
        })
    outputs = [
        common.logger.TerminalOutput(cfg.logger.filter),
    ]
    if cfg.logger.get("wandb", False):
        wandb_run_name = f"{cfg.task}.{cfg.method}.{cfg.seed}" if not cfg.param_sweep else None
        wandb_output = common.logger.WandBOutput(wandb_run_name, cfg)
        if cfg.param_sweep:
            with open_dict(cfg):  # for eval/debugging
                cfg.update({
                    "method": f"{cfg.method}.{wandb_output._wandb.run.name}",
                })
        else:
            print("WARNING: param_sweep set to true but wandb disabled. Continuing anyways...")
        outputs.append(wandb_output)
    step = common.Counter()
    logger = common.Logger(step, outputs)

    # Create log and output directories
    log_dir = utils.output_log_dir(cfg)
    sample_dir = os.path.join(log_dir, "samples")
    checkpointer = common.Checkpointer(os.path.join(log_dir, "latest.ckpt"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    print(f"saving eval outputs to: {log_dir}")

    # Output config used
    utils.save_cfg(cfg, os.path.join(log_dir, "config.yaml"))
    utils.write_summary_metrics({}, os.path.join(log_dir, "metrics_summary"))
    print(OmegaConf.to_yaml(cfg))

    # Load checkpoint if exists. Here we only load the model
    checkpointer.register({
        "model": model,
    })
    checkpointer.load(
        utils.resolve_checkpoint_path(cfg, cfg.from_checkpoint)
    )

    # Start training
    print(f"model has {num_params} params")
    print(f"==== Start Eval on Device: {device} ====")
    thermal_evaluator = ThermalEvaluator(thermal_cfg, device)
    def report_eval_function(samples, x_val, cond_val):
        sample_metrics = utils.eval_samples(samples, x_val, cond_val)
        for idx, sample_metric in enumerate(sample_metrics):
            sample_metric.update(thermal_evaluator(samples[idx], cond_val))
        return sample_metrics

    if cfg.eval_samples > 0:
        print("generating evaluation report")
        t1 = time.time()
        utils.generate_report(
            cfg.eval_samples, 
            dataloader, 
            model, 
            logger, 
            policy = cfg.eval_policy_algorithm, 
            intermediate_every = cfg.show_intermediate_every,
            eval_function = report_eval_function,
            )
        logger.write()
        t2 = time.time()
        print(f"generated report in {t2-t1:.3f} sec")

    # output eval samples
    t3 = time.time()
    print("generating output samples")
    output_metrics = {}
    log_metrics = common.Metrics()
    for i in range(cfg.num_output_samples):
        x, cond = val_set[i]
        metrics, metrics_special, image, image_legalized = save_outputs_with_thermal(
            x, 
            cond, 
            model, 
            save_folder=sample_dir, 
            thermal_evaluator=thermal_evaluator,
            output_number_offset=0, 
            policy=cfg.eval_policy_algorithm,
            policy_kwargs=cfg.eval_policy,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
            legalization_fn=legalize_fn,
        )
        print(f"Finished sample {i+1} of {cfg.num_output_samples} \t {metrics}")
        t5 = time.time()
        logger.add({
            "reverse_samples": {
                **metrics,
                **metrics_special,
                "image": utils.logging_image(image_legalized, logger),
                "image_raw": utils.logging_image(image, logger),
                "time_elapsed": t5-t3,
            }
        })
        # update metrics
        for k, v in metrics.items():
            if k in output_metrics:
                output_metrics[k].append(v)
            else:
                output_metrics[k] = [v]
        log_metrics.add(metrics)
    if output_metrics:
        utils.dict_to_csv(output_metrics, os.path.join(log_dir,"metrics.csv"))
        if cfg.logger.get("wandb", False):
            for plot_keys in cfg.scatter_plots:
                x_name = plot_keys[0]
                y_name = plot_keys[1]
                if x_name in output_metrics and y_name in output_metrics:
                    scatter_plot = utils.plot_scatter(output_metrics[x_name], output_metrics[y_name], x_title=x_name, y_title=y_name)
                    logger.add({f"{x_name}_vs_{y_name}": scatter_plot})
        summary_metrics = log_metrics.result()
        sweep_metrics = cost(output_metrics)
        utils.write_summary_metrics(
            output_metrics,
            os.path.join(log_dir, "metrics_summary"),
            extra_metrics={f"sweep/{k}": v for k, v in sweep_metrics.items()},
        )
        logger.add(summary_metrics)
        logger.add(sweep_metrics, prefix = "sweep")
        logger.write()

if __name__=="__main__":
    main()
