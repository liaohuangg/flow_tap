from typing import Any
import torch
import os

class Checkpointer:
    def __init__(self, ckpt_path = None):
        self.ckpt_path = ckpt_path
        self.models = {}
        self.ckpt = {}

    def save(self, path_override = None):
        path = path_override or self.ckpt_path
        try:
            os.makedirs(os.path.dirname(path))
        except FileExistsError:
            pass
        models = {}
        for k, v in self.models.items():
            try:
                models[k] = v.state_dict()
            except:
                try:
                    models[k] = v.save()
                except:
                    models[k] = v
        torch.save(models, path)
    
    def register(self, models):
        self.models.update(models)

    def load(self, path_override = None, filter_keys=None):
        path = path_override or self.ckpt_path
        if not os.path.exists(path):
            print(f"no checkpoint at {path} found")
            return False
        print(f"loading checkpoint from {path}")
        map_location = None if torch.cuda.is_available() else torch.device("cpu")
        self.ckpt = torch.load(path, map_location=map_location)
        for k, v in self.ckpt.items():
            if (filter_keys is not None) and (k not in filter_keys):
                continue
            try:
                self.models[k].load_state_dict(v)
                print(f"successfully loaded state dict for {k}")
            except Exception as exc:
                if self._load_partial_state_dict(k, v):
                    print(f"partially loaded compatible state dict for {k}: {exc}")
                    continue
                try:
                    self.models[k].load(v)
                    print(f"successfully loaded {k}")
                except:
                    self.models[k] = v
                    print(f"successfully loaded {k}")
        return True

    def _load_partial_state_dict(self, key, state_dict):
        model = self.models.get(key)
        if model is None or not hasattr(model, "state_dict") or not isinstance(state_dict, dict):
            return False

        current = model.state_dict()
        compatible = {
            name: value
            for name, value in state_dict.items()
            if name in current and getattr(current[name], "shape", None) == getattr(value, "shape", None)
        }
        if not compatible:
            return False

        skipped = sorted(set(state_dict.keys()) - set(compatible.keys()))
        current.update(compatible)
        model.load_state_dict(current, strict=True)
        print(f"loaded {len(compatible)} compatible tensors for {key}; skipped {len(skipped)} tensors")
        if skipped:
            print("skipped incompatible tensors:")
            for name in skipped[:20]:
                src_shape = tuple(state_dict[name].shape) if hasattr(state_dict[name], "shape") else type(state_dict[name])
                dst_shape = tuple(current[name].shape) if name in current and hasattr(current[name], "shape") else None
                print(f"  {name}: checkpoint {src_shape}, model {dst_shape}")
            if len(skipped) > 20:
                print(f"  ... {len(skipped) - 20} more")
        return True
