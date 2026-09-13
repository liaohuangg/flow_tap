from typing import Any
import torch
import os
import time
import uuid

class Checkpointer:
    def __init__(self, ckpt_path = None):
        self.ckpt_path = ckpt_path
        self.models = {}
        self.ckpt = {}

    def save(self, path_override = None):
        path = path_override or self.ckpt_path
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        models = {}
        for k, v in self.models.items():
            try:
                models[k] = v.state_dict()
            except:
                try:
                    models[k] = v.save()
                except:
                    models[k] = v
        # Directly overwriting a torch zip archive on WSL DrvFS (/mnt/*) can
        # intermittently fail with PyTorchFileWriter "Invalid argument". Write
        # a new file and atomically replace the destination so the previous
        # checkpoint also remains valid until the new archive is complete.
        last_error = None
        for attempt in range(3):
            temp_path = os.path.join(
                directory,
                f".{os.path.basename(path)}.{os.getpid()}.{uuid.uuid4().hex}.tmp",
            )
            try:
                torch.save(models, temp_path)
                if not os.path.isfile(temp_path) or os.path.getsize(temp_path) == 0:
                    raise RuntimeError(f"checkpoint temporary file is empty: {temp_path}")
                os.replace(temp_path, path)
                return
            except (OSError, RuntimeError) as error:
                last_error = error
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except OSError:
                    pass
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        raise last_error
    
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
