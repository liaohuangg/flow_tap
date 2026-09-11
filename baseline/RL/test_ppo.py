"""PPO regression tests; skipped when PyTorch is unavailable."""
import math
import unittest

try:
    import torch
    from chiplet_model import Chiplet, LayoutProblem
    from env import ChipletPlacementEnv
    from train import PPOTrainer, _set_global_seed
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class PPOTests(unittest.TestCase):
    @staticmethod
    def _make_trainer():
        problem = LayoutProblem()
        problem.add_chiplet(Chiplet("A", 1, 1))
        problem.add_chiplet(Chiplet("B", 1, 1))
        env = ChipletPlacementEnv(
            problem, max_width=3, max_height=3, grid_resolution=3,
            exact_action_slots=100, lenbase_samples=0,
            terminal_rlplanner_cost_scale=0,
        )
        return PPOTrainer(env, num_minibatches=2, update_epochs=2, target_kl=None)

    def test_masked_minibatch_update_has_scalar_samples(self):
        trainer = self._make_trainer()
        _, success, _ = trainer.collect_episode()
        self.assertTrue(success)
        self.assertEqual(trainer.buffer_size, 2)
        self.assertTrue(all(isinstance(value, float) for value in trainer.log_probs))
        self.assertTrue(all(isinstance(value, float) for value in trainer.values))
        self.assertEqual(trainer.compute_returns().shape, (2,))
        metrics = trainer.update()
        self.assertEqual(metrics["samples"], 2)
        self.assertEqual(metrics["optimizer_steps"], 4)
        self.assertTrue(math.isfinite(metrics["kl_div"]))

    def test_global_seed_reproduces_initial_policy_and_sampled_episode(self):
        _set_global_seed(3)
        first = self._make_trainer()
        first_weights = next(first.model.parameters()).detach().cpu().clone()
        first.collect_episode()

        _set_global_seed(3)
        second = self._make_trainer()
        second_weights = next(second.model.parameters()).detach().cpu().clone()
        second.collect_episode()

        self.assertTrue(torch.equal(first_weights, second_weights))
        self.assertEqual(first.actions, second.actions)

    def test_lightweight_checkpoint_omits_optimizer_state(self):
        import tempfile
        from pathlib import Path

        trainer = self._make_trainer()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "policy.pt"
            trainer.save(
                path,
                model_state_dict=trainer.snapshot_model_state(),
                include_optimizer=False,
                metadata={"seed": 4},
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.assertNotIn("optimizer_state_dict", checkpoint)
        self.assertEqual(checkpoint["metadata"]["seed"], 4)


if __name__ == "__main__":
    unittest.main()
