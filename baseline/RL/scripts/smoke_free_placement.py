"""Run real benchmark placement episodes without requiring a trained policy."""
import json
import random
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env import create_env_from_json
from chiplet_model import get_adjacency_info
from unit import calculate_layout_utilization, visualize_layout_with_bridges


def main():
    import matplotlib
    matplotlib.use('Agg')
    from datetime import datetime
    out = ROOT / 'results' / ('free_smoke_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True)
    e = create_env_from_json(str(ROOT / 'examples' / 'cpu-dram.json'),
                             max_width=50, max_height=50, grid_resolution=50,
                             lenbase_samples=0, terminal_rlplanner_cost_scale=0)
    rng = random.Random(42)
    episodes = []
    best = None
    start = perf_counter()
    for episode in range(10):
        e.reset()
        total_reward = 0.0
        counts = []
        while e.state.remaining:
            actions = e.get_valid_actions()
            counts.append(len(actions))
            if not actions:
                break
            _, reward, done, info = e.step(rng.choice(actions))
            assert 'error' not in info, info
            total_reward += reward
            if done:
                break
        chips = list(e.state.layout.values())
        occupied = {
            chip_id: e._occupied_bounds(chip, chip_id)
            for chip_id, chip in e.state.layout.items()
        }
        assert all(left >= -1e-9 and bottom >= -1e-9 and right <= 50 + 1e-9 and top <= 50 + 1e-9
                   for left, bottom, right, top in occupied.values())
        occupied_items = list(occupied.items())
        for i, (chip_id, (left, bottom, right, top)) in enumerate(occupied_items):
            for other_id, (o_left, o_bottom, o_right, o_top) in occupied_items[i + 1:]:
                assert not (left < o_right - 1e-9 and right > o_left + 1e-9
                            and bottom < o_top - 1e-9 and top > o_bottom + 1e-9), \
                    f'microbump overlap: {chip_id}, {other_id}'
        success = len(chips) == e.num_chiplets
        util, area, _, _, _ = calculate_layout_utilization(e.state.layout)
        record = dict(episode=episode, success=success, placed=len(chips),
                      reward=total_reward, utilization_percent=util, bbox_area=area,
                      candidate_counts=counts)
        episodes.append(record)
        if success and (best is None or util > best['utilization_percent']):
            separated = [(a,b) for a,b in e.problem.connection_graph.edges()
                         if not get_adjacency_info(e.state.layout[a], e.state.layout[b])[0]]
            best = dict(record, separated_connections=separated,
                        chiplets=[dict(name=c.id, x=c.x, y=c.y, width=c.width,
                                      height=c.height, hubump=c.hubump,
                                      occupied_bounds=e._occupied_bounds(c, c.id))
                                  for c in chips])
            e.save_layout_json(str(out / 'layout.json'))
            visualize_layout_with_bridges(e.state.layout, e.problem,
                                          output_file=str(out / 'layout.png'), show_bridges=False)
    summary = dict(method='random legal actions; not PPO training', benchmark='cpu-dram',
                   seed=42, grid_resolution=50, canvas=[50,50],
                   order=e.placement_order, scores=e.placement_scores, pins=e.pin_counts,
                   microbump_halos=e.microbump_halos,
                   successes=sum(r['success'] for r in episodes), episodes=episodes,
                   best=best, seconds=perf_counter()-start, output=str(out))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
