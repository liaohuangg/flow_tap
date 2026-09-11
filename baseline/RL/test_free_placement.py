"""Regression tests for score ordering and topology-independent placement."""
import unittest

from chiplet_model import Chiplet, LayoutProblem
from env import ChipletPlacementEnv, create_env_from_json


def problem():
    p = LayoutProblem()
    for name, width, height in [('A', 4, 4), ('B', 2, 2), ('C', 1, 1)]:
        p.add_chiplet(Chiplet(name, width, height))
    p.add_connection('B', 'C', 100)
    p.connection_graph['B']['C'].update(wireCount=100, EMIB_length=999)
    return p


def environment(p=None, **kwargs):
    return ChipletPlacementEnv(p or problem(), max_width=10, max_height=10,
                              grid_resolution=10, lenbase_samples=0, **kwargs)


class FreePlacementTests(unittest.TestCase):
    def test_score_and_weights(self):
        e = environment()
        self.assertEqual(e.pin_counts, {'A': 0, 'B': 100, 'C': 100})
        self.assertEqual(e.placement_order, ['B', 'C', 'A'])
        self.assertEqual(environment(placement_pin_weight=0).placement_order, ['A', 'B', 'C'])
        self.assertEqual(environment(placement_area_weight=0).placement_order, ['B', 'C', 'A'])
        with self.assertRaises(ValueError):
            environment(placement_area_weight=-1)

    def test_parallel_connections_count_each_endpoint(self):
        p = problem()
        p.all_connections = [dict(node1='B', node2='C', wireCount=3),
                             dict(node1='B', node2='C', wireCount=7)]
        e = environment(p)
        self.assertEqual(e.pin_counts, {'A': 0, 'B': 10, 'C': 10})
        order = e._chiplet_order()
        matrix = e._connection_matrix(order)
        b, c = order.index('B'), order.index('C')
        # Match the author's FastTM inputs: each JSON interface is represented
        # in both directions and CPLEX routes both directed demands.
        self.assertEqual(matrix[b][c], 10)
        self.assertEqual(matrix[c][b], 10)

    def test_positions_equal_exhaustive_geometry_check(self):
        e = environment()
        e.state.layout['B'] = Chiplet('B', 2, 2, x=4, y=4)
        actual = set(e._get_valid_positions('C'))
        expected = set()
        halo = e.microbump_halos['C']
        for rotation in (0, 1):
            width, height = (1, 1)
            upper_x = e.max_width - width - halo
            upper_y = e.max_height - height - halo
            xs = set([halo, upper_x] + [x for x in range(11) if halo <= x <= upper_x])
            ys = set([halo, upper_y] + [y for y in range(11) if halo <= y <= upper_y])
            for x in xs:
                for y in ys:
                    c = Chiplet('C', width, height, x=x, y=y)
                    if e._is_valid_placement(c, 'C')[0]:
                        expected.add((x, y, rotation))
        self.assertEqual(actual, expected)
        self.assertIn((halo, halo, 0), actual)  # connected but separated
        self.assertIn((9 - halo, 9 - halo, 0), actual)
        self.assertNotIn((4, 4, 0), actual)
        e.problem.connection_graph['B']['C']['EMIB_length'] = 0.01
        self.assertEqual(actual, set(e._get_valid_positions('C')))

    def test_free_episode_and_disconnected_chip(self):
        e = environment()
        targets = [
            ('B', (e.microbump_halos['B'], e.microbump_halos['B'], 0)),
            ('C', (9 - e.microbump_halos['C'], 9 - e.microbump_halos['C'], 0)),
            ('A', (5, e.microbump_halos['A'], 0)),
        ]
        for name, target in targets:
            self.assertEqual(e._get_current_chip_id(), name)
            actions = e.get_valid_actions()
            action = next(a for a in actions if e._decode_action_to_position(a) == target)
            _, reward, done, info = e.step(action)
            self.assertNotIn('error', info)
        self.assertTrue(done)
        self.assertEqual(e.state.placed, ['B', 'C', 'A'])
        self.assertEqual(e.get_valid_actions(), [])

    def test_exact_boundary_and_rotation(self):
        p = LayoutProblem()
        p.add_chiplet(Chiplet('A', 2.5, 1.5))
        e = environment(p)
        actions = e.get_valid_actions()
        positions = {e._decode_action_to_position(a) for a in actions}
        halo = e.microbump_halos['A']
        self.assertIn((7.5 - halo, 8.5 - halo, 0), positions)
        self.assertIn((8.5 - halo, 7.5 - halo, 1), positions)
        self.assertIn((halo, halo, 0), positions)
        self.assertFalse(e._is_valid_placement(Chiplet('A', 2.5, 1.5, x=0, y=halo), 'A')[0])
        self.assertFalse(e._is_valid_placement(Chiplet('A', 2.5, 1.5, x=8, y=0), 'A')[0])

    def test_microbump_halos_are_computed_and_cannot_overlap(self):
        p = LayoutProblem()
        p.add_chiplet(Chiplet('A', 1, 1))
        p.add_chiplet(Chiplet('B', 1, 1))
        e = environment(p)
        halo = e.microbump_halos['A']
        self.assertAlmostEqual(halo, 0.045)
        self.assertAlmostEqual(e.chiplets['A'].hubump, halo)

        a = Chiplet('A', 1, 1, x=halo, y=halo)
        a.hubump = halo
        e.state.layout['A'] = a

        # Bodies do not overlap, but their peripheral bump bands do.
        bump_overlap = Chiplet('B', 1, 1, x=1.05, y=halo)
        valid, reason = e._is_valid_placement(bump_overlap, 'B')
        self.assertFalse(valid)
        self.assertEqual(reason, 'microbump_overlap_with_A')

        # Touching bump-envelope boundaries is legal; crossing them is not.
        bump_touch = Chiplet('B', 1, 1, x=1 + 3 * halo, y=halo)
        self.assertTrue(e._is_valid_placement(bump_touch, 'B')[0])
        self.assertTrue(e._is_valid_placement(Chiplet('B', 1, 1, x=halo, y=2), 'B')[0])
        self.assertFalse(e._is_valid_placement(Chiplet('B', 1, 1, x=halo - 0.001, y=2), 'B')[0])

    def test_observation_is_finite_and_normalized(self):
        e = environment()
        observation = e.reset()
        self.assertEqual(observation.shape, (10,))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in observation))

    def test_auto_grid_is_capped_at_a_practical_action_dimension(self):
        p = LayoutProblem()
        p.add_chiplet(Chiplet('A', 1.001, 1.003))
        e = ChipletPlacementEnv(
            p,
            max_width=50,
            max_height=50,
            grid_resolution='auto',
            exact_action_slots=50000,
            lenbase_samples=0,
        )
        self.assertEqual(e.grid_resolution, 100)
        self.assertEqual(e.action_dim, 70000)
        self.assertEqual(e.grid_auto_info['mode'], 'capped')


class FootprintEnvelopeTests(unittest.TestCase):
    """The declared occupied envelope must drive placement.

    Priority is footprint_w/footprint_h, then hubump, then the halo recomputed
    from input connectivity. A declared footprint equal to the body is how a
    chiplet with no microbump ring (a DUMMY die) is expressed, and a hubump of
    exactly 0 counts as unset.
    """

    @staticmethod
    def problem(chiplets, connections=()):
        p = LayoutProblem()
        for chip in chiplets:
            p.add_chiplet(chip)
        for node1, node2, wire_count in connections:
            p.add_connection(node1, node2, wire_count)
            for source, target in ((node1, node2), (node2, node1)):
                p.connection_graph[source][target].update(wireCount=wire_count)
        return p

    def environment(self, p, **kwargs):
        kwargs.setdefault('max_width', 50.0)
        kwargs.setdefault('max_height', 50.0)
        kwargs.setdefault('grid_resolution', 50)
        kwargs.setdefault('lenbase_samples', 0)
        return ChipletPlacementEnv(p, **kwargs)

    def test_declared_footprint_is_used_verbatim(self):
        p = self.problem([Chiplet('A', 14.5, 31.4, footprint_w=14.68, footprint_h=31.58)])
        e = self.environment(p)
        self.assertEqual(e.footprint_sources['A'], 'json_footprint')
        self.assertEqual(e.footprint_of('A'), (14.68, 31.58))
        self.assertEqual(e.footprint_sizes['A'], (14.68, 31.58))
        halo_x, halo_y = e._halo_axes('A')
        self.assertAlmostEqual(halo_x, 0.09)
        self.assertAlmostEqual(halo_y, 0.09)
        self.assertAlmostEqual(e.microbump_halos['A'], 0.09)

    def test_footprint_equal_to_body_expresses_zero_halo(self):
        p = self.problem([Chiplet('A', 8.0, 12.0, footprint_w=8.0, footprint_h=12.0)])
        e = self.environment(p)
        self.assertEqual(e.footprint_of('A'), (8.0, 12.0))
        self.assertEqual(e._halo_axes('A'), (0.0, 0.0))
        # With no halo the body may sit exactly on the canvas corner.
        self.assertTrue(e._is_valid_placement(Chiplet('A', 8.0, 12.0, x=0.0, y=0.0), 'A')[0])
        self.assertIn((0.0, 0.0, 0), set(e._get_valid_positions('A')))

    def test_recomputed_envelope_keeps_the_body_off_the_corner(self):
        p = self.problem([Chiplet('A', 8.0, 12.0, footprint_w=8.0, footprint_h=12.0)])
        e = self.environment(p, footprint_mode='recompute')
        self.assertEqual(e.footprint_sources['A'], 'recomputed')
        self.assertAlmostEqual(e._halo_axes('A')[0], 0.045)
        self.assertFalse(e._is_valid_placement(Chiplet('A', 8.0, 12.0, x=0.0, y=0.0), 'A')[0])

    def test_declared_footprint_overrides_the_connectivity_derived_halo(self):
        p = self.problem(
            [Chiplet('A', 4.0, 4.0, footprint_w=4.2, footprint_h=4.2),
             Chiplet('B', 4.0, 4.0)],
            [('A', 'B', 4096)],
        )
        e = self.environment(p)
        self.assertEqual(e.footprint_of('A'), (4.2, 4.2))
        self.assertGreater(e.recomputed_halos['A'], 0.1)
        self.assertFalse(e.footprint_report()[0]['matches_connectivity'])
        self.assertTrue(any('A:' in message for message in e.footprint_warnings))

    def test_verify_footprint_raises_on_a_mismatch(self):
        p = self.problem([Chiplet('A', 2.0, 2.0, footprint_w=5.0, footprint_h=5.0)])
        with self.assertRaises(ValueError):
            self.environment(p, verify_footprint=True)

    def test_asymmetric_footprint_resolves_per_axis_and_rotates(self):
        p = self.problem([Chiplet('A', 10.0, 6.0, footprint_w=10.2, footprint_h=6.04)])
        e = self.environment(p)
        halo_x, halo_y = e._halo_axes('A', 0)
        self.assertAlmostEqual(halo_x, 0.10)
        self.assertAlmostEqual(halo_y, 0.02)
        self.assertAlmostEqual(e._halo_axes('A', 1)[0], 0.02)
        self.assertAlmostEqual(e._halo_axes('A', 1)[1], 0.10)
        self.assertAlmostEqual(e.footprint_of('A', 0)[0], 10.2)
        self.assertAlmostEqual(e.footprint_of('A', 0)[1], 6.04)
        self.assertAlmostEqual(e.footprint_of('A', 1)[0], 6.04)
        self.assertAlmostEqual(e.footprint_of('A', 1)[1], 10.2)

    def test_footprint_is_enforced_against_a_rotated_neighbour(self):
        # Two 10x6 bodies whose declared envelope is 10.2x6.04, so the halo is
        # (0.1, 0.02) unrotated and (0.02, 0.1) once rotated.
        p = self.problem([Chiplet('A', 10.0, 6.0, footprint_w=10.2, footprint_h=6.04),
                          Chiplet('B', 10.0, 6.0, footprint_w=10.2, footprint_h=6.04)])
        e = self.environment(p)
        e.state.layout['A'] = Chiplet('A', 10.0, 6.0, x=0.1, y=0.02)
        bounds = e._occupied_bounds(e.state.layout['A'], 'A')
        for actual, expected in zip(bounds, (0.0, 0.0, 10.2, 6.04)):
            self.assertAlmostEqual(actual, expected)

        rotated = Chiplet('B', 6.0, 10.0, rotation=1)
        rotated.x, rotated.y = 10.22, 0.10          # envelopes touch at x = 10.2
        bounds = e._occupied_bounds(rotated, 'B')
        for actual, expected in zip(bounds, (10.2, 0.0, 16.24, 10.2)):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(e._is_valid_placement(rotated, 'B')[0])

        rotated.x = 10.21                            # envelope crosses into A
        self.assertFalse(e._is_valid_placement(rotated, 'B')[0])

    def test_hubump_is_used_when_footprint_is_absent(self):
        p = self.problem([Chiplet('A', 4.0, 4.0, hubump=0.3)])
        e = self.environment(p)
        self.assertEqual(e.footprint_sources['A'], 'json_hubump')
        self.assertEqual(e.footprint_of('A'), (4.6, 4.6))

    def test_zero_hubump_without_a_footprint_falls_back_to_recompute(self):
        p = self.problem([Chiplet('A', 4.0, 4.0, hubump=0.0)])
        e = self.environment(p)
        self.assertEqual(e.footprint_sources['A'], 'recomputed')
        self.assertAlmostEqual(e._halo_axes('A')[0], 0.045)

    def test_footprint_modes(self):
        undeclared = self.problem([Chiplet('A', 2.0, 2.0)])
        with self.assertRaises(ValueError):
            self.environment(undeclared, footprint_mode='json_only')
        with self.assertRaises(ValueError):
            self.environment(undeclared, footprint_mode='bogus')

        declared = self.problem([Chiplet('A', 2.0, 2.0, footprint_w=5.0, footprint_h=5.0)])
        self.assertEqual(self.environment(declared).footprint_of('A'), (5.0, 5.0))
        legacy = self.environment(declared, footprint_mode='recompute')
        self.assertEqual(legacy.footprint_sources['A'], 'recomputed')
        self.assertAlmostEqual(legacy.footprint_of('A')[0], 2.09)
        self.assertEqual(self.environment(declared, footprint_mode='json_only').footprint_of('A'),
                         (5.0, 5.0))

    def test_json_input_and_export_round_trip(self):
        import json as json_module
        import shutil
        from pathlib import Path

        payload = {
            'case': 'probe',
            'chiplets': [
                {'name': 'A', 'width': 8.0, 'height': 12.0, 'power': 25.0,
                 'hubump': 0.0, 'footprint_w': 8.0, 'footprint_h': 12.0},
                {'name': 'B', 'width': 5.0, 'height': 5.0, 'power': 10.0,
                 'hubump': 0.045, 'footprint_w': 5.09, 'footprint_h': 5.09},
            ],
            'connections': [{'node1': 'A', 'node2': 'B', 'wireCount': 64}],
        }
        # Keep the scratch files inside the package so they are always writable.
        scratch = Path(__file__).resolve().parent / 'results' / 'test_footprint_probe'
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            source = scratch / 'probe.json'
            source.write_text(json_module.dumps(payload), encoding='utf-8')
            e = create_env_from_json(str(source), max_width=50.0, max_height=50.0,
                                     grid_resolution=50, lenbase_samples=0)
            self.assertEqual(e.footprint_of('A'), (8.0, 12.0))
            self.assertEqual(e.footprint_sources['A'], 'json_footprint')
            self.assertEqual(e.footprint_sources['B'], 'json_footprint')
            self.assertTrue(e._is_valid_placement(Chiplet('A', 8.0, 12.0, x=0.0, y=0.0), 'A')[0])

            e.state.layout['A'] = Chiplet('A', 8.0, 12.0, x=0.0, y=0.0)
            e.state.layout['B'] = Chiplet('B', 5.0, 5.0, x=20.0, y=0.0, rotation=1)
            e.state.placed = ['A', 'B']
            out = scratch / 'layout.json'
            e.save_layout_json(str(out))
            data = json_module.loads(out.read_text(encoding='utf-8'))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

        self.assertEqual(data['footprint_mode'], 'json_first')
        exported = {chip['id']: chip for chip in data['chiplets']}
        self.assertEqual(exported['A']['footprint_w'], 8.0)
        self.assertEqual(exported['A']['footprint_h'], 12.0)
        self.assertEqual(exported['A']['hubump_x'], 0.0)
        self.assertEqual(exported['A']['occupied_envelope_source'], 'json_footprint')
        self.assertEqual(exported['A']['occupied_bounds'], [0.0, 0.0, 8.0, 12.0])
        self.assertEqual(exported['B']['footprint_w'], 5.09)
        self.assertEqual(exported['B']['rotation'], 1)
        self.assertEqual(exported['B']['occupied_envelope_source'], 'json_footprint')

    def test_real_benchmark_cases_use_the_declared_envelope(self):
        """Every shipped example must resolve to its declared footprint."""
        import json as json_module
        from pathlib import Path

        examples = Path(__file__).resolve().parent / 'examples'
        for path in sorted(examples.glob('*.json')):
            payload = json_module.loads(path.read_text(encoding='utf-8'))
            chiplets = payload['chiplets']
            # Build the problem the same way the JSON loader does.
            e = create_env_from_json(str(path), max_width=50.0, max_height=50.0,
                                     grid_resolution=20, lenbase_samples=0)
            for chiplet in chiplets:
                name = chiplet.get('name') or chiplet.get('id')
                self.assertEqual(
                    e.footprint_sources[name], 'json_footprint',
                    f'{path.name}/{name} did not use its declared footprint',
                )
                self.assertAlmostEqual(e.footprint_of(name, 0)[0], float(chiplet['footprint_w']))
                self.assertAlmostEqual(e.footprint_of(name, 0)[1], float(chiplet['footprint_h']))


if __name__ == '__main__':
    unittest.main()
