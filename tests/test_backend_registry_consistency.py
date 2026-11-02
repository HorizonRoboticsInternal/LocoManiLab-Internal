"""One backend, several tables -- hold them together.

A backend name appears in: the g1_bridge registry (``_BACKEND_MODULES``), the
recording schema table (``modality.G1_BACKEND_TELEOP_KEYS``), and sim2real's
family table (``rl_policy.families``). The registry is the single place a
backend is *added*; the CLI lists derive from it. The other two are genuinely
per-backend data and cannot be derived -- so this test is what keeps them from
drifting apart, the failure mode that used to be silent (a backend registered
but missing from modality dies at construction; one missing from families
loses its command-space/checkpoint documentation without any error at all).

Also pinned here: the facts duplicated by design between a backend class and
its family entry. ``vla_action_space`` is deliberately a literal on the class
(it is whatever ``teleop_sample`` records -- the recording IS the training
target), so it must equal the family's ``command_space``; ``POLICY_CONFIGS``
is now *derived from* the family table, so equality is construction -- but the
test keeps asserting it so a future hand-edit of either side fails loudly.
"""

from __future__ import annotations

import unittest


class BackendRegistryConsistencyTest(unittest.TestCase):
    def setUp(self):
        from g1_bridge.backends import backend_names

        self.names = set(backend_names())
        self.assertTrue(self.names, "backend registry is empty")

    def test_modality_covers_every_backend(self):
        from g1_bridge.modality import G1_BACKEND_TELEOP_KEYS

        self.assertEqual(self.names, set(G1_BACKEND_TELEOP_KEYS))

    def test_families_cover_every_backend(self):
        from sim2real.rl_policy.families import family_names

        # families may list MORE than the bridge drives (it documents the
        # checkpoints/ tree), but every driveable backend must be in it.
        self.assertLessEqual(self.names, set(family_names()))

    def test_vla_action_space_matches_family_command_space(self):
        from g1_bridge.backends import get_backend_class
        from sim2real.rl_policy.families import get_family

        for name in sorted(self.names):
            cls = get_backend_class(name)
            self.assertEqual(
                cls.vla_action_space,
                get_family(name).command_space,
                f"backend {name}: class vla_action_space disagrees with "
                "families.command_space -- one of them was edited alone",
            )

    def test_policy_configs_match_family_paths(self):
        from g1_bridge.backends import get_backend_class
        from sim2real.rl_policy.families import get_family

        for name in sorted(self.names):
            cls = get_backend_class(name)
            self.assertIn(name, cls.POLICY_CONFIGS)
            self.assertEqual(
                cls.POLICY_CONFIGS[name], get_family(name).policy_config
            )

    def test_cli_choices_derive_from_registry(self):
        from g1_bridge.recording.cli import BACKENDS
        from g1_bridge.collect_real import SUPPORTED_BACKENDS

        self.assertEqual(self.names, set(BACKENDS))
        self.assertEqual(self.names, set(SUPPORTED_BACKENDS))

    def test_controller_jsons_exist(self):
        from g1_bridge.backends import get_backend_class
        from g1_bridge.config import CONFIG_DIR

        for name in sorted(self.names):
            path = CONFIG_DIR / get_backend_class(name).controller_json
            self.assertTrue(path.is_file(), f"{name}: missing {path}")


if __name__ == "__main__":
    unittest.main()
