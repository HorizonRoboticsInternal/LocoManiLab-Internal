"""
Placeholder living-room task.

``TidyLivingRoom`` has the full structure of a kitchen task -- it claims a
fixture by semantic type, spawns a manipulable object onto it, and defines a
success check + language instruction -- but ``_check_success`` always returns
``False``. It exists so the task plumbing (refs -> objects -> success -> lang)
is exercised end-to-end and a real success condition can be dropped in later.
"""

from robocasa.environments.living_room.living_room import LivingRoomEnv
from robocasa.models.fixtures import ArticulatedObject, FurnitureType


class TidyLivingRoom(LivingRoomEnv):
    def __init__(self, *args, **kwargs):
        # layout 1 (sofa/table/TV along x); single style for now
        kwargs.setdefault("layout_ids", 1)
        kwargs.setdefault("style_ids", 1)
        # spawn the robot in the clear, east of the coffee table
        kwargs.setdefault("robot_base_pos", (0.8, 0.0, 0.0))
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        # build the standard arena/furniture, then drop the converted USD
        # articulated asset (electric cooker) onto the coffee table top so its
        # hinge/collision/visuals can be inspected in-scene.
        super()._setup_model()

        # This task needs a coffee table + TV stand to place its articulated
        # assets on. Look them up by *type* (robust to furniture naming), and
        # fail with a clear message if the chosen layout doesn't have them
        # (e.g. layout 4 has side tables, not a coffee table) instead of a
        # cryptic KeyError. TidyLivingRoom pins layout 1 by default; only an
        # explicit --layout override can land here.
        ct = self.get_fixture(FurnitureType.COFFEE_TABLE)
        tv = self.get_fixture(FurnitureType.TV_STAND)
        if ct is None or tv is None:
            raise ValueError(
                f"TidyLivingRoom requires a layout with a coffee table AND a TV "
                f"stand (its default is layout 1); layout {self.layout_id} is "
                f"missing {'coffee_table' if ct is None else 'tv_stand'}. Run "
                f"`--env TidyLivingRoom` without overriding --layout/--style."
            )
        # world-space top surface of the coffee table (from its bbox region)
        top_z = ct.pos[2] + ct._regions["bbox"]["pz"][2]

        cooker = ArticulatedObject(name="electric_cooker", rng=self.rng)
        # The USD-converted mesh is Y-up; rotate -90 deg about x to stand it
        # upright (Z-up) with the lid facing up. (+90 stands it upright too but
        # upside-down, lid facing the table.)
        cooker.set_euler((-1.5708, 0.0, 0.0))
        # After that rotation the upright height is the mesh's original y-extent,
        # so rest it on the table by lifting by half that extent (reg_main y half).
        half_height = cooker._regions["main"]["py"][1]
        cooker.set_pos((ct.pos[0], ct.pos[1], top_z + half_height))

        self.furniture["electric_cooker"] = cooker
        self.electric_cooker = cooker

        # Second articulated asset. Its upright "lid-up" orientation is baked
        # into the asset (converted with --extra-rot-x -90), so no set_euler is
        # needed here -- it is born standing the right way up.
        a124 = ArticulatedObject(
            xml="fixtures/articulated/Asset124",
            name="asset124",
            rng=self.rng,
        )
        a124_half_height = -a124._regions["main"]["p0"][2]  # bbox bottom = -half_h
        a124.set_pos((ct.pos[0] + 0.4, ct.pos[1], top_z + a124_half_height))
        self.furniture["asset124"] = a124
        self.asset124 = a124

        # Third articulated asset (top-hinged front-hatch appliance) on the TV
        # stand top (looked up above). It is already upright in the asset (z is
        # height, no extra rotation needed), so set_pos only.
        tv_top_z = tv.pos[2] + tv._regions["bbox"]["pz"][2]
        a886 = ArticulatedObject(
            xml="fixtures/articulated/Asset886",
            name="asset886",
            rng=self.rng,
        )
        a886_half_height = -a886._regions["main"]["p0"][2]  # bbox bottom = -half_h
        a886.set_pos((tv.pos[0], tv.pos[1], tv_top_z + a886_half_height))
        self.furniture["asset886"] = a886
        self.asset886 = a886

        self.fixtures = self.furniture

    def _setup_furniture_references(self):
        # claim the coffee table by type (the concrete instance is chosen by the
        # layout) -- mirrors kitchen's register_fixture_ref
        self.coffee_table = self.register_furniture_ref(
            "coffee_table", dict(id=FurnitureType.COFFEE_TABLE)
        )
        self.sofa = self.register_furniture_ref("sofa", dict(id=FurnitureType.SOFA))

    def _get_obj_cfgs(self):
        # reuse a kitchen object ("apple") and sample it onto the sofa (the coffee
        # table top is occupied by the electric cooker)
        return [
            dict(
                name="apple",
                obj_groups="apple",
                placement=dict(
                    fixture=self.sofa,
                    size=(0.3, 0.3),
                ),
            )
        ]

    def _setup_scene(self):
        # The electric cooker's hinge needs noslip iterations to actually hold
        # the joint at the placed angle ("stays where placed" via frictionloss).
        # Robosuite's scene merge drops the fixture's <option> block, so we set
        # the runtime option here, after the sim is built.
        super()._setup_scene()
        self.sim.model.opt.noslip_iterations = max(
            int(self.sim.model.opt.noslip_iterations), 10
        )

    def _check_success(self):
        # placeholder: never succeeds. Replace with a real condition later
        # (e.g. apple moved off the coffee table onto the sofa).
        return False

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "tidy up the apple on the sofa"
        return ep_meta
