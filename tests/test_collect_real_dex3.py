"""How the collector builds the Dex3 driver.

The two drivers do not take their hand side the same way -- ``Dex3HandDriver`` is
keyword-only (``*, hand_dir=...``) while ``DryRunDex3HandDriver`` is positional --
so passing it positionally works in every dry-run and sim2sim test and raises
``TypeError`` the first time someone attaches a real hand. That is the worst place
to find it: the dataset directory has already been created, the robot is powered,
and the operator is standing in the headset.
"""

from __future__ import annotations

import argparse

import pytest

from g1_bridge import collect_real


def _args(**over):
    ns = argparse.Namespace(
        dex3="double",
        dex3_dry_run=False,
        dex3_domain_id=None,
        dex3_interface=None,
        robot_interface="eno1",
        robot_io="zmq",
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_real_driver_gets_hand_dir_by_keyword(monkeypatch) -> None:
    seen = {}

    class SpyDriver:
        def __init__(self, *args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs

    monkeypatch.setattr("sim2real.utils.dex3.Dex3HandDriver", SpyDriver)
    collect_real._build_dex3(_args())
    assert seen["args"] == (), "hand_dir must not be passed positionally"
    assert seen["kwargs"]["hand_dir"] == "double"


def test_dds_interface_follows_the_robot_interface(monkeypatch) -> None:
    """The hand sits on the robot's own DDS network, so it must not keep the
    driver's own "eth0" default -- that is right onboard the G1 and wrong on every
    workstation, where it surfaces as an opaque "channel factory init error"."""
    seen = {}

    class SpyDriver:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("sim2real.utils.dex3.Dex3HandDriver", SpyDriver)
    collect_real._build_dex3(_args(dex3="right", robot_interface="enp4s0"))
    assert seen == {"hand_dir": "right", "interface": "enp4s0"}

    seen.clear()
    collect_real._build_dex3(
        _args(dex3="left", dex3_domain_id=3, dex3_interface="eth1", robot_interface="eno1")
    )
    assert seen == {"hand_dir": "left", "domain_id": 3, "interface": "eth1"}, (
        "an explicit --dex3-interface still wins"
    )


def test_a_bad_interface_names_the_alternatives(monkeypatch) -> None:
    class Boom:
        def __init__(self, **kwargs):
            raise Exception("channel factory init error.")

    monkeypatch.setattr("sim2real.utils.dex3.Dex3HandDriver", Boom)
    with pytest.raises(SystemExit) as excinfo:
        collect_real._build_dex3(_args(robot_interface="nope0"))
    message = str(excinfo.value)
    assert "nope0" in message and "interfaces on this machine" in message
    assert "--dex3-dry-run" in message and "--dex3 none" in message


def test_dry_run_driver_is_positional_and_real() -> None:
    driver = collect_real._build_dex3(_args(dex3_dry_run=True))
    assert driver.hand_dir == "double"
    driver.apply_grips(0.0, 1.0)          # the call the control loop makes every tick
    assert driver.last_grips == (0.0, 1.0)


def test_none_means_no_driver() -> None:
    assert collect_real._build_dex3(_args(dex3="none")) is None


@pytest.mark.parametrize("hand_dir", ["left", "right", "double"])
def test_every_choice_the_cli_offers_constructs(hand_dir) -> None:
    driver = collect_real._build_dex3(_args(dex3=hand_dir, dex3_dry_run=True))
    assert driver.hand_dir == hand_dir
