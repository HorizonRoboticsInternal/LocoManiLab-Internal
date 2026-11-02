# robosuite — G1 benchmark fork

This directory is the maintained robosuite 1.5.2 fork used by
[humanoid_benchmark](../README.md). It builds on
[ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite).
The upstream [LICENSE](LICENSE) and [AUTHORS](AUTHORS) are preserved.

## Supported robot catalog

The bundled robot models and runtime registry contain only `G1`,
`G1FixedLowerBody`, and `G1FloatingBody`. G1 meshes, joint/actuator order,
hand definitions, and controller configurations remain together in this fork.
Other robots' concrete models, grippers, mounts, configurations, demo tasks,
and assets have been removed. Optional `robosuite_models` and Mink example
controllers are not imported automatically.

Robot, mobile/legged runtime, environment, controller, camera, object, and
MJCF abstractions remain shared framework code. `NullGripper` is intentional:
G1's fingers already live inside the robot MJCF. The lightweight null bases
and G1's floating/passive support bases are retained as well.

## Installation and use

Install this fork with `python -m pip install -e robosuite_src` from the
benchmark checkout, then install the benchmark itself. Follow the parent
[installation and G1 examples](../README.md) for environment, controller,
and selected scene requirements. Import `robocasa` to register the benchmark
environments; the removed upstream Lift/Stack/two-arm demos are not registered.

The remaining `robosuite/scripts/` MJCF and setup utilities are optional
framework tools. Upstream documentation under `docs/` is retained as framework
reference; robot catalogs and examples there describe the original upstream
release. Current robot support and runnable benchmark commands are documented
in the parent repository.

## Citation
Please cite [**robosuite**](https://robosuite.ai) if you use this framework in your publications:
```bibtex
@inproceedings{robosuite2020,
  title={robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author={Yuke Zhu and Josiah Wong and Ajay Mandlekar and Roberto Mart\'{i}n-Mart\'{i}n and Abhishek Joshi and Soroush Nasiriany and Yifeng Zhu and Kevin Lin},
  booktitle={arXiv preprint arXiv:2009.12293},
  year={2020}
}
```
