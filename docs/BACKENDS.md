# Backend integration

CardiEP has one stable `EPBackend` protocol:

```python
class EPBackend(Protocol):
    name: str

    def available(self) -> bool: ...
    def simulate(self, request: EPSimulationRequest) -> EPSimulationResult: ...
    def calibrate(self, request: EPCalibrationRequest) -> EPCalibrationResult: ...
```

## Python plugin

A package can expose a backend through:

```toml
[project.entry-points."cardiep.backends"]
my-backend = "my_package.backend:MyBackend"
```

`cardiep doctor` discovers it automatically.

## External command wrapper

For engines that should remain isolated in their own environment, use `SubprocessEPBackend`:

```python
from cardiep import CardiEPService, SubprocessEPBackend

service = CardiEPService()
service.register_backend(
    SubprocessEPBackend(
        name="opencarp-local",
        command=[
            "python",
            "opencarp_cardiep_adapter.py",
            "--request",
            "{request}",
            "--output",
            "{output}",
        ],
    )
)
```

The wrapper receives a serialized CardiEP request and must emit a valid CardiEP result JSON. CardiEP invokes it without a shell.

## Recommended high-fidelity targets

### FEniCSx-beat

Best fit when a Python-native finite-element monodomain backend is desired. Keep FEniCSx, PETSc and MPI in the plugin environment rather than in CardiEP core.

Suggested mapping:

- CardiEP anatomy → DOLFINx mesh and conductivity tensor;
- `fibre_speed/sheet_speed/normal_speed` → calibrated conductivity mapping;
- roots/stimuli → FEniCSx-beat stimuli;
- cellular model → gotranx-generated ODE or another explicit ODE solver;
- solver state → activation/repolarization extraction;
- optional torso/lead-field module → ECG artifact.

### MonoAlg3D_C

Best fit for GPU/high-throughput monodomain verification.

Use CardiEP as the experiment contract and artifact manager, while the plugin translates geometry, tissue tags, cell-model configuration, stimulus/Purkinje network, and output cadence into MonoAlg3D configuration.

### openCARP

Best fit for mature research experiments requiring mono/bidomain, ionic model libraries, reaction-Eikonal/DREAM, lead-field ECG, and extensive tooling.

openCARP distribution terms are not treated as equivalent to CardiEP's AGPL license. The integration should remain an explicit external installation/adapter.

### Chaste / cbcbeat

Useful for validation, alternative PDE implementations, adjoint-enabled workflows, and comparison studies. Their dependency/runtime profile is too heavy for CardiEP core.

## Required backend behavior

A production backend should:

1. fail on unknown coordinate units;
2. record all solver versions and numerical settings;
3. preserve input hashes;
4. expose convergence/numerical warnings;
5. avoid embedding large arrays directly in API responses;
6. write outputs as immutable artifacts;
7. distinguish software validation, numerical validation, and empirical validation;
8. provide at least one analytic or manufactured-solution test;
9. declare whether output ECGs are torso/lead-field based or only proxies;
10. never claim clinical validation from software tests.
