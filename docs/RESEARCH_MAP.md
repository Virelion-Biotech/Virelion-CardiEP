# Research and open-source map

This file records the projects reviewed while expanding CardiEP and the ideas adopted from them. CardiEP's implementation is independently written; external projects remain under their original licenses.

| Project | Useful capability | What CardiEP adopts | Integration stance |
|---|---|---|---|
| Julia Camps Cardiac-Digital-Twin | Modular geometry → conduction → propagation → EP → ECG → discrepancy → sampling pipeline; Eikonal and reaction-Eikonal personalization | The same separation of scientific responsibilities, explicit parameter routing, ECG discrepancy as a first-class inverse-model component | Architectural reference; MIT |
| Cardiac-Digital-Twin-Purkinje | ECG/MRI ventricular activation twinning, realistic Purkinje workflows, monodomain verification | Explicit Purkinje/root timing contract and future network plugin boundary | Architectural/scientific reference; MIT |
| fenicsx-beat | Modern FEniCSx monodomain PDE/ODE splitting | Recommended Python high-fidelity plugin target | External plugin; MIT |
| MonoAlg3D_C | Extensible finite-volume monodomain, GPU execution | Recommended high-throughput verification backend | External plugin/command backend; MIT |
| openCARP + carputils | Mature multi-scale EP, mono/bidomain, ionic models, reaction-Eikonal/DREAM, lead-field ECG, reproducible experiment tooling | Experiment-bundle philosophy, explicit solver provenance, future lead-field backend | External academic installation; do not vendor |
| Chaste | Mature tested cardiac EP framework and model ecosystem | Cross-solver validation philosophy | External comparison backend; BSD-3-Clause |
| cbcbeat | Adjoint-enabled monodomain/bidomain cardiac EP | Future gradient/adjoint calibration target | External plugin; LGPL-3.0+ |
| pyCEPS | CARTO/EnSite EAM import and computational-model conversion | Keep vendor EAM parsing outside CardiEP; consume normalized calibration artifacts | ElectroTrace/adapter boundary; GPL-3.0+ |
| Cobiveco | Consistent biventricular coordinate system | Consume named ventricular coordinate fields for APD/region gradients | CardiAnatomy producer; Apache-2.0 |
| cardiac-geometries / LDRB | Geometry and rule-based fibres | Keep geometry/fibre generation in CardiAnatomy | Upstream anatomy tools; MIT |
| gotranx | ODE model/code generation | Future ionic-model plugin interface | Optional cellular backend |
| purkinje-learning-demo | Purkinje parameter learning from ECG using Eikonal + Bayesian optimization/ABC | Supports keeping network parameters explicit and inference outside the forward solver | Research reference; MIT |
| CardioMat | End-to-end patient-specific geometry, fibres, Purkinje and GPU monodomain workflow | Reinforces the need for one pipeline contract spanning geometry through simulation | Research reference |

## Deliberate non-copy decisions

- No openCARP source is vendored because its academic-use terms differ from CardiEP's license.
- No pyCEPS vendor parser is copied; ElectroTrace or a dedicated adapter should normalize EAM data.
- No FEniCSx/PETSc/CUDA dependency is imposed on the base package.
- No paper-specific inference script is embedded into the solver; CardiInfer owns general inference.
- No external project is presented as validation evidence for CardiEP itself.

## References

- https://github.com/juliacamps/Cardiac-Digital-Twin
- https://github.com/juliacamps/Cardiac-Digital-Twin-Purkinje
- https://github.com/finsberg/fenicsx-beat
- https://github.com/rsachetto/MonoAlg3D_C
- https://opencarp.org/
- https://github.com/Chaste/Chaste
- https://github.com/ComputationalPhysiology/cbcbeat
- https://github.com/medunigraz/pyCEPS
- https://github.com/KIT-IBT/Cobiveco
- https://github.com/ComputationalPhysiology/cardiac-geometries
- https://github.com/finsberg/ldrb
- https://github.com/finsberg/gotranx
- https://github.com/fsahli/purkinje-learning
- https://github.com/niccolobiasi/CardioMat
