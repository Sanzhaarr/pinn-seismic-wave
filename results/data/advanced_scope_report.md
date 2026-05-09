# Advanced Case Scope

This project deliberately keeps the main scientific claim on 2D acoustic wave propagation in heterogeneous media. Some advanced cases are useful as optional robustness checks, while others are better presented as future work.

| case | recommendation | status | why | dissertation_use |
| --- | --- | --- | --- | --- |
| Multiple seismic sources | implement_now | implemented_optional_case | Same 2D acoustic equation, higher wavefield complexity, strong dissertation value. | Robustness experiment: interacting wavefronts from several shots. |
| Velocity inversion for unknown velocity | diagnostic_now_full_method_future | implemented_apparent_velocity_diagnostic | Full inversion is a separate research problem, but an apparent-velocity diagnostic is useful and honest. | Discuss as feasibility evidence and future work, not as full inversion. |
| 3D wave propagation | future_work | not_in_main_claim | Needs much larger memory, runtime, and 3D visualization; full validation would dominate the thesis. | Mention as natural extension after 2D validation. |
| Elastic P-wave/S-wave equations | future_work | not_in_main_claim | Requires vector displacement, Lamé parameters, P/S mode conversion, and new baselines. | Mention as physics extension beyond acoustic approximation. |
| Anisotropic media | future_work | not_in_main_claim | Requires anisotropic stiffness/velocity tensors and a different residual formulation. | Mention as realistic subsurface extension after isotropic heterogeneous media. |

Recommended defense position:

Multiple sources and apparent-velocity diagnostics strengthen the current dissertation without changing the main equation. Full 3D, elastic, and anisotropic PINNs should be described as future work because each requires a substantially different physical model and validation pipeline.