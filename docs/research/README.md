# Robotics Research Links

This folder collects papers and links that are useful for connecting this WidowX ACT project with larger robot-learning directions: HAMSTER-style hierarchical action models, VLA policies, ACT imitation learning, ALOHA-style data collection, 3D manipulation policies, digital-twin benchmarks, and Trossen WidowX AI hardware.

Local paper PDFs are stored in [`papers/`](papers/).

## How This Relates To This Project

This repository currently targets a practical single-arm setup:

- Trossen Robotics WidowX AI arm
- optional Intel RealSense D405 wrist camera
- local demonstration recording and replay
- ACT-style imitation learning from recorded trajectories
- conservative policy replay with motion limits

The papers below provide a roadmap for future extensions:

- **ACT / ALOHA**: closest to the current local trainer and demonstration workflow.
- **OpenVLA / VLA methods**: larger language-conditioned policies that can be fine-tuned on robot data.
- **HAMSTER / ViLa / VILA-style planning**: high-level planning or latent-action structure above low-level control.
- **RVT / RVT-2**: 3D and multi-view manipulation policies, useful if the project expands beyond RGB wrist images.
- **RoboTwin**: synthetic and digital-twin data generation for scaling datasets.
- **ManiFlow**: flow/consistency-based action generation as a stronger policy class than a small ACT baseline.

## Core Papers

| Topic | Local PDF | Paper / Project | Code / Models | Notes |
| --- | --- | --- | --- | --- |
| HAMSTER | [hamster_2025.pdf](papers/hamster_2025.pdf) | [arXiv 2502.05485](https://arxiv.org/abs/2502.05485), [NVIDIA SRL page](https://research.nvidia.com/labs/srl/publication/li-2025-hamster/) | [Project page](https://hamster-robot.github.io/) | Hierarchical action models for open-world robot manipulation. Relevant as a high-level planning layer above this project's local ACT policy. |
| HAMSTER local copy | [Hamster.pdf](papers/Hamster.pdf) | Local paper copy | [Project page](https://hamster-robot.github.io/) | Existing local copy of the HAMSTER paper. Kept because it is directly related to the project's higher-level roadmap. |
| HAMSTER + ManiFlow thesis | [2025_MasterThesis_太田.pdf](papers/2025_MasterThesis_太田.pdf) | Local master's thesis | - | Thesis on hierarchical action models with 2D paths and consistency flow training for general robot manipulation. Useful bridge between HAMSTER, VLM path guidance, ManiFlow, and RoboTwin. |
| RVT | [rvt_2023.pdf](papers/rvt_2023.pdf) | [arXiv 2306.14896](https://arxiv.org/abs/2306.14896), [NVIDIA research page](https://research.nvidia.com/publication/2023-11_rvt-robotic-view-transformer-3d-object-manipulation) | [Project page](https://robotic-view-transformer.github.io/) | Multi-view transformer for 3D manipulation. Useful if adding multi-camera / 3D workspace reasoning. |
| RVT-2 | [rvt2_2024.pdf](papers/rvt2_2024.pdf) | [arXiv 2406.08545](https://arxiv.org/abs/2406.08545), [NVIDIA SRL page](https://research.nvidia.com/labs/srl/publication/goyal-2024-rvt-2/) | [Project page](https://robotic-view-transformer-2.github.io/) | More precise and faster few-demonstration 3D manipulation than RVT. Good reference for precise insertion/pick tasks. |
| OpenVLA | [openvla_2024.pdf](papers/openvla_2024.pdf) | [arXiv 2406.09246](https://arxiv.org/abs/2406.09246), [Hugging Face paper page](https://huggingface.co/papers/2406.09246) | [GitHub](https://github.com/openvla/openvla), [HF model](https://huggingface.co/openvla/openvla-7b) | Open-source 7B vision-language-action policy. Relevant for future language-conditioned control and fine-tuning. |
| ACT / ALOHA | [act_aloha_2023.pdf](papers/act_aloha_2023.pdf) | [arXiv 2304.13705](https://arxiv.org/abs/2304.13705), [RSS paper page](https://roboticsconference.org/program/papers/016/) | [Project page](https://tonyzhaozh.github.io/aloha/) | Main reference for Action Chunking Transformer imitation learning and low-cost teleoperation. Closest to this repository's local ACT trainer. |
| Mobile ALOHA | [mobile_aloha_2024.pdf](papers/mobile_aloha_2024.pdf) | [arXiv 2401.02117](https://arxiv.org/abs/2401.02117) | [Project page](https://mobile-aloha.github.io/), [GitHub](https://github.com/MarkFzp/mobile-aloha) | Extends ALOHA to whole-body mobile bimanual manipulation. Useful for data collection and co-training ideas. |
| ALOHA 2 | [aloha2_2024.pdf](papers/aloha2_2024.pdf) | [arXiv 2405.02292](https://arxiv.org/abs/2405.02292) | [Project page](https://aloha-2.github.io/) | Improved low-cost bimanual teleoperation hardware and simulator assets. Relevant for future dual-arm hardware inspiration. |
| RoboTwin | [robotwin_2025.pdf](papers/robotwin_2025.pdf) | [arXiv 2504.13059](https://arxiv.org/abs/2504.13059), [Hugging Face paper page](https://huggingface.co/papers/2504.13059) | [Project page](https://robotwin-platform.github.io/), [GitHub](https://github.com/RoboTwin-Platform/RoboTwin) | Dual-arm benchmark with generative digital twins. Useful for synthetic data, benchmark design, and simulation scaling. |
| ManiFlow | [maniflow_2025.pdf](papers/maniflow_2025.pdf) | [arXiv 2509.01819](https://arxiv.org/abs/2509.01819) | [Project page](https://maniflow-policy.github.io/) | General robot manipulation policy via consistency flow training. Relevant as a stronger action-generation approach than simple ACT. |

## VILA / ViLa Related

The name `VILA` is overloaded. These are the most relevant robotics entries:

| Topic | Local PDF | Paper / Project | Notes |
| --- | --- | --- | --- |
| ViLa planning | [vila_planning_2023.pdf](papers/vila_planning_2023.pdf) | [arXiv 2311.17842](https://arxiv.org/abs/2311.17842), [Hugging Face paper page](https://huggingface.co/papers/2311.17842) | Robotic vision-language planning with multimodal feedback. Relevant as a high-level planner above low-level policies. |
| VILAS low-cost VLA platform | [vilas_2026.pdf](papers/vilas_2026.pdf) | [arXiv 2605.02037](https://arxiv.org/abs/2605.02037) | Low-cost architecture for fine-tuning and deploying VLA models on accessible hardware. Relevant for turning this project into a cleaner data-collection/policy-deployment platform. |
| VILA view-invariant latent actions | [vila_view_invariant_latent_actions_2026.pdf](papers/vila_view_invariant_latent_actions_2026.pdf) | [arXiv 2601.02994](https://arxiv.org/abs/2601.02994), [Hugging Face paper page](https://huggingface.co/papers/2601.02994) | Learns view-invariant representations via latent actions. Relevant if the camera viewpoint changes or multiple cameras are used. |

## Trossen WidowX AI Hardware

Primary sources:

- [WidowX AI product page](https://www.trossenrobotics.com/widowx-ai)
- [Store page](https://store.trossenrobotics.com/products/widowx-ai)
- [Trossen Arm documentation](https://docs.trossenrobotics.com/trossen_arm/main/)
- [Trossen Arm downloads](https://docs.trossenrobotics.com/trossen_arm/main/downloads.html)
- [Trossen Arm GitHub driver](https://github.com/TrossenRobotics/trossen_arm)
- [OpenPI tutorial for Trossen AI hardware](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/openpi.html)

Key specs from Trossen documentation/product pages:

- 6 degrees of freedom
- 1.5 kg payload
- around 700 mm reach on product page; 0.769 m reach in Trossen Arm docs
- Ethernet controller communication
- variants: `base`, `leader`, `follower`
- follower variant includes an Intel RealSense D405 RGB-D camera with arm mount

Current project assumptions:

- default arm IP: `192.168.1.2`
- default variant in commands: `base`
- local Python environment: `.venv-trossen-ui`
- local recordings: `widowx_ai/recordings/`
- local checkpoints: `widowx_ai/models/`

## Adjacent Topics To Add Later

- Diffusion Policy
- LeRobot and Trossen LeRobot integration
- OpenPI / pi0 / pi0.5 on Trossen hardware
- RT-1 / RT-2 / RT-X / Open X-Embodiment
- LIBERO, RLBench, CALVIN, SimplerEnv benchmarks
- GR00T and other current VLA systems
