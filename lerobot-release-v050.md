---
title: "LeRobot v0.5.0: XYZ" 
thumbnail: /blog/assets/lerobot-release-v050/thumbnail.png
authors:
- user: imstevenpmwork
- user: aractingi
- user: pepijn223
- user: CarolinePascal
- user: jadechoghari
- user: lilkm
- user: nepyope
- user: Nico-robot
- user: VirgileBatto
- user: thomwolf
---

# LeRobot v0.5.0: Draft

### Hardware
* Earth Rover added support
* OMX robot added support
* Consolidated SO100 and SO101 (including bi manual setup)
* Unitree G1 Humanoid support, locomotion & manipulation
* Added support for Can Bus motors (RobStride & Damaio)
* OpenArm robot support & OpenArm Mini
* Remote camera support via ZMQ
### Software & API
* Improved installation steps
* Docs have versioning now
* Remote rerun viewer added support
* 3rd-party policies compatibility
* Gr00t supported for async_inference
* Camera API got extended for supporting high frequency datasets
* Better typing + dependencies bumped
* Pytorch version bump to suppoert blackwell chips
* Transformers v5 migration!
* Remote rerun visualization
* Bunch of bug fixes & improvements
* Python3.12 bump
### AI & Policies
* Real time chunking added support
* X-VLA policy added
* SARM added
* Wall-X policy added
* PEFT added support
* Autoregressive VLAs PI0FAST policy re-introduced
### Datasets
* Streaming encoding for 0 wait between episodes recording
* 10x image training performance, encoding time 3x faster
* Improved cpu usage & dataset compression at recording time
* Added tool to convert image datasets to videos
* Plenty of fixes and improvements in the dataset tools
* Expose more options to the user for dataset recording and encoding
### Simulation Envs & Benchmarks
* Envhub added functionality
* LeIsaac × LeRobot EnvHub
* Nvidia IsaacLab Arena support
### Community
* Modernized readme
* Modernized Discord server
* Clean template for issues, PRs, contributing and security
* Automated labelling of GH tickets & PRs
### Other projects
* LeRobot visualizer got a refresh
* Task annotation space