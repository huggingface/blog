---
title: "Hugging Face Unveils Microduck, a Tiny $399 Biped Robot You Can Teach New Tricks"
thumbnail: /blog/assets/microduck/thumbnail.jpg
authors:
- user: thomwolf
- user: matthieu-lapeyre
  guest: true
  org: pollen-robotics
- user: apirrone
  guest: true
  org: pollen-robotics
- user: acrampette
  guest: true
  org: pollen-robotics
- user: cdeplanne
  guest: true
  org: pollen-robotics
- user: Anne-Charlotte
  guest: true
  org: pollen-robotics
---

# Hugging Face Unveils Microduck, a Tiny $399 Biped Robot You Can Teach New Tricks

The compact, pet-like biped robot is designed to make AI on real hardware accessible to everyone, and even fun. It ships with complex built-in behaviors and an open-source software stack that lets users create new AI behaviors through reinforcement learning in simulation and sim-to-real deployment on the robot.

<img src="/blog/assets/microduck/microduck-hero.jpg" style="max-width:100%; width:694px" />\
*Microduck is a 25 cm tall biped robot designed for physical AI and reinforcement learning. Credit: Pollen Robotics / Hugging Face.*

| | |
| :-- | :-- |
| **PRODUCT PAGE** | [http://pollen-robotics.com/microduck](http://pollen-robotics.com/microduck) |
| **PRESS KIT** | [https://drive.google.com/drive/folders/1_wf1LEw9x3SrYlvm62zPmtH20pdQw8-B](https://drive.google.com/drive/folders/1_wf1LEw9x3SrYlvm62zPmtH20pdQw8-B?usp=drive_link) |
| **LAUNCH VIDEO** | [https://www.youtube.com/watch?v=reiTh7K4KSc](https://www.youtube.com/watch?v=reiTh7K4KSc) |
| **GITHUB** | [https://github.com/pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) |
| **JOIN DISCORD** | [https://discord.com/invite/pollen-community-519098054377340948](https://discord.com/invite/pollen-community-519098054377340948) |
| **MICRODUCK AT A GLANCE** | 25 cm tall, less than 800 g<br>15 actuators across articulated legs, head, neck and grasping beak<br>Front camera, compact ToF LiDAR and 2 IMUs<br>Speaker, microphone, 2 NFC sensors, Wi-Fi, Bluetooth<br>Open-source SDK and sim-to-real training stack<br>Pre-order price: $399 before taxes and shipping |

**August 27, 2026 -** Hugging Face and Pollen Robotics today unveil their latest robot: Microduck, a tiny biped designed by the Bordeaux-based robotics team.

With 15 degrees of freedom packed into a 25 cm tall body weighing less than 800 g, Microduck can walk, sit, crouch, roller-skate, pick up objects with its articulated beak, and recover on its own from many common falls, all with a distinctive and endearing waddle.

Those behaviors are just a starting point to get you up and running out of the box with family and friends. Our goal is simple: show everyone that building AI on a real robot can be as easy and accessible as it already is for software AI models. Microduck's accompanying software tackles the entire robotics stack, from reinforcement learning to simulation and sim-to-real transfer.

It is a playful toy and an educational tool. Whether you want to demonstrate what physical AI is, test your own robotics ideas, or even build a team MVP, the sky is the limit, all at a "micro" price.

Microduck is also packed with sensors to let you free your builder creativity: a front camera, a compact 8x8 depth LiDAR, 2 IMUs, a speaker, microphone, 2 NFC sensors and of course Wi-Fi and Bluetooth connectivity. It’s a dream toy for AI builders.

Microduck has been designed from scratch so developers can teach it entirely new physical skills using reinforcement learning. The open-source stack is easy to use, tightly integrated, and covers everything from robot control and simulation to RL training and sim-to-real deployment. We plan to release the full stack on GitHub alongside the first robot shipments.

Microduck is available for pre-order today at an introductory price of $399 (before taxes and shipping). It comes in four colorways. First customer deliveries are targeted before Christmas 2026. An extension pack gives you even more accessories to experiment, play and develop with Microduck.

Microduck is Hugging Face and Pollen Robotics' second consumer robot, following the highly successful Reachy Mini, unveiled last summer. Reachy Mini has already been shipped to over 10,000 customers around the world.

Where Reachy Mini was primarily designed around **human-robot interaction**, with support for conversational AI models like ElevenLabs or OpenAI's Realtime API as well as vision models, the new Microduck is built around **action and movement**: interacting with the physical world and acquiring new physical skills. What both robots share is our commitment to building educational robots that let users have fun while also learning how to build physical AI applications and mastering ML/AI training and deployment techniques.

> "AI is moving from screens to the physical world. With Microduck, we want developers to be able to experiment with learning and movement on real hardware without needing an expensive robotics lab. And just like with models on Hugging Face, the best part will be seeing what the community builds and shares."\
> **Clem Delangue, co-founder and CEO, Hugging Face**

## Teach your robot new tricks

Using an AI training technique called reinforcement learning, you can teach your robot new tricks, known as "policies." Reinforcement learning (RL) works by trying things, failing, and trying again until the policy is good enough for the robot to succeed.

Running this training directly on a physical robot can be slow and impractical. It usually involves thousands or millions of iterations, and failed policies will make the robot fall, collide with its surroundings, and often break, or simply end up in a position that requires someone to intervene before a new training episode can begin, limiting both speed and efficiency.

The solution is to train the robot in simulation and transfer the learned policy to the real world, known as sim-to-real transfer.

Microduck's open-source SDK walks users through these techniques and includes virtual training environments, reinforcement learning training scripts and tools, and a tested sim-to-real workflow. The SDK lets developers train policies in simulation before transferring them to the physical robot.

Sim-to-real transfer is not always straightforward. The Pollen Robotics team has invested considerable effort in improving this step, with the goal of making behaviors trained in simulation transfer efficiently and reproducibly to Microduck in as many cases as possible.

A typical development loop looks like this: **train in simulation (on your computer), deploy on the real-world robot, observe the behavior, refine the simulation parameters, repeat the process, share the policy with the community.**

The SDK gives developers access to the robot's motors, camera, audio, IMUs, LiDAR and NFC interfaces, as well as the tools used to simulate and train it.

Developers will also be able to use Hugging Face infrastructure to train, store and share policies and environments.

<img src="/blog/assets/microduck/microduck-sim2real.jpg" style="max-width:100%; width:694px" />\
*Microduck behaviors can be trained in simulation and transferred to the physical robot using its reinforcement learning stack. Credit: Pollen Robotics / Hugging Face.*

Watch the sim-to-real demo: [microduck_sim2real.mp4](https://drive.google.com/file/d/1-c0w6vik8GcvCzIxfCDnyyrk7gyeMM9X/view?usp=drive_link)

Full simulation and sim-to-real footage is available in the press kit.

## Made to move. Ready to fall.

Microduck is designed around whole-body mobility, but its small size is every bit as deliberate as its 15 motors.

At 25 cm tall, 14 cm wide and under 800 g, Microduck is built for the messy reality of reinforcement learning: imperfect policies. A policy can make it stumble, topple, or move in unexpected ways. Its low mass and small footprint make those failures far easier to deal with than a large humanoid accidentally falling on its owner.

The 15 degrees of freedom provide enough range of motion to walk, sit, crouch, grasp, recover from many common falls, and even perform highly dynamic behaviors such as roller-skating. Self-recovery reduces the need for manual resets during repeated experiments.

Despite its small size, Microduck is packed with sensors. Two IMUs, one in the body and one in the head, provide motion and orientation data used for balance and body-state estimation.

A front camera lets Microduck see and track people and objects, while a compact LiDAR (8x8 time-of-flight matrix) adds range sensing for those interested in exploring localization, navigation and mapping.

Microduck's articulated beak is not only playful and expressive but doubles as a gripper. The robot can lower its whole body toward an object, pick it up with its beak and carry it, giving it a simple manipulation capability without the need for complex articulated arms.

> "Reinforcement learning means the robot is going to try things, get them wrong and sometimes fall. We designed Microduck around that reality: keep it small, keep it light, let it get back up when it can, and make the loop from simulation to the real robot as short and easy as possible."\
> **Matthieu Lapeyre, co-founder, Pollen Robotics**

<img src="/blog/assets/microduck/microduck-action-1.jpg" style="max-width:100%; width:417px" /><img src="/blog/assets/microduck/microduck-action-2.jpg" style="max-width:100%; width:245px" />

*With 15 degrees of freedom in a body weighing less than 800 g, Microduck can walk, crouch, get back on its feet from many common fall positions, manipulate objects with its articulated beak and even do roller-skating.*

**See Microduck walking, recovering, grasping and roller-skating:**[microduck_in_action_raw_footage.mp4](https://drive.google.com/file/d/1-xjwwTxOu-dHnfG_nvPs707ceAuBSToO/view?usp=sharing)

## Designed to be playful, built to have personality

Microduck was designed to feel like more than just a robotics platform: dynamic, lively and expressive, with behaviors that make it engaging even before users start customizing. Its compact proportions and deliberately playful gait give it a distinctive waddle that reinforces its duck-like character.

Microduck's camera, LiDAR and motion sensing let users create behaviors that react to people, objects and the surrounding environment, making interactions feel less scripted and more pet-like. A dedicated camera indicator, inspired by classic REC lights, turns on whenever the camera is active. At launch, Microduck will include perception-driven modes such as detecting and chasing a laser dot across the floor, much like a cat.

Microphones let Microduck listen, while a speaker gives it a voice. The first time Microduck wakes up, it speaks with a procedurally generated voice unique to that individual robot. The voice is tied to its hardware identity, so it stays with the same Microduck for life.

Microduck also has two NFC antennas, one in its head and another inside its beak. Developers can use NFC-tagged physical objects as part of an application, opening up possibilities for toys, accessories, collectibles, checkpoints, or objects that the robot recognizes when it picks them up.

## Extensible and customizable

Microduck can also be used with a laser pointer or even a game controller to let users easily walk it around, trigger movements and interact with the robot before writing any code. Note that this actually triggers under the hood various AI trained behaviors. The robot has also been taught a few autonomous behaviors to be able to interact with its surroundings without any controller. The robot is running all policies by default locally without access to the internet even though users can later customize and extend this setup.

Microduck is designed for long play and experimentation sessions. It uses a standard removable NP-F550 camera battery (2,600 mAh), so users can swap batteries instead of waiting for the robot to recharge. Battery life is approximately one hour, depending on use.

With more than one Microduck, users can race, play football, or let the robots interact with each other, turning the platform into something closer to a physical multiplayer game.

For developers, having several small, affordable robots also opens the door to multi-robot experiments, possibly with Reachy Mini and other types of robots as well.

<img src="/blog/assets/microduck/microduck-group-1.jpg" style="max-width:100%; width:372px" /><img src="/blog/assets/microduck/microduck-group-2.jpg" style="max-width:100%; width:287px" />

Microduck can be played individually or in groups, from robot football and races to multi-robot experiments.

> "Continuing Hugging Face and Pollen Robotics' journey to offer a more positive and open vision of AI, we are super proud to unveil Microduck, a tiny robot that is both playful and educational. Microduck will let you bring your family and friends into the world of AI while also allowing you to learn and experiment with training reinforcement learning policies in its open-source software stack. State of the art doesn't have to be boring."\
> **Thomas Wolf, co-founder and CSO, Hugging Face**

## Built by Pollen Robotics, part of Hugging Face

Hugging Face acquired Pollen Robotics in April 2025, bringing its Bordeaux-based robotics team and open-source hardware expertise into the Hugging Face ecosystem.

Together, the teams combine Pollen Robotics' experience building physical robots with Hugging Face's open-source AI community and infrastructure.

Microduck is intended as both a product and a platform for the AI builder community. The public repository will open at launch at https://github.com/pollen-robotics/microduck. The open-source SDK, simulation environment, reinforcement learning training tools and sim-to-real workflow will be released there before the first robots ship.

Developers can join the Pollen Robotics Discord community to follow development and software release updates: [https://discord.com/invite/pollen-community-519098054377340948](https://discord.com/invite/pollen-community-519098054377340948)

The goal is for new skills and applications to emerge from the community and be easy for Microduck owners to reproduce, modify and share.

## Price and availability

Microduck will be available for pre-order beginning August 27, 2026 at [http://pollen-robotics.com/microduck](http://pollen-robotics.com/microduck). Mass production is currently underway.

**Pre-order price: $399 before taxes and shipping**

**First deliveries:** targeted before Christmas 2026

**Markets: North America and Europe/UK at launch**

**Colorways:**

- Cream

- Graphite

- Lavender

- Sky

<img src="/blog/assets/microduck/microduck-colorways.jpg" style="max-width:100%; width:694px" />

*Microduck launches in four colorways: Sky, Graphite, Cream and Lavender. Credit: Pollen Robotics / Hugging Face.*

## Key specifications

| **SPEC** | **DETAIL** | **SPEC** | **DETAIL** |
|----|----|----|----|
| **Motors** | 15 | **Compute** | Rockchip RK3566 + AI accelerator |
| **Vision** | Wide angle front camera | **RAM** | 1 GB |
| **Motion sensing** | 2 IMUs | **Storage** | 32 GB |
| **Range sensing** | Compact LiDAR ToF 8x8 matrix | **Audio** | Microphones + speaker |
| **Physical interaction** | Articulated grasping beak | **NFC** | 2 antennas: head + beak |
| **Connectivity** | Wi-Fi / Bluetooth | **Battery** | Removable NP-F550 camera battery, 2600 mAh, approx. 1 hour runtime depending on use |
| **Dimensions** | 25 cm tall x 14 cm wide | **Weight** | 780 g |
| **Programming** | Open-source SDK https://github.com/pollen-robotics/microduck | **RL stack** | Open-source simulation, training and sim-to-real tooling |
| **Pre-order price** | $399 before taxes and shipping | **Target delivery** | **Before Christmas 2026** |

## About Hugging Face

Hugging Face is an open platform and community for machine learning, giving developers, researchers and organizations tools to build, train, share and deploy AI models, datasets and applications. The Hugging Face Hub brings together a broad ecosystem of open-source AI resources and collaborative tools used across research and industry. In 2025, Hugging Face expanded its work in physical AI by bringing Pollen Robotics into the company.

## About Pollen Robotics

Pollen Robotics is a Bordeaux-based robotics team known for open-source robots and developer platforms. Founded in 2016 and part of Hugging Face since 2025, the team is behind Reachy, Reachy Mini and Microduck.

## Press contact

**Press:** Thomas Wolf, Rémi Fabre\
**Email:** [thomas@huggingface.co](mailto:thomas@huggingface.co), remi.fabre@pollen-robotics.com\
**Phone / Signal / WhatsApp:** thomwolf
