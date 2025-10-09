---
title: "LeRobot goes to driving school: World’s largest open-source self-driving dataset"
thumbnail: /blog/assets/193_l2d/lerobot-driver.gif
authors:
- user: sandhawalia
  guest: true
  org: yaak-ai
- user: cadene

---

# LeRobot goes to driving school

---

TL;DR of [L2D](https://huggingface.co/datasets/yaak-ai/L2D), the world's largest self-driving dataset!
- 90+ TeraBytes of multimodal data (5000+ hours of driving) from 30 cities in Germany
- 6x surrounding HD cameras and complete vehicle state: Speed/Heading/GPS/IMU
- Continuous: Gas/Brake/Steering and discrete actions: Gear/Turn Signals
- Environment state: Lane count, Road type (highway|residential), Road surface (asphalt, cobbled, sett), Max speed limit.
- Environment conditions: Precipitation, Conditions (Snow, Clear, Rain), Lighting (Dawn, Day, Dusk)
- Designed for training end-to-end models conditioned on natural language instructions or future waypoints
- Natural language instructions. F.ex ["When the light turns green, drive over the tram tracks and then through the roundabout"](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=yaak-ai%2FL2D&episode=82) for each episode
- [Future waypoints](#OpenStreetMap) snapped to OpenStreetMap graph, aditionally rendered in birds-eye-view
- Expert (driving instructors) and student (learner drivers) policies

---

State-of-the art [Vision Language Models](https://huggingface.co/blog/vlms) and Large Language Models are trained on open-source
image-text corpora sourced from the internet, which spearheaded the recent acceleration of open-source AI. Despite these
breakthroughs, the adoption of end-to-end AI within the robotics and automotive community remains low, primarily due to a
lack of high quality, large scale multimodal datasets [like OXE](https://robotics-transformer-x.github.io/).
To unlock the potential for robotics AI, Yaak teamed up with the LeRobot team at 🤗 and is excited to announce
**Learning to Drive (L2D)** to the robotics AI community. L2D is the **world’s largest multimodal dataset** aimed at
building an open-sourced spatial intelligence for the automotive domain with first class support for 🤗’s LeRobot
training pipeline and models. Drawing inspiration from the best practices of
[source version control](https://en.wikipedia.org/wiki/Version_control), Yaak also invites the AI
community to [search and discover](https://nutron-sandbox.yaak.ai/datasets/fcbb0dfd-40ae-4fd2-b023-7f300f35c5c7/search/list/session-logs?context=5s)
novel episodes in our entire dataset (\> 1 PetaBytes), and queue their collection for review to be merged into
future release ([R5+](#releases)).

<div style="margin-left: auto;
            margin-right: auto;
            width: 100%">

| Dataset  | Observation | State | Actions | Task/Instructions | Episodes | Duration (hr) | Size TB |
| :-----: | :---: | :---: | :---: | :---: | -----: | ---: | ---: |
| [WAYMO](https://waymo.com/open/data/perception/) | RGB (5x) | — | — | — | 2030 | 11.3 | 0.5* |
| [NuScenes](https://www.nuscenes.org/nuscenes#data-collection) | RGB (6x) | GPS/IMU | — | — | 1000 | 5.5 | 0.67* |
| [MAN](https://brandportal.man/d/QSf8mPdU5Hgj) | RGB (4x) | GPS/IMU | — | — | 747 | 4.15 | 0.17* |
| [ZOD](https://zod.zenseact.com/) | RGB (1x) | GPS/IMU/CAN | ☑️ | — | 1473 | 8.2 | 0.32* |
| [COMMA](https://github.com/commaai/comma2k19) | RGB (1x) | GPS/IMU/CAN | ☑️ | — | 2019 | 33 | 0.1 |
| [L2D (**R4**)](https://huggingface.co/datasets/yaak-ai/L2D) | RGB **(6x)** | GPS/IMU/CAN | ☑️ | ☑️ | **1000000** | **5000**\+ | **90+** |

</div>

<p align="center">
  <em> Table 1: Open source self-driving datasets (*excluding lidar and radar). <a href="https://arxiv.org/pdf/2305.02008">Source</a> </em>
</p>

L2D was collected with [identical sensor suites](#a2-data-collection-hardware) installed on 60 EVs operated by driving schools in 30 German
cities over the span of 3 years. The policies in L2D are divided into two groups — **expert policies** executed by
driving instructors and **student policies** by learner drivers. Both the policy groups include natural language instructions
for the driving task. For example, “When you have the right of way, *take the third exit from the roundabout, carefully driving over the pedestrian crossing*”.

| Expert policy — Driving instructor | Student policy — Learner driver |
| :----: | :----: |
| ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/193_l2d/image1.gif) | ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/193_l2d/image10.gif) |

<p align="center">
  <em> Fig 1: Visualization: <a href="https://nutron-sandbox.yaak.ai/datasets/">Nutron</a> (3 of 6 cameras shown for clarity)
  Instructions: “When you have the right of way, drive through the roundabout and take the third exit”.
  </em>
</p>

Expert policies have zero driving mistakes and are considered as optimal, whereas student policies have known
sub optimality (Fig 2).

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/193_l2d/image3.gif)
<p align="center">
  <em> Fig 2: Student policy with jerky steering to prevent going into lane of the incoming truck </em>
</p>

Both groups cover all driving scenarios that are mandatory for completion to obtain a driving license
[within the EU](https://bmdv.bund.de/SharedDocs/DE/Artikel/StV/Strassenverkehr/fahrerlaubnispruefung.html)
(German version), for example, overtaking, roundabouts and train tracks. In the release (See below [R3+](#releases)),
for suboptimal student policies, a natural language reasoning for sub-optimality will be included.
F.ex “*incorrect/jerky handling of the steering wheel in the proximity of in-coming traffic* ” (Fig 2\)

| **Expert: Driving Instructor** | **Student: Learner Driver** |
|:------------------------------:|:---------------------------:|
|![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/193_l2d/image13.png)|![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/193_l2d/image2.jpg)|
| **Expert policies** are collected when driving instructors are operating the vehicle. The driving instructors have at least **10K+ hours** of experience in teaching learner drivers. The expert policies group covers the same driving tasks as the student policies group. | **Student policies** are collected when learner drivers are operating the vehicle. Learner drivers have varying degrees of experience (**10–50 hours**). By design, learner drivers cover all **EU-mandated driving tasks**, from high-speed lane changes on highways to navigating narrow pedestrian zones. |

# L2D: Learning to Drive

L2D ([R2+](#releases)) aims to be the largest open-source self-driving dataset that empowers the AI community
with unique and diverse ‘episodes’ for training end-to-end spatial intelligence. With the inclusion of a full
spectrum of driving policies (student and experts), L2D captures the intricacies of safely operating a vehicle.
To fully represent an operational self-driving fleet, we include episodes with
diverse [environment conditions](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=yaak-ai%2FL2D&episode=4),
[sensor failures](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=yaak-ai%2FL2D&episode=36),
[construction zones](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=yaak-ai%2FL2D&episode=57) and
[non-functioning traffic signals](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=yaak-ai%2FL2D&episode=82).

Both the expert and student policy groups are captured with the [identical sensor setup](#a2-data-collection-hardware)
detailed in the table below. **Six RGB** cameras capture the vehicle’s context in 360o, and on-board GPS captures the vehicle
location and heading. An IMU collects the vehicle dynamics, and we read speed, gas/brake pedal, steering angle,
turn signal and gear from the vehicle’s CAN interface. We synchronized all modality types with the front left camera
(observation.images.front\_left) using their respective unix epoch timestamps. We also interpolated data points where
feasible to enhance precision (See Table 2.) and finally reduced the sampling rate to 10 hz.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/193_l2d/image4.png)
<p align="center">
  <em> Fig 3: Multimodal data visualization with Visualization: <a href="https://nutron-sandbox.yaak.ai/dataset/session-logs/a58deaa0-247e-4e86-b10a-8b3a3f42d646?begin=0&end=3073&context=5&offset=0&datasetId=fcbb0dfd-40ae-4fd2-b023-7f300f35c5c7">Nutron</a> (only 3 of 6 cameras shown for clarity) </em>
</p>

| Modality | LeRobotDataset v3.0 key | Shape | alignment\[tol\]\[strategy\] |
| :---- | :---- | :---- | :---- |
| image (x6) | [observation.images.front\_left](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L193)\[left\_forward,..\] | N3HW | [asof\[20ms\]\[nearest\]](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html) |
| speed | [observation.state.vehicle.speed](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L24) | N1 | [interp](https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.interpolate_by.html) |
| heading | [observation.state.vehicle.heading](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L25)\[heading\_error\] | N1 | [asof\[50ms\]\[nearest\]](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html) |
| GPS | [observation.state.vehicle.latitude](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L27)\[longitude/altitude\] | N1 | [asof\[50ms\]\[nearest\]](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html) |
| IMU | [observation.state.vehicle.acceleration\_x](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L30)\[y\] | N1 | [interp](https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.interpolate_by.html) |
| waypoints | [observation.state.vehicle.waypoints](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L84) | N2L | [asof\[10m\]\[nearest\]](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html) |
| timestamp | [observation.state.timestamp](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L95) | N1 | observation.images.front\_left |
| gas | [action.continous.gas\_pedal\_normalized](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L127) | N1 | [interp](https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.interpolate_by.html) |
| brake | [action.continous.brake\_pedal\_normalized](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L128) | N1 | [interp](https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.interpolate_by.html) |
| steering | [action.continous.steering\_angle\_normalized](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L129) | N1 | [interp](https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.interpolate_by.html) |
| turn signal | [action.discrete.turn\_signal](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L141) | N1 | [asof\[100ms\]\[nearest\]](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html) |
| gear | [action.discrete.gear](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L140) | N1 | [asof\[100ms\]\[nearest\]](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html) |
| language | [task.policy](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L106) | N1 | — |
| language | [task.instructions](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L113) | N1 | — |
| lane count | [observation.state.lanes](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L35) | N1 | [asof\[500ms\]\[nearest\]](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html) |
| road type | [observation.state.road](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L42) | N1 | [asof\[500ms\]\[nearest\]](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html) |
| road surface | [observation.state.surface](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L49) | N1 | [asof\[500ms\]\[nearest\]](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html) |
| max speed | [observation.state.max_speed](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L56) | N1 | [asof\[500ms\]\[nearest\]](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html) |
| precipitation | [observation.state.precipitation](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L63) | N1 | [asof\[1hr\]\[nearest\]](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html) |
| conditions | [observation.state.conditions](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L70) | N1 | [asof\[1hr\]\[nearest\]](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html) |
| lighting | [observation.state.lighting](https://huggingface.co/datasets/yaak-ai/L2D/blob/main/meta/info.json#L77) | N1 | [asof\[1hr\]\[nearest\]](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html) |

<p align="center">
  <em> Table 2: Modality types, LeRobot v3.0 key, shape and interpolation strategy. </em>
</p>

L2D follows the official German [driving task catalog](https://docs.google.com/spreadsheets/d/1phItCf1n6AVQqEIiP7lmfj0G07gR52iD7q93mMmrZio/edit?usp=sharing)
([detailed version](https://docs.google.com/document/d/1Xjg04uwXOjOpFuKmFZ-sqwDlcCnjPmbJeY0ACKwnLgU/edit?usp=sharing)) definition of driving tasks,
driving sub-tasks and task definition. We assign a unique Task ID and natural language instructions to all episodes.
The [LeRobot:task](https://github.com/huggingface/lerobot/pull/711) for all episodes is set to
“*Follow the waypoints while adhering to traffic rules and regulations*”. The table below shows a few sample episodes,
their natural language instruction, driving tasks and subtasks. Both expert and student policies have an identical
Task ID for similar scenarios, whereas the instructions vary with the episode.

| Episode | Instructions | Driving task | Driving sub-task | Task Definition | Task ID |
| :---- | :---- | :---- | :---- | :---- | :---- |
| [Visualization LeRobot](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=yaak-ai%2FL2D&episode=18) [Visualization Nutron](https://nutron-sandbox.yaak.ai/dataset/session-logs/8b3130ed-49ed-422c-b25c-a8408074e0e8?context=5&begin=118.359093&end=153.359093&offset=113.359093&datasetId=fcbb0dfd-40ae-4fd2-b023-7f300f35c5c7) | *Drive straight through going around the parked delivery truck and yield to the incoming traffic* | 3 Passing, overtaking | 3.1 Passing obstacles and narrow spots | This sub-task involves passing obstacles or navigating narrow roads while following priority rules. | 3.1.1.3a Priority regulation without traffic signs (standard) |
| [Visualization LeRobot](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=yaak-ai%2FL2D&episode=14) [Visualization Nutron](https://nutron-sandbox.yaak.ai/dataset/session-logs/c7f221eb-8019-43e2-ab74-102b1d739cff?context=5&begin=41.828257&end=75.828257&offset=36.828257&datasetId=fcbb0dfd-40ae-4fd2-b023-7f300f35c5c7) | *Drive through the unprotected left turn yielding to through traffic* | 4 Intersections, junctions, entering moving traffic  | 4.1 Crossing intersections & junctions  | This sub-task involves crossing intersections and junctions while following priority rules and observing other traffic. | 4.1.1.3a Right before left |
| [Visualization LeRobot](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=yaak-ai%2FL2D&episode=0)  [Visualization Nutron](https://nutron-sandbox.yaak.ai/dataset/session-logs/0cb2795c-6aa0-4205-82a3-93b412151499?context=5&datasetId=fcbb0dfd-40ae-4fd2-b023-7f300f35c5c7&begin=39.789473&end=54.789473&offset=34.789473) | *Drive straight up to the yield sign and take first exit from the roundabout* | 5 Roundabouts | 5.1 Roundabouts  | This sub-task involves safely navigating roundabouts, understanding right-of-way rules, and positioning correctly. | 5.1.1.3a With one lane |

<p align="center">
  <em> Table 3: Sample episodes in L2D, their instructions and Task ID derived from EU driving task catalog </em>
</p>

We automate the construction of the instructions and waypoints using the vehicle position (GPS),
[Open-Source Routing Machine](https://project-osrm.org/docs/v5.5.1/api/#match-service),
[OpenStreetMap](https://www.openstreetmap.org/#map=14/52.46346/13.40838&layers=D) and a Large Language Model (LLM)
([See below](#multimodal-search)). The natural language queries are constructed to closely follow the turn-by-turn
navigation available in most GPS navigation devices. The waypoints (Fig 4\) are computed by map-matching the raw GPS
trace to the OSM graph and sampling 10 equidistant points (orange) spanning 100 meters from the vehicles current location
(green), and serve as drive-by-waypoints.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/193_l2d/image6.gif" alt="A sample L2D episode">
</div>

<p align="center">
  <em> Fig 4: L2D 6x RGB cameras, waypoints (orange) and vehicle location (green)
  Instructions: drive straight up to the stop stop sign and then when you have right of way,
  merge with the moving traffic from the left </em>
</p>

# Search & Curation

| ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/193_l2d/image3.png) |  | ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/193_l2d/image11.png) |
| :----------------------: | :---: | :----------------------: |
| [Expert policies](https://api.mapbox.com/styles/v1/yaak-driving-curriculum/cm6up5as0019a01r5e6n33wmn.html?title=view&access_token=pk.eyJ1IjoieWFhay1kcml2aW5nLWN1cnJpY3VsdW0iLCJhIjoiY2txYzJqb3FwMWZweDJwbXY0M3R5cDAzYyJ9.vfIvUIpyVbBXqPjOjM7hEg&zoomwheel=true&fresh=true#12.29/51.96097/7.6215) |  | [Student policies](https://api.mapbox.com/styles/v1/yaak-driving-curriculum/cm6upwi4f01ay01pb8o939vq9.html?title=view&access_token=pk.eyJ1IjoieWFhay1kcml2aW5nLWN1cnJpY3VsdW0iLCJhIjoiY2txYzJqb3FwMWZweDJwbXY0M3R5cDAzYyJ9.vfIvUIpyVbBXqPjOjM7hEg&zoomwheel=true&fresh=true#12.91/51.96046/7.62953) |
| GPS traces from the expert policies collected from the driving school fleet. [Click here](https://api.mapbox.com/styles/v1/yaak-driving-curriculum/cm6up5as0019a01r5e6n33wmn.html?title=view&access_token=pk.eyJ1IjoieWFhay1kcml2aW5nLWN1cnJpY3VsdW0iLCJhIjoiY2txYzJqb3FwMWZweDJwbXY0M3R5cDAzYyJ9.vfIvUIpyVbBXqPjOjM7hEg&zoomwheel=true&fresh=true#12.29/51.96097/7.6215) to see the full extent of expert policies in L2D. |  | Student policies cover the same geographical locations as expert policies. [Click here](https://api.mapbox.com/styles/v1/yaak-driving-curriculum/cm6upwi4f01ay01pb8o939vq9.html?title=view&access_token=pk.eyJ1IjoieWFhay1kcml2aW5nLWN1cnJpY3VsdW0iLCJhIjoiY2txYzJqb3FwMWZweDJwbXY0M3R5cDAzYyJ9.vfIvUIpyVbBXqPjOjM7hEg&zoomwheel=true&fresh=true#12.91/51.96046/7.62953) to see the full extent of student policies in L2D. |

We collected the expert and student policies with a fleet of 60 KIA E-niro driving school vehicles
operating in 30 German cities, with an [identical sensor suite](#a2-data-collection-hardware). The multimodal logs collected with the
fleet are unstructured and void of any task or instructions information. To search and curate for episodes
we enrich the raw multimodal logs with information extracted through map matching the GPS traces with OSRM
and assigning [node](https://www.openstreetmap.org/node/5750552027#map=19/52.452765/13.384440) and
[way](https://www.openstreetmap.org/way/1144620134) tags from OSM ([See next section](#openstreetmap)).
Coupled with a [LLM](#multimodal-search), this enrichment step enables searching for episodes through
the natural language description of the task.

## OpenStreetMap

For efficiently searching relevant episodes, we enrich the GPS traces with turn information obtained by map-matching
the traces using [OSRM](https://project-osrm.org/). We additionally use the map-matched route and assign route features,
route restrictions and route maneuvers, collectively referred to as route tasks, to the trajectory using
[OSM](https://www.openstreetmap.org/) (See sample [Map](https://api.mapbox.com/styles/v1/yaak-driving-curriculum/cm6z3pjsh01df01qr6mhs636z.html?title=view&access_token=pk.eyJ1IjoieWFhay1kcml2aW5nLWN1cnJpY3VsdW0iLCJhIjoiY2txYzJqb3FwMWZweDJwbXY0M3R5cDAzYyJ9.vfIvUIpyVbBXqPjOjM7hEg&zoomwheel=true&fresh=true#17.74/48.776551/9.133615)).
[Appendix A1-A2](#appendix) provides for more details on the route tasks we assign to GPS traces.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/193_l2d/image12.png)

<p align="center">
  <em> Fig 5: Driving tasks assigned to raw GPS trace <a href="https://api.mapbox.com/styles/v1/yaak-driving-curriculum/cm6z3pjsh01df01qr6mhs636z.html?title=view&access_token=pk.eyJ1IjoieWFhay1kcml2aW5nLWN1cnJpY3VsdW0iLCJhIjoiY2txYzJqb3FwMWZweDJwbXY0M3R5cDAzYyJ9.vfIvUIpyVbBXqPjOjM7hEg&zoomwheel=true&fresh=true#15.75/48.770781/9.129999">(View map)</a> </em>
</p>

The route tasks which get assigned to the map-matched route, are assigned the beginning and end timestamps (unix epoch),
which equates to the time when the vehicle enters and exits the geospatial linestring or point defined by the task (Fig 6).

| Begin: Driving task (Best viewed in a separate tab)  | End: Driving task (Best viewed in a separate tab) |
|:--------------------------:|:----------------------:|
| ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/193_l2d/image8.png) | ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/193_l2d/image7.png) |

<p align="center">
    <em> Fig 6: Pink: GNSS trace, Blue: Matched route, tasks: Yield, Train crossing and Roundabout <a href="https://api.mapbox.com/styles/v1/yaak-driving-curriculum/cm6z3pjsh01df01qr6mhs636z.html?title=view&access_token=pk.eyJ1IjoieWFhay1kcml2aW5nLWN1cnJpY3VsdW0iLCJhIjoiY2txYzJqb3FwMWZweDJwbXY0M3R5cDAzYyJ9.vfIvUIpyVbBXqPjOjM7hEg&zoomwheel=true&fresh=true#17.81/48.776693/9.133048">(View Map) </em>
</p>

## Multimodal search

We perform semantic spatiotemporal indexing of our multimodal data with the route tasks as described in Fig 5\.
This step provides a rich semantic overview of our multimodal data. To search within the semantic space for representative episodes by
instructions, for example, “*drive up to the roundabout and when you have the right of way turn right*”,
we built a LLM-powered multimodal natural language search, to search within all our drive data (\> 1 PetaBytes) and retrieve matching episodes.

We structured the natural language queries (instructions) to closely resemble turn-by-turn navigation
available in GPS navigation devices. To translate instructions to route tasks,
we prompt the LLM with the instructions and [steer its output](https://pydantic.dev/articles/llm-intro) to a list of route features,
route restrictions, route maneuvers and retrieve episodes assigned to these route tasks. We perform a strict validation of the
output from the LLM with a [pydantic model to minimize hallucinations](https://pydantic.dev/articles/llm-validation).
Specifically we use [llama-3.3-70b](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md) and
steer the output to the schema defined by the pydantic model. To further improve the quality of the structured output,
we used approx 30 pairs of known natural language queries and route tasks for in-context learning.
[Appendix A. 2](#a.2-llm-prompts) provides details on the in-context learning pairs we used.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/193_l2d/hf-blog-search-roundabout-01.gif" alt="Natural language search" style="width: 80%;">
</div>

<p align="center">
    <em> Instructions: Drive up to the roundabout and when you have the right of way turn right </em>
</p>

## LeRobot

L2D on 🤗 is converted to [LeRobotDataset v2.1](https://github.com/huggingface/lerobot/pull/711) and [LeRobotDataset v3.0](https://huggingface.co/blog/lerobot-datasets-v3) format to fully leverage
the current and future models supported within [LeRobot](https://github.com/huggingface/lerobot).
The AI community can now build end-to-end self-driving models leveraging the state-of-the-art imitation learning
and reinforcement learning models for real world robotics like [ACT](https://tonyzhaozh.github.io/aloha),
[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), and [Pi0](https://github.com/huggingface/lerobot/pull/681).

Existing self-driving datasets (table below) focus on intermediate perception and planning tasks like 2D/3D object detection,
tracking, segmentation and motion planning, which require high quality annotations making them difficult to scale.
Instead L2D is focused on the development of end-to-end learning which learns to predict actions (policy) directly from sensor input (Table 1.).
These models leverage internet pre-trained [VLM](https://huggingface.co/blog/vlms) and [VLAM](https://huggingface.co/blog/pi0).

# Releases

Robotics AI models’ performances are bounded by the quality of the episodes within the training set.
To ensure the highest quality episodes, we plan a phased release for L2D. With each new release we add additional
information about the episodes. Each release **R1+** is a superset of the previous releases to ensure clean episode history.

1\. ***instructions***: Natural language instruction of the driving task
2\. ***task_id***: Mapping of episodes to EU mandated driving tasks Task ID
3\. ***observation.state.route*** : Information about lane count, turn lanes from [OSM](https://www.openstreetmap.org/)
4\. ***suboptimal***: Natural language description for the cause of sub-optimal policies

| HF | Nutron | Date | Episodes | Duration | Size | instructions  | task\_id | observation.state.route | suboptimal |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [R0](https://huggingface.co/datasets/yaak-ai/L2D/tree/R0) | [R0](https://nutron-sandbox.yaak.ai/collections/fcbb0dfd-40ae-4fd2-b023-7f300f35c5c7/300b7174-b6aa-4598-83e8-fc28cc5fcbe3/search/list/session-logs?context=5s) | March 2025 | 100 | 0.5+ hr | 9,5 GB | ☑️ |  |  |  |
| [R1](https://huggingface.co/datasets/yaak-ai/L2D/tree/R1) | [R1](https://nutron-sandbox.yaak.ai/collections/fcbb0dfd-40ae-4fd2-b023-7f300f35c5c7/1cb18573-f731-47b1-ae89-7ea2f026b8d0/search/list/session-logs?context=5s) | April 2025 | 1K | 5+ hr | 95 GB | ☑️ |  |  |  |
| [R2](https://huggingface.co/datasets/yaak-ai/L2D/tree/R2) | [R2](https://nutron-sandbox.yaak.ai/collections/fcbb0dfd-40ae-4fd2-b023-7f300f35c5c7/6e53636a-59ed-466b-8722-2c0b415f9bca/search/list/session-logs?context=5s) | May 2025 | 10K | 50+ hr | 0.5 TB | ☑️ |  | ☑️ | ☑️ |
| [R3](https://huggingface.co/datasets/yaak-ai/L2D/tree/main) | [R3](https://nutron-sandbox.yaak.ai/collections/fcbb0dfd-40ae-4fd2-b023-7f300f35c5c7/8930821d-b793-4885-b8c1-98cc10e20e81/search/list?context=5s) | Sept 2025 | 100K | 500+ hr | 5 TB | ☑️ |  | ☑️ | ☑️ |
| R4 | R4 | Nov 2025 | 1M | 5000+ hr | 90 TB | ☑️ | ☑️ | ☑️ | ☑️ |

<p align="center">
  <em> Table 5: L2D release dates </em>
</p>

The entire multimodal dataset collected by Yaak with the driving school fleet
[**is 5x larger**](https://api.mapbox.com/styles/v1/yaak-driving-curriculum/cm6up5as0019a01r5e6n33wmn.html?title=view&access_token=pk.eyJ1IjoieWFhay1kcml2aW5nLWN1cnJpY3VsdW0iLCJhIjoiY2txYzJqb3FwMWZweDJwbXY0M3R5cDAzYyJ9.vfIvUIpyVbBXqPjOjM7hEg&zoomwheel=true&fresh=true#12.29/51.96097/7.6215)
than the planned [release](#releases). To further the growth of L2D beyond R4, we invite the AI community to search
and uncover scenarios within our entire data collection and build a community powered open-source L2D.
The AI community can now search for episodes through [our natural language search](https://nutron-sandbox.yaak.ai/) and queue
their collection for review by the community for merging them into the upcoming releases. With L2D, we hope to unlock an
ImageNet moment for spatial intelligence.

<div align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/193_l2d/traffic-lights-left-turn-hf-01.gif" alt="Natural language search" style="width: 80%;">
</div>

<p align="center">
  <em> Fig 7: Searching episodes by natural language instructions</em>
</p>

# Using L2D with HF/LeRobot

For R0, R1 we recommend using `LeRobotDataset`, with `revision=[R0|R1]`, which can be used directly from the pypi release of LeRobot. For R2+, please follow installation [outlined here](https://huggingface.co/blog/lerobot-datasets-v3#install-lerobot-and-record-a-dataset) or install from main as below, as we recommend using [`StreamingLeRobotDataset`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/streaming_dataset.py#L43) as R3 is is [Dataset v3.0](https://huggingface.co/blog/lerobot-datasets-v3) format.

```
# uv for python deps
curl -LsSf https://astral.sh/uv/install.sh | sh
# install python version and pin it
uv init && uv python install 3.12.4 && uv python pin 3.12.4
# add lerobot to deps for R0, R1
uv add lerobot
# for R2+
GIT_LFS_SKIP_SMUDGE=1 uv add "git+https://github.com/huggingface/lerobot.git@main"
uv run python
>>> from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
# This will load 3 episodes=[0, 9999, 99999], to load all the episodes please remove it
>>> dataset = StreamingLeRobotDataset("yaak-ai/L2D", episodes=[0, 9999, 99999], streaming=True, buffer_size=1000)
>>> dataset.meta
LeRobotDatasetMetadata({
    Repository ID: 'yaak-ai/L2D',
    Total episodes: '100000',
    Total frames: '19042712',
    Features: '['observation.state.vehicle', 'observation.state.lanes', 'observation.state.road', 'observation.state.surface', 'observation.state.max_speed', 'observation.state.precipitation', 'observation.state.conditions', 'observation.state.lighting', 'observation.state.waypoints', 'observation.state.timestamp', 'task.policy', 'task.instructions', 'action.continuous', 'action.discrete', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'observation.images.left_forward', 'observation.images.front_left', 'observation.images.right_forward', 'observation.images.left_backward', 'observation.images.rear', 'observation.images.right_backward', 'observation.images.map']',
})',
```

# Closed Loop Testing

## LeRobot driver

For real world testing of the AI models trained with [L2D](#l2d-learning-to-drive) and LeRobot,
we invite the AI community to submit models for closed loop testing with a safety driver, starting summer of 2025.
The AI community will be able to queue their models for closed loop testing, on our fleet and choose the tasks
they’d like the model to be evaluated on and, for example, navigating roundabouts or parking.
The model would run in inference mode (Jetson AGX or similar) on-board the vehicle.
The models will drive the vehicle with LeRobot driver in two modes

1. drive-by-waypoints: “*Follow the waypoints adhering to driving rules and regulations*” given [observation.state.vehicle.waypoints](https://huggingface.co/datasets/yaak-ai/lerobot-driving-school/blob/main/meta/info.json#L196)
2. drive-by-language: “*Drive straight and turn right at the pedestrian crossing*”

# Additional Resources

- [Driving task catalog](https://docs.google.com/document/d/1Xjg04uwXOjOpFuKmFZ-sqwDlcCnjPmbJeY0ACKwnLgU/edit?tab=t.0) (Fahraufgabenkatalog)
- [Official German practical driving exam](https://www.pfep.de/#/)
- [Groq](https://console.groq.com/docs/models)

# References
```bibtex
@article{yaak2023novel,
    author = {Yaak team},
    title ={A novel test for autonomy},
    journal = {https://www.yaak.ai/blog/a-novel-test-for-autonomy},
    year = {2023},
}
@article{yaak2023actiongpt,
    author = {Yaak team},
    title ={Next action prediction with GPTs},
    journal = {https://www.yaak.ai/blog/next-action-prediction-with-gpts},
    year = {2023},
}
@article{yaak2024si-01,
    author = {Yaak team},
    title ={Building spatial intelligence part - 1},
    journal = {https://www.yaak.ai/blog/buildling-spatial-intelligence-part1},
    year = {2024},
}
@article{yaak2024si-01,
    author = {Yaak team},
    title ={Building spatial intelligence part - 2},
    journal = {https://www.yaak.ai/blog/building-spatial-intelligence-part-2},
    year = {2024},
}
```

## Appendix

## A.1 Route tasks

List of route restrictions. We consider route tags from OSM a restriction if it imposes restrictions on the policy,
for example speed limit, yield or construction. Route features are physical structures along the route, for example inclines,
tunnels and pedestrian crossing. Route maneuvers are different scenarios which a driver encounters during a normal operation
of the vehicle in an urban environment, for example, multilane left turns and roundabouts.

| Type | Name | Assignment | Task ID | Release |
| ----- | ----- | :---: | :---: | ----- |
| Route restriction | CONSTRUCTION | VLM |  | R1 |
| Route restriction | CROSS\_TRAFFIC | VLM | 4.3.1.3a, 4.3.1.3b, 4.3.1.3d, 4.2.1.3a, 4.2.1.3b, 4.2.1.3d | R2 |
| Route restriction | INCOMING\_TRAFFIC | VLM |  | R2 |
| Route restriction | [LIMITED\_ACCESS\_WAY](https://wiki.openstreetmap.org/wiki/Tag:highway%3Dservice) | OSM |  | R0 |
| Route restriction | [LIVING\_STREET](https://wiki.openstreetmap.org/wiki/Tag:highway%3Dliving_street) | OSM |  | R0 |
| Route restriction | [LOW\_SPEED\_REGION](https://wiki.openstreetmap.org/wiki/Key:maxspeed) (5, 10, 20 kph) | OSM |  | R0 |
| Route restriction | [ONE\_WAY](https://wiki.openstreetmap.org/wiki/Key:oneway) | OSM | 3.2.1.3b | R0 |
| Route restriction | PEDESTRIANS | VLM | 7.2.1.3b | R1 |
| Route restriction | [PRIORITY\_FORWARD\_BACKWARD](https://wiki.openstreetmap.org/wiki/Key:priority) | OSM | 3.1.1.3b | R0 |
| Route restriction | [ROAD\_NARROWS](https://wiki.openstreetmap.org/w/index.php?title=Key:turn&uselang=en#Indicated_turns_by_lane) | OSM |  | R0 |
| Route restriction | [STOP](https://wiki.openstreetmap.org/wiki/Tag:highway%3Dstop) | OSM | 4.1.1.3b, 4.2.1.3b, 4.3.1.3b | R0 |
| Route restriction | [YIELD](https://wiki.openstreetmap.org/wiki/Tag:highway%3Dgive_way) | OSM | 4.1.1.3b, 4.2.1.3b, 4.3.1.3b | R0 |
| Route feature | [BRIDGE](https://wiki.openstreetmap.org/wiki/Key:bridge) | OSM |  | R0 |
| Route feature | CURVED\_ROAD | OSM (derived) | 2.1.1.3a, 2.1.1.3b | R0 |
| Route feature | [BUS\_STOP](https://wiki.openstreetmap.org/wiki/Tag:highway%3Dbus_stop) | OSM | 7.1.1.3a | R0 |
| Route feature | [HILL\_DRIVE](https://wiki.openstreetmap.org/wiki/Key:incline) | OSM |  | R0 |
| Route feature | [LOWERED\_KERB](https://wiki.openstreetmap.org/wiki/Tag:barrier%3Dkerb) | OSM |  | R0 |
| Route feature | NARROW\_ROAD | VLM |  |  |
| Route feature | PARKING | OSM |  | R0 |
| Route feature | [PEDESTRIAN\_CROSSING](https://wiki.openstreetmap.org/wiki/Tag:crossing%3Dtraffic_signals) | OSM | 7.2.1.3b | R0 |
| Route feature | [TRAFFIC\_CALMER](https://wiki.openstreetmap.org/wiki/Key:traffic_calming) | OSM |  | R0 |
| Route feature | [TRAIN\_CROSSING](https://wiki.openstreetmap.org/wiki/Tag:railway%3Dlevel_crossing) | OSM | 6.1.1.3a, 6.1.1.3b | R0 |
| Route feature | [TRAM\_TRACKS](https://wiki.openstreetmap.org/wiki/Tag:railway%3Dtram) | OSM | 6.2.1.3a | R0 |
| Route feature | [TUNNEL](https://wiki.openstreetmap.org/wiki/Key:tunnel) | OSM |  | R0 |
| Route feature | [UNCONTROLLED\_PEDESTRIAN\_CROSSING](https://wiki.openstreetmap.org/wiki/Tag:crossing%3Duncontrolled) | OSM | 7.2.1.3b | R0 |
| Route maneuver | ENTERING\_MOVING\_TRAFFIC | OSM (derived) | 4.4.1.3a | R0 |
| Route maneuver | CUTIN | VLM |  | R3 |
| Route maneuver | LANE\_CHANGE | VLM | 1.3.1.3a, 1.3.1.3b | R3 |
| Route maneuver | [MERGE\_IN\_OUT\_ON\_HIGHWAY](https://wiki.openstreetmap.org/wiki/Tag:highway=motorway%20link?uselang=en) | OSM | 1.1.1.3a, 1.1.1.3b, 1.1.1.3c, 1.2.1.3a, 1.2.1.3b, 1.2.1.3c | R0 |
| Route maneuver | [MULTILANE\_LEFT](https://wiki.openstreetmap.org/wiki/Key:turn) | OSM (derived) | 4.3.1.3b, 4.3.1.3c, 4.3.1.3d | R0 |
| Route maneuver | [MULTILANE\_RIGHT](https://wiki.openstreetmap.org/wiki/Key:turn) | OSM (derived) | 4.2.1.3b, 4.2.1.3c, 4.2.1.3d | R0 |
| Route maneuver | PROTECTED\_LEFT | OSM (derived) | 4.3.1.3c, 4.3.1.3d | R0 |
| Route maneuver | PROTECTED\_RIGHT\_WITH\_BIKE | OSM (derived) | 4.2.1.3c, 4.2.1.3d | R0 |
| Route maneuver | RIGHT\_BEFORE\_LEFT | OSM (derived) | 4.1.1.3a, 4.2.1.3a, 4.3.1.3a | R0 |
| Route maneuver | [RIGHT\_TURN\_ON\_RED](https://wiki.openstreetmap.org/wiki/Key:red_turn:right) | OSM | 4.2.1.3c | R0 |
| Route maneuver | [ROUNDABOUT](https://wiki.openstreetmap.org/wiki/Tag:junction%3Droundabout) | OSM | 5.1.1.3a, 5.1.1.3b | R0 |
| Route maneuver | STRAIGHT | OSM (derived) | 8.1.1.3a | R0 |
| Route maneuver | OVER\_TAKE | VLM | 3.2.1.3a, 3.2.1.3b | R4 |
| Route maneuver | UNPROTECTED\_LEFT | OSM (derived) | 4.3.1.3a, 4.3.1.3b | R0 |
| Route maneuver | [UNPROTECTED\_RIGHT\_WITH\_BIKE](https://wiki.openstreetmap.org/wiki/Key:cycleway#Cycle_lanes) | OSM | 4.2.1.3a, 4.2.1.3b | R0 |

<p align="center">
  <em> OSM = Openstreetmap, VLM= Vision Language Model, derived: Hand crafted rules with OSM data </em>
</p>


## A.2 LLM prompts

Prompt template and pseudo code for configuring the LLM using [groq](https://console.groq.com/playground)
to parse natural language queries into [structured prediction](https://console.groq.com/docs/text-chat#json-mode) for route
features, restrictions and maneuvers with a pydantic model. The natural language queries are constructed to closely follow
the turn-by-turn navigation available in most GPS navigation devices.

```py
prompt_template: "You are parsing natural language driving instructions into PyDantic Model's output=model_dump_json(exclude_none=True) as JSON. Here are a few example pairs of instructions and structured output: {examples}. Based on these examples parse the instructions. The JSON must use the schema: {schema}"
groq:
model: llama-3.3-70b-versatile
temperature: 0.0
seed: 1334
response_format: json_object
max_sequence_len: 60000
```

Example pairs (showing 3 / 30\) for in-context learning to steer the structured prediction of LLM,
where ParsedInstructionModel is a pydantic model.

```py
PROMPT_PAIRS = [
(
    		"Its snowing. Go straight through the intersection, following the right before left rule at unmarked intersection",
    	ParsedInstructionModel(
       	 	eventSequence=[
            		EventType(speed=FloatValue(value=10.0, operator="LT", unit="kph")),
            	EventType(osmRouteManeuver="RIGHT_BEFORE_LEFT"),
      	      		EventType(speed=FloatValue(value=25.0, operator="LT", unit="kph")),
       	 	],
        	turnSignal="OFF",
        	weatherCondition="Snow",
		),
    	),
(
    		"stop at the stop sign, give way to the traffic and then turn right",
    	ParsedInstructionModel(
        	eventSequence=[
       	     	EventType(osmRouteRestriction="STOP"),
            	EventType(turnSignal="RIGHT"),
       	     	EventType(speed=FloatValue(value=5.0, operator="LT", unit="kph")),
            	EventType(osmRouteManeuver="RIGHT"),
        	],
    		),
	),
	(
    		"parking on a hill in the rain on a two lane road",
    	ParsedInstructionModel(
       	 	osmLaneCount=[IntValue(value=2, operator="EQ")],
        		osmRouteFeature=["PARKING", "HILL_DRIVE"],
        	weatherCondition="Rain",
    		),
	),
]

EXAMPLES = ""
for idx, (instructions, parsed) in enumerate(PROMPT_PAIRS):
	parsed_json = parsed.model_dump_json(exclude_none=True)
	update = f"instructions: {instructions.lower()} output: {parsed_json}"
	EXAMPLES += update

from groq import Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
            	messages=[
                	{
                    	"role": "system",
                    	"content": prompt_template.format(examples=EXAMPLES, schema=json.dumps(ParsedInstructionModel.model_json_schema(), indent=2))
                	},
                	{
                    	"role": "user",
                    	"content": f"instructions : its daytime. drive to the traffic lights and when it turns green make a left turn",
                	},
            	],
            	model=config["groq"]["model"],
            	temperature=config["groq"]['temperature'],
            	stream=False,
            	seed=config["groq"]['seed'],
            	response_format={"type": config['groq']['response_format']},
        	)

        	parsed_obj = ParsedInstructionModel.model_validate_json(chat_completion.choices[0].message.content)
        	parsed_obj = parsed_obj.model_dump(exclude_none=True)
```

## A.2 Data collection hardware

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/193_l2d/image9.png)

**Onboard compute: NVIDIA Jetson AGX Xavier**

* 8 cores @ 2/2.2 GHz, 16/64 GB DDR5
* 100 TOPS , 8 lanes MIPI CSI-2 D-PHY 2.1 (up to 20Gbps)
* 8x 1080p30 video encoder (H.265)
* Power: 10-15V DC input, \~90W power consumption
* Storage: SSD M.2 (4gen PCIe 1x4)
* Video input 8 cameras:
  * 2x Fakra MATE-AX with 4x GMSL2 with Power-over-Coax support

**Onboard compute: Connectivity**

* Multi-band, Centimeter-level accuracy RTK module
* 5G connectivity: M.2 USB3 module with maximum downlink rates of 3.5Gbps and uplink rates of 900Mbps, dual SIM

| Component | \# | Vendor | Specs |
| :---- | :---- | :---- | :---- |
| RGB: Camera | 1 | [connect-tech](https://www.e-consystems.com/camera-modules/ar0233-hdr-gmsl-camera-module.asp#) | [Techspecs](https://www.e-consystems.com/camera-modules/ar0233-hdr-gmsl-camera-module.asp#) |
| RGB: Rugged Camera | 5 | [connect-tech](https://www.e-consystems.com/camera-modules/ip67-ar0233-gmsl2-hdr-camera.asp) | [Techspecs](https://www.e-consystems.com/camera-modules/ip67-ar0233-gmsl2-hdr-camera.asp) |
| GNSS | 1 | [Taoglas](https://www.taoglas.com/product/aa-175-magmax2-gps-l1-l2-glonass-magnetic-mount-antenna/) | [Techspecs](https://www.taoglas.com/product/aa-175-magmax2-gps-l1-l2-glonass-magnetic-mount-antenna/) |
| 5G antenna | 2 | [2J Antenna](https://techship.com/product/2-j-2-j5283-p-5-g-sub6-antenna-t-bar-2-m-sma-m/?variant=001) | [Datasheet](https://techship.com/download/datasheet-2j5283p/) |
| NVIDIA Jetson Orin NX \- 64 GB | 1 | Nvidia | [Techspecs](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) |

<p align="center">
  <em> Table 6: Information on hardware kit used for data collection </em>
</p>

[Complete hardware kit specs available here](https://yaak.slab.com/public/posts/yaak-data-collection-kit-1w8z0ln0?shr=nmM43GRRDwIZWvFuzYIRBLmn)
