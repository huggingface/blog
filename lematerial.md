---
title: "LeMaterial: an open source initiative to accelate materials discovery and research" 
thumbnail: /blog/assets/lematerial/thumbnail_lematerial.png
authors:
- user: TODO add entalpic folks
  guest: true
- user: IAMJB
- user: lvwerra
- user: thomwolf
---

# LeMaterial: an open source initiative to accelate materials discovery and research

> Today, we are thrilled to announce the launch of **LeMaterial**, an open-source collaborative project led by [*Entalpic*](https://entalpic.ai/) and [*Hugging Face*](https://huggingface.co/) ⚛️🤗. 
LeMaterial aims to simplify and accelerate materials research, making it easier to train ML models, to identify novel materials and to explore chemical spaces. **As a first step, we are releasing a dataset called `LeMat-Bulk`, which unifies, cleans and standardizes the most prominent material datasets, including [Materials Project](https://next-gen.materialsproject.org/), [Alexandria](https://alexandria.icams.rub.de/) and [OQMD](https://oqmd.org/) — giving rise to a single harmonized data format with **6.7M entries** and **7 materials properties.**
> 

## Why LeMaterial?

The world of materials science, at the intersection of quantum chemistry and machine learning, is brimming with opportunity — from brighter LEDs, to electro-chemical batteries, more efficient photovoltaic cells and recyclable plastics, the applications are endless. By leveraging machine learning (ML) on large, structured datasets, researchers can perform high-throughput screening and testing of new materials at unprecedented scales, significantly accelerating the discovery cycle of novel compounds with desired properties. In this paradigm, **data becomes the essential fuel powering ML models** that can guide experiments, reduce costs, and unlock breakthroughs faster than ever before.

However, **this field is hampered by** **fragmented datasets** **that vary in format, parameters, and scope, presenting the following challenges:**

- Dataset integration issues (eg. inconsistent formats or field definitions, incompatible calculations)
- Biases in dataset composition (for eg. Material Project’s focus on oxides and battery materials)
- Limited scope (eg. NOMAD’s focus on quantum chemistry calculations rather than material properties)
- Lack of clear connections or identifiers between similar materials across different databases

This fragmented landscape makes it challenging for researchers in AI4Science and materials informatics to leverage existing data effectively. Whether the application involves training foundational ML models, constructing accurate phase diagrams, identifying novel materials or exploring the chemical space effectively, there is no simple solution. While efforts like [Optimade](https://optimade.org/) standardize structural data, they don’t address discrepancies in material properties or biases in dataset scopes.

**LeMaterial** addresses these challenges by unifying and standardizing data from three major databases—Materials Project, Alexandria, and OQMD—into a high-quality resource with consistent and systematic properties. The elemental composition treemap below highlights the value of this integration, showcasing how we increase the scope of existing datasets, like Materials Project, which are biased toward specific material types, such as battery materials (Li, O, P) or oxides.

![LeMat_Bulk_unique_materials.png](TODO LINK)

*LeMat-BulkUnique treemap*

![mp_treemap.png](TODO LINK)

*MaterialProject treemap*

> Add a quote from Meta commenting on this dataset
> 

## Achieving a clean, unified and standardized dataset

`LeMat-Bulk` is more than a large-scale merged dataset with a permissive license (CC-4-BY). With its 6.7M entries with consistent properties, it represents a foundational step towards creating a curated and standardized open ecosystem for material science, designed to simplify research workflows and improve data quality. Here's a closer view of what is looks like and how it was constructed.

![Screenshot 2024-12-09 at 06.45.32.png](TODO LINK)

DataCard: [https://huggingface.co/datasets/LeMaterial/LeMat-Bulk#download-and-use-within-python](https://huggingface.co/datasets/LeMaterial/LeMat-Bulk#download-and-use-within-python)


TODO: REMOVE
<aside>

Vizu → Insert a the dataset viewer + the link to the datacard @Inel Djafar 

can we use this perhaps?? @Lucile Ritchie @Inel Djafar  (its essentially a snapshot of the data):

![image.png](TODO LINK)

</aside>

**New releases** will come in the next months and quarters, bringing again more value to this dataset:

TODO: we need to reformat that table - markdown doesn't support lists in tables
(https://stackoverflow.com/questions/19950648/how-to-write-lists-inside-a-markdown-table)
maybe multiple rows per version where each row is an element would work?

| **Release** | **Description & Value** | **Date** |
| --- | --- | --- |
| v.1.0 | - **Data collection and merging** from various sources: Materials Project, Alexandria and OQMD datasets, including multiple fields and DFT functionals (PBE, PBESol, SCan). 
- **Data cleaning**: identify and remove datapoints not conforming to the set standard (e.g. with non-compatible calculations). 
- **Standardization**: uniformly format the set of fields across databases, using Optimade standard 
- **Material fingerprint**: attribute a unique identifier to each material with a well benchmarked hashing function, enabling to remove duplicates
- **Accessibility and visualization**: propose several tools to easily explore and visualize the proposed new dataset (spits, phase diagrams, material explorer, property explorer)  | Dec. 10, 2024 |
| v.1.1 | - **New data**: calculate charge data for many materials in Materials Project database (additional 53k data points), band gaps information, unified energy correction scheme
- **Evaluation**: Code hashing function benchmarks
- **New functions**: Material similarity metric and retriever tool of similar structures
- **New models**: Equiformerv2 and FAENet trained on the data
- **Data validation**: check for compatibility of fields, formatting, etc. | Q1 2025 |
| Future releases | - **New data:** release of R2SCAN data from MaterialsProject 
- **New surface datasets**: OC20 & OC22 datasets and ability to cover other models than bulk (surface and slab+ads and molecule structures)
- **Further harmonisation:** include trajectories across MPTrj, OMat24 in similar format, include relational data in our DB | Q2 2025 |

**We offer different dataset and subsets,** enabling tailored workflows for researchers depending on their needs (consistent calculations, deduplicating materials, or comprehensive exploration)

- **Compatibility:** these subsets only provides calculations which are compatible to mix. This is available in 3 functionals today (PBE, PBESol and SCAN)
- **Non-compatible:** this subset provides all materials not included in the compatibility subsets.
- `LeMat-BulkUnique**:**` this dataset split provides de-duplicated material using our structure fingerprint algorithm. It is available in 3 subsets, for each functional available.

## Integrating a well-benchmarked materials fingerprint

Beside building this standardized dataset, one of the key contribution of LeMaterial is to propose a definition of a material fingerprint through a **hashing function** that assigns a unique identifier to each material.

Current approaches to identifying a material as novel relative to a database have predominantly relied on similarity metrics, which necessitate a combinatorial effort to screen the existing database for novelty. **To provide for faster detection of novelty in a dataset, Entalpic proposes a hashing method to compute the fingerprint of a material. 

![Platform Team Charts - Hash Function (5).jpeg](TODO LINK)

> **Caption**: above is a breakdown of the fingerprint processing. We use a bonding algorithm (e.g. EconNN) on the crystal structure to extract the graph, on which we then compute the WL algorithm to get a hash. This hash is combined with composition and space group information to create the material fingerprint.
> 

**Our fingerprinting approach offers several benefits**: 

- Quickly identifying whether a material is novel or already cataloged.
- Ensuring the dataset is free from duplicates and inconsistencies.
- Allowing to connect materials between datasets.
- Supporting more efficient calculations for thermodynamic properties, such as energy above the hull.

Below lies a comparison of our hash function with the [StructureMatcher](https://pymatgen.org/pymatgen.analysis.html#pymatgen.analysis.structure_matcher.StructureMatcher:~:text=%3D%20True%2C-,attempt_supercell,-%3A%20bool%20%3D) of Pymatgen, to find all duplicates of a dataset. The experiment was run on two datasets having very different  structures. Hashes were pre-computed (as they are released in LeMaterial), so we do not take into account their compute time. 

| **Dataset** | **Number of structures** | Task time for our Hash function | Task time for StructureMatcher |
| --- | --- | --- | --- |
| Carbon-24 | 10,153 | 0.15 s | 17 hours |
| MPTS-52 | 40,476 | 0.053 s | 4.9 hours |

Additionally, we are planning on releasing **a set of well-curated benchmarks** to evaluate the validity of our hashing function. For instance, we investigated:

- If distinct materials lead to different hashes based on material identification tags across existing databases
- Whether adding small noises or applying symmetry operations to a material leads to the same hash
- If materials sharing the same hash, across or within databases, could indeed be the same material — with manual and DFT checks
- How fast and accurate our hash is compared to Pymatgen’s StructureMatcher on existing databases

> **Call to the community**: the aim is not to position our fingerprint method as the single solution to de-duplicate materials databases and find novel materials, but rather to foster discussion around this question. We would like to push the community towards achieving a consensus, while proposing a relatively simple and efficient fingerprint method in the meantime.
> 

## LeMaterial in Action: applications and impact

LeMaterial aims to be a community-driven initiative that gathers machine learning models, visualization tools, large & curated datasets, handy toolkits, etc. It is designed to be practical and flexible, enabling a wide range of applications, such as:

- **Exploring extended phase diagrams** ([link to Hugging Face](https://huggingface.co/spaces/LeMaterial/phase_diagram)), ****constructed with a broader dataset, to analyze chemical spaces in greater detail. Combining larger datasets means that we can provide a finer resolution of material stability in a given compositional space:
    
    
    ![Screenshot 2024-12-09 at 20.14.05.png](TODO LINK)
    
    Materials Project phase diagram for Mg, Sc, Zn
    
    ![newplot (5).png](TODO LINK)
    
    LeMat-Bulk phase diagram for Mg, Sc, Zn
    
- **Compare materials properties across databases and functionals** ([Link to Hugging Face](https://huggingface.co/spaces/LeMaterial/PropertyExplorer))**:** by providing researchers with data across DFT functionals, and by linking materials via our materials fingerprint algorithm we are able to establish and connect materials properties calculated via different parameters. This gives researchers insight into how functionals might behave and differ across compositional space.
- **Determining if a material is novel** . Using the hashing function allows a researcher to quickly assess whether materials are unique or duplicates, which is particularly useful in the discovery phases.
    - Example 1: Our fingerprint method identified the following Alexandria entries (`agm002153972`, `agm002153975`) as *potentially* being the same material — having the same hash. When we did a relaxation on the higher energy entry, the material relaxed to the lower energy configuration.
    
    ![1.gif](TODO LINK)
    
    ![image.png](TODO LINK)
    
    ![image.png](TODO LINK)
    
    - Example 2: applying our hash to another dataset ([AIRSS](https://archive.materialscloud.org/record/2020.0026/v1)) that is often used in training generative models, we found the following materials with the same hash.
    
    ![image.png](TODO LINK)
    
    ![image.png](TODO LINK)
    
    From an untrained eyes these visually appear like very different materials. However when we replicate the lattice we quickly identify that they are quite similar:
    

![image.png](TODO LINK)

![image.png](TODO LINK)

Its important to note here that Pymatgen’s StructureMatcher failed on this example, identifying these two unit cells as different materials, when they are indeed the exact same structures. Here our hashing algorithm was able to identify them as indeed the same.

- **Training predictive ML models.** We can also train machine learning interatomic potentials like EquiformerV2 on `LeMat-Bulk`. These models should benefits from its scale and data quality and the removal of bias across compositional space, and it would be interesting to assess the benefits of this new dataset. An example of how to incorporate LeMaterial with fairchem can be found in [Colab](https://colab.research.google.com/drive/1y8_CzKM5Rgsiv9JoPmi9mXphi-kf6Lec?usp=sharing).
    
    > This is ongoing work — stay tuned 💫
    > 

## Take-aways

As a community, we often place a lot of value in the **quality** of these large-scale open-source databases. However the lack of standardization makes utilizing multiple dataset a huge challenge. **LeMaterial** offers a solution that unifies, standardizes, performs extra cleaning and validation efforts on existing major data sources. This new open-science project is designed to accelerate research, improve the quality of ML models in the space, and make materials discovery more efficient and accessible.

**We are just getting started** — we know there are still flaws and improvements to be made — and would thus love to hear your feedback ! So please reach out if you are interested to contribute to this open-source initiative ⚛️🤗. We would be excited to continue expanding LeMaterial with new datasets, tools, and applications — alongside the community !

> To learn more about LeMaterial and get involved: 
👉🏻 [Form](https://docs.google.com/forms/d/1BDGaUusCis9XiIi6VaNlix9EcGyU8uZ1XixWEQRr-34/prefill) to join the community
https://huggingface.co/LeMaterial
https://lematerial.github.io/lematerial-website/