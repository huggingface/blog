---
title: "LeMaterial: an open source initiative to accelerate materials discovery and research" 
thumbnail: /blog/assets/lematerial/thumbnail_lematerial.png
authors:
- user: AlexDuvalinho
  guest: true
- user: lritchie
  guest: true
- user: msiron
  guest: true
- user: inelgnu
  guest: true
- user: etiennedufayet
  guest: true
- user: amandinerossello
  guest: true
- user: Ramlaoui
  guest: true
- user: IAMJB
- user: lvwerra
- user: thomwolf
---

![logos](/blog/assets/lematerial/thumbnail_lematerial.png)

# LeMaterial: an open source initiative to accelerate materials discovery and research

Today, we are thrilled to announce the launch of **LeMaterial**, an open-source collaborative project led by [*Entalpic*](https://entalpic.ai/) and [*Hugging Face*](https://huggingface.co/). LeMaterial aims to simplify and accelerate materials research, making it easier to train ML models, identify novel materials and explore chemical spaces. ⚛️🤗

As a first step, we are releasing a dataset called `LeMat-Bulk`, which unifies, cleans and standardizes the most prominent material datasets, including [Materials Project](https://next-gen.materialsproject.org/), [Alexandria](https://alexandria.icams.rub.de/) and [OQMD](https://oqmd.org/) — giving rise to a single harmonized data format with **6.7M entries** and **7 materials properties.**

## Why LeMaterial?

The world of materials science, at the intersection of quantum chemistry and machine learning, is brimming with opportunity — from brighter LEDs, to electro-chemical batteries, more efficient photovoltaic cells and recyclable plastics, the applications are endless. By leveraging machine learning (ML) on large, structured datasets, researchers can perform high-throughput screening and testing of new materials at unprecedented scales, significantly accelerating the discovery cycle of novel compounds with desired properties. In this paradigm, **data becomes the essential fuel powering ML models** that can guide experiments, reduce costs, and unlock breakthroughs faster than ever before.

However, **this field is hampered by fragmented datasets that vary in format, parameters, and scope, presenting the following challenges:**

- Dataset integration issues (eg. inconsistent formats or field definitions, incompatible calculations)
- Biases in dataset composition (for eg. Material Project's focus on oxides and battery materials)
- Limited scope (eg. NOMADs focus on quantum chemistry calculations rather than material properties)
- Lack of clear connections or identifiers between similar materials across different databases

This fragmented landscape makes it challenging for researchers in AI4Science and materials informatics to leverage existing data effectively. Whether the application involves training foundational ML models, constructing accurate phase diagrams, identifying novel materials or exploring the chemical space effectively, there is no simple solution. While efforts like [Optimade](https://optimade.org/) standardize structural data, they don't address discrepancies in material properties or biases in dataset scopes.

**LeMaterial** addresses these challenges by unifying and standardizing data from three major databases—Materials Project, Alexandria, and OQMD—into a high-quality resource with consistent and systematic properties. The elemental composition treemap below highlights the value of this integration, showcasing how we increase the scope of existing datasets, like Materials Project, which are biased toward specific material types, such as battery materials (Li, O, P) or oxides.

<p align="center">
    <img src="https://huggingface.co/datasets/LeMaterial/admin/resolve/main/MP_LeMatBulk_Combined_Treemap.png" alt="drawing" width="1000"/>
</p>

<p align="center">
  <em>Materials Project and LeMat-BulkUnique treemap</em>
</p>


## Achieving a clean, unified and standardized dataset

`LeMat-Bulk` is more than a large-scale merged dataset with a permissive license (CC-BY-4.0). With its 6.7M entries with consistent properties, it represents a foundational step towards creating a curated and standardized open ecosystem for material science, designed to simplify research workflows and improve data quality. Below is a closer view of what is looks like. To interactively browse through our materials, check out the [Materials Explorer space.](https://huggingface.co/spaces/LeMaterial/materials_explorer)

<iframe
  src="https://huggingface.co/datasets/LeMaterial/LeMat-Bulk/embed/viewer/compatible_pbe/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

*[View the complete Datacard](https://huggingface.co/datasets/LeMaterial/LeMat-Bulk#download-and-use-within-python)*


| **Release** | **Description & Value** | **Date** |
| --- | --- | --- |
| v.1.0 | <ul><li><strong>Data collection and merging</strong> from various sources: Materials Project, Alexandria and OQMD datasets, including multiple fields and DFT functionals (PBE, PBESol, SCAN). </li><li><strong>Data cleaning</strong>: identify and remove datapoints not conforming to the set standard (e.g. with non-compatible calculations). </li><li><strong>Standardization</strong>: uniformly format the set of fields across databases, using Optimade standard </li><li><strong>Material fingerprint</strong>: attribute a unique identifier to each material with a well benchmarked hashing function, enabling to remove duplicates</li><li><strong>Accessibility and visualization</strong>: propose several tools to easily explore and visualize the proposed new dataset (spits, phase diagrams, material explorer, property explorer)</li></ul>  | Dec. 10, 2024 |
| v.1.1 | <ul><li><strong>New data</strong>: calculate charge data for many materials in Materials Project database (additional 53k data </li>points), band gaps information, unified energy correction scheme<li><strong>Evaluation</strong>: Code hashing function benchmarks</li><li><strong>New functions</strong>: Material similarity metric and retriever tool of similar structures</li><li><strong>New models</strong>: Equiformerv2 and FAENet trained on the data</li><li><strong>Data validation</strong>: check for compatibility of fields, formatting, etc.</li></ul> | Q1 2025 |
| Future releases | <li><strong>New data</strong>: release of r2SCAN data from Materials Project </li><li><strong>New surface datasets</strong>: OC20 & OC22 datasets and ability to cover other models than bulk (surface and slab+ads and molecule structures)</li><li><strong>Further harmonisation</strong>: include trajectories across MPTrj, OMat24 in similar format, include relational data in our DB </li> | Q2 2025 |

**We offer different datasets and subsets,** enabling tailored workflows for researchers depending on their needs (consistent calculations, deduplicating materials, or comprehensive exploration):

- **Compatibility:** these subsets only provides calculations which are compatible to mix. This is available in 3 functionals today (PBE, PBESol and SCAN)
- **Non-compatible:** this subset provides all materials not included in the compatibility subsets.
- [**LeMat-BulkUnique**](https://huggingface.co/datasets/LeMaterial/LeMat-BulkUnique) : this dataset split provides de-duplicated material using our structure fingerprint algorithm. It is available in 3 subsets, for PBE, PBESol, and SCAN functionals. More Details on the dataset can be found on [🤗Hugging Face](https://huggingface.co/datasets/LeMaterial/LeMat-BulkUnique)
## Integrating a well-benchmarked materials fingerprint

Beside building this standardized dataset, one of the key contribution of LeMaterial is to propose a definition of a material fingerprint through a **hashing function** that assigns a unique identifier to each material.

Current approaches to identifying a material as novel relative to a database have predominantly relied on similarity metrics, which necessitate a combinatorial effort to screen the existing database for novelty. To provide faster novelty detection in a dataset, Entalpic introduces a hashing method to compute the fingerprint of a material.


![image.png](https://huggingface.co/datasets/LeMaterial/admin/resolve/main/Hash%20Function.jpeg)
<blockquote>Above is a breakdown of the fingerprinting. We use a bonding algorithm (e.g. EconNN) on the crystal structure to extract a graph, on which we then compute the Weisfeiler-Lehman algorithm to get a hash. This hash is combined with composition and space group information to create the material fingerprint.</blockquote>


**Our fingerprinting approach offers several benefits**: 

- Quickly identifying whether a material is novel or already catalogued.
- Ensuring the dataset is free from duplicates and inconsistencies.
- Allowing to connect materials between datasets.
- Supporting more efficient calculations for thermodynamic properties, such as energy above the hull.

Below lies a comparison of our hash function with the [StructureMatcher](https://pymatgen.org/pymatgen.analysis.html#pymatgen.analysis.structure_matcher.StructureMatcher:~:text=%3D%20True%2C-,attempt_supercell,-%3A%20bool%20%3D) of Pymatgen, to find all duplicates of a dataset. The experiment was run on two datasets having very different  structures.

When using our method, **almost all of the task time was dedicated to calculating material hashes**; the follow-up comparison step is negligible time-wise. When using `StructureMatcher`, the vast majority of the task time was spent **comparing pairs of structures**; building said structures is negligible time-wise.

| **Dataset** | **Number of structures** | Task time for the hash function (parallelized on 12 CPUs) | Task time for StructureMatcher (parallelized on 64 CPUs) |
| --- | --- | --- | --- |
| Carbon-24 | 10,153 | 100 seconds | 17 hours |
| MPTS-52 | 40,476 | 330 seconds | 4.9 hours |

Additionally, we are planning on releasing **a set of well-curated benchmarks** to evaluate the validity of our hashing function. For instance, we investigated:

- If distinct materials lead to different hashes based on material identification tags across existing databases
- Whether adding small noises or applying symmetry operations to a material leads to the same hash
- If materials sharing the same hash, across or within databases, could indeed be the same material — with manual and DFT checks
- How fast and accurate our hash is compared to Pymatgen's StructureMatcher on existing databases

<blockquote>
<strong> 🤗 Call to the community:</strong> our aim is not to position this fingerprint method as the single solution to de-duplicate materials databases and find novel materials, but rather to foster discussion around this question. One current limitation of this hashing technique is that it does not cover disordered structures; we would like to push the community towards finding a consensus, while proposing a relatively simple and efficient fingerprint method in the meantime.</blockquote>

## LeMaterial in Action: applications and impact

LeMaterial aims to be a community-driven initiative that gathers machine learning models, visualization tools, large & curated datasets, handy toolkits, etc. It is designed to be practical and flexible, enabling a wide range of applications, such as:

- **Exploring extended phase diagrams** ([link to our explorer](https://huggingface.co/spaces/LeMaterial/phase_diagram)), constructed with a broader dataset, to analyze chemical spaces in greater detail. Combining larger datasets means that we can provide a finer resolution of material stability in a given compositional space:

<p align="center">
    <img src="https://huggingface.co/datasets/LeMaterial/admin/resolve/main/experimental%20phase%20diagram%20Ti%3Abb%3ASn%20(research%20paper).png" alt="drawing" width="350"/>
</p>

<p align="center">
  <em>Experimental phase diagram of Ti, Bb, Sn from
  <a href="https://www.mdpi.com/2075-4701/6/3/60">this research paper</a>
  </em>
</p>

<p align="center">
    <img src="https://huggingface.co/datasets/LeMaterial/admin/resolve/main/Ti_Nb_Sn_LeMat110_PD.png" alt="drawing" width="400"/>
</p>

<p align="center">
  <em>LeMat-Bulk phase diagram for Sn, Ti, Nb
  </em>
</p>
    
- **Compare materials properties across databases and functionals:** by providing researchers with data across DFT functionals, and by linking materials via our materials fingerprint algorithm we are able to establish and connect materials properties calculated via different parameters. This gives researchers insight into how functionals might behave and differ across compositional space.
- **Determining if a material is novel**. Our hashing function allows researchers to quickly assess whether a material is unique or a duplicate, streamlining the discovery process and avoiding redundant calculations.
    - Example 1: Our fingerprint method identified the following Alexandria entries (`agm002153972`, `agm002153975`) as *potentially* being the same material — having the same hash. When we did a relaxation on the higher energy entry, the material relaxed to the lower energy configuration.
 
      <table border="0" class="ck-table-resized">
          <colgroup>
              <col style="width:50%;">
              <col style="width:50%;">
          </colgroup>
          <tbody>
              <tr>
                  <td>
                      <p style="text-align:center;"><figure class="image image_resized"><img src="https://huggingface.co/datasets/LeMaterial/admin/resolve/main/lower%20energy%20structure.png"></figure></p>
                  </td>
                  <td>
                      <p style="text-align:center;"><figure class="image image_resized"><img src="https://huggingface.co/datasets/LeMaterial/admin/resolve/main/higher%20energy%20structure.png"></figure></p>
                  </td>
              </tr>
              <tr>
                  <td>
                      <p style="text-align:center;"><em>Lower energy structure</em></p>
                  </td>
                  <td>
                      <p style="text-align:center;"><em>Higher energy structure</em></p>
                  </td>
              </tr>
          </tbody>
      </table>
    
    - Example 2: applying our hash to another dataset ([AIRSS](https://archive.materialscloud.org/record/2020.0026/v1)) that is often used in training generative models, we found the following materials with the same hash.
        <table border="0" class="ck-table-resized">
        <colgroup>
            <col style="width:50%;">
            <col style="width:50%;">
        </colgroup>
        <tbody>
            <tr>
                <td>
                    <p style="text-align:center;"><img src="https://huggingface.co/datasets/LeMaterial/admin/resolve/main/unit%20cells%20%3Asame%20materials%201%3A2.png">
                  </p>
                </td>
                <td>
                    <p style="text-align:center;"><img src="https://huggingface.co/datasets/LeMaterial/admin/resolve/main/unit%20cells%20%3A%20same%20materials%202%3A2.png"></p>
                </td>
            </tr>
        </tbody>
          <caption>
            <p><em>Unit cells of materials sharing the same fingerprint</em></p>
          </caption>
        </table>


        To an untrained eye these visually appear like very different materials. However when we replicate the lattice we quickly identify that they are quite similar:
            <table border="0" class="ck-table-resized">
            <colgroup>
                <col style="width:50%;">
                <col style="width:50%;">
            </colgroup>
            <tbody>
                <tr>
                    <td>
                        <p style="text-align:center;"><img src="https://huggingface.co/datasets/LeMaterial/admin/resolve/main/supercells%201%3A2.png">
                      </p>
                    </td>
                    <td>
                        <p style="text-align:center;"><img src="https://huggingface.co/datasets/LeMaterial/admin/resolve/main/unit%20cells%20%3A%20same%20materials%202%3A2.png"></p>
                    </td>
                </tr>
            </tbody>
              <caption>
                <p><em>Supercells of materials sharing the same fingerprint</em></p>
              </caption>
            </table>
        
        Its important to note here that Pymatgen's StructureMatcher failed on this example, identifying these two unit cells as different materials, when they are indeed the exact same structures. Here our hashing algorithm was able to identify them as indeed the same.

- **Training predictive ML models.** We can also train machine learning interatomic potentials like EquiformerV2 on `LeMat-Bulk`. These models should benefits from its scale and data quality and the removal of bias across compositional space, and it would be interesting to assess the benefits of this new dataset. An example of how to incorporate LeMaterial with fairchem can be found in [Colab](https://colab.research.google.com/drive/1y8_CzKM5Rgsiv9JoPmi9mXphi-kf6Lec?usp=sharing). We are currently in process of training an EquiformerV2 model using this dataset — stay tuned 💫

## Take-aways

As a community, we often place a lot of value in the **quality** of these large-scale open-source databases. However the lack of standardization makes utilizing multiple dataset a huge challenge. **LeMaterial** offers a solution that unifies, standardizes, performs extra cleaning and validation efforts on existing major data sources. This new open-science project is designed to accelerate research, improve quality of ML models, and make materials discovery more efficient and accessible.

**We are just getting started** — we know there are still flaws and improvements to be made — and would thus love to hear your feedback! So please reach out if you are interested to contribute to this open-source initiative. We would be excited to continue expanding LeMaterial with new datasets, tools, and applications — alongside the community! ⚛️🤗


> We extend our heartfelt thanks to [Zachary Ulissi](https://zulissi.github.io/) and [Luis Barroso-Luque](https://www.linkedin.com/in/luis-barroso-luque-9742598a/) (Meta), and [Matt McDermott](https://github.com/mattmcdermott) (Newfound Materials, Inc.) for their valuable feedback regarding is initiative.
>
> We also acknowledge the incredible contributions of the creators of Optimade, Alexandria, Materials Project, and OQMD. Their foundational work has been instrumental in enabling the development of LeMaterial, and we are proud to build upon their legacy as part of the collaborative effort to advance materials science.

---

To learn more about LeMaterial and get involved: 

- 👉🏻 [Form](https://forms.gle/KvZLmo12Ps7252gi9) to join the community

- 🤗 [Hugging Face Space](https://huggingface.co/LeMaterial)

- 💻 [Github](https://github.com/lematerial)
