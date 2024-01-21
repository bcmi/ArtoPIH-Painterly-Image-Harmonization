# ArtoPIH-Painterly-Image-Harmonization

We release the dataset and code used in the following paper:
> **Painterly Image Harmonization by Learning from Painterly Objects**  [[arXiv]](https://arxiv.org/pdf/2312.10263.pdf)<br>
>
> Li Niu, Junyan Cao, Yan Hong, Liqing Zhang
>
> Accepted by AAAI 2024

## Artist-Object-Dataset-Arto

This dataset contains: the painterly image (*i.e.*, [WikiArt](https://pan.baidu.com/s/192pGtJeMzj5VqTDjH6DUXg) (access code: sc0c)) and the painterly object mask (in [Baidu Cloud](https://pan.baidu.com/s/1VacWN_5FgOXnzd2q9cIyYA) (access code: ait8)), the photographic object with its corresponding object mask (in [Baidu Cloud](https://pan.baidu.com/s/1x3xqoNvKOdocSjRHFq-pJA) (access code: 3ujl)), and the training list (in [Baidu Cloud](https://pan.baidu.com/s/15ZCUIj9rFc0m_LDpkVCeDA) (access code: l629)).

### How to construct Arto dataset?
First, we use off-the-shelf object detection model pretrained on COCO to detect objects in artistic paintings from WikiArt. 
Then, we train an object retrieval model to retrieve similar object pairs. For each painterly object, we retrieve 100 nearest photographic objects with similar appearance and semantics automatically. 
Finally, as the retrieved results are very noisy and far from usable, we ask annotators to manually remove those dissimilar photographic objects. 

We have 33294 painterly objects associated with similar photographic objects, and each painterly object has an average of 9.83 similar photographic objects.

### The structure of Arto dataset?
For each painterly object in painterly image, we have the annotated similar photographic objects with its corresponding object mask as well as the object category. The training pair information is recorded in the training list.
Examples are shown below.
