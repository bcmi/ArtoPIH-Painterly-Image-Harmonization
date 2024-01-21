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
First, we use off-the-shelf [object detection model](https://github.com/facebookresearch/detectron2) pretrained on COCO to detect objects in artistic paintings from WikiArt. 
Then, we train an object retrieval model to retrieve similar object pairs. The photographic objects are from COCO2014 training set. For each painterly object, we retrieve 100 nearest photographic objects with similar appearance and semantics automatically. 
Finally, as the retrieved results are very noisy and far from usable, we ask annotators to manually remove those dissimilar photographic objects. 

We have 33294 painterly objects associated with similar photographic objects, and each painterly object has an average of 9.83 similar photographic objects.

### The structure of Arto dataset?
For each painterly object in painterly image, we have the annotated similar photographic objects with its corresponding object mask as well as the object category. We crop the photographic object from the original photographic image based on the object bounding box, and name it `<photographic-image-name>_<id>`. Each painterly object mask is named `<painterly-image-name>_<id>`, indicating the painterly object in this painterly image. A file with the same name as the painterly object mask records the training pair information.

Example is shown below. The painterly object is outlined in yellow.
![dataset_example](https://github.com/bcmi/ArtoPIH-Painterly-Image-Harmonization/assets/59011028/f67e439e-314e-42ca-af4e-878091c19868)

In the painterly object's name `Art_Nouveau_Modern/andrei-ryabushkin_moscow-street-of-xvii-century_0`, `Art_Nouveau_Modern/andrei-ryabushkin_moscow-street-of-xvii-century` is the name of the painterly image where the painterly object belongs to, and `0` is the annotation id.
The training file has the same name as the painterly object, *i.e.*, `Art_Nouveau_Modern/andrei-ryabushkin_moscow-street-of-xvii-century_0`. In this file, each line records the information of a similar photographic object. From left to right are: painterly object name, category id and name, similar photographic object name. In the photographic object's name `COCO_train2014_000000021830_481814`, `COCO_train2014_000000021830` is the name of the photographic image where the photographic object belongs to, and `481814` is the annotation id.
