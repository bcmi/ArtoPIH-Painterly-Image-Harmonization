# ArtoPIH-Painterly-Image-Harmonization

We release the dataset and code used in the following paper:
> **Painterly Image Harmonization by Learning from Painterly Objects**  [[arXiv]](https://arxiv.org/pdf/2312.10263.pdf)<br>
>
> Li Niu, Junyan Cao, Yan Hong, Liqing Zhang
>
> Accepted by AAAI 2024
## Artist-Object-Dataset-Arto

Our Arto dataset contains 33,294 painterly objects in artistic images with accurate object masks. Each painterly object is associated with a set of similar photographic objects.


### Painterly Objects
First, we use off-the-shelf [object detection model](https://github.com/facebookresearch/detectron2) to detect the objects in artistic images from WikiArt training set. We use [segment anything model](https://segment-anything.com/demo) and manual refinement to get accurate masks of painterly objects. We obtain 33,294 painterly objects with accurate object masks. We release the artistic images in [WikiArt](https://pan.baidu.com/s/192pGtJeMzj5VqTDjH6DUXg) (access code: sc0c) and the painterly object masks in [Baidu Cloud](https://pan.baidu.com/s/1VacWN_5FgOXnzd2q9cIyYA) (access code: ait8). Painterly object mask is named `<painterly-image-name>_<id>.png`, indicating the painterly object in the artistic image.



### Similar Photographic Objects

Then, we train an object retrieval model (see the supplementary in [[arXiv]](https://arxiv.org/pdf/2312.10263.pdf)) to retrieve similar photographic objects for each painterly object. Specifically, provided with one painterly object, we retrieve 100 nearest photographic objects from COCO 2014 training set, which have similar color and semantics with the given painterly object. However, the retrieved results are very noisy, so we ask annotators to manually remove those dissimilar photographic objects. Each painterly object has an average of 9.83 similar photographic objects. 


We release all the photographic objects with object masks in [Baidu Cloud](https://pan.baidu.com/s/1x3xqoNvKOdocSjRHFq-pJA) (access code: 3ujl). Note that the photographic objects are cropped from the original photographic images based on the bounding boxes.  Photographic object image is named `<photographic-image-name>_<id>.jpg` and its object mask is named `<photographic-image-name>_<id>.png`. For each painterly object, we provide a similar object list named `<painterly-image-name>_<id>.txt` which records its similar photographic objects. Each line in the list records the information of one similar photographic object. We release the similar object lists in [Baidu Cloud](https://pan.baidu.com/s/15ZCUIj9rFc0m_LDpkVCeDA) (access code: l629).

### Examples

An example is shown below. The painterly object in the leftmost column is outlined in yellow. The rest of columns are similar photographic objects. 
![dataset_example](https://github.com/bcmi/ArtoPIH-Painterly-Image-Harmonization/assets/59011028/f67e439e-314e-42ca-af4e-878091c19868)

In the painterly object name `Art_Nouveau_Modern/andrei-ryabushkin_moscow-street-of-xvii-century_0`, `Art_Nouveau_Modern/andrei-ryabushkin_moscow-street-of-xvii-century` is the name of the painterly image where the painterly object belongs to, and `0` is the annotation id.
The similar object list is  `Art_Nouveau_Modern/andrei-ryabushkin_moscow-street-of-xvii-century_0.txt`. In this list, each line records the information of a similar photographic object. From left to right are: painterly object name, category id and name, similar photographic object name. In the photographic object name `COCO_train2014_000000021830_481814`, `COCO_train2014_000000021830` is the name of the photographic image where the photographic object belongs to, and `481814` is the annotation id.

We provide more example pairs of painterly objects and photographic objects below.

![dataset_example_pairs](https://www.ustcnewly.com/github_images/xn8ejrp2.jpg)


### Dataset Statistics

The distribution of painterly object categories:

![1706083873335](https://github.com/bcmi/ArtoPIH-Painterly-Image-Harmonization/assets/59011028/bb43f3b0-bab1-4fda-a2c5-feee9a13f242)

The distribution of the number of similar photographic objects for each painterly object:

![1706084867354](https://github.com/bcmi/ArtoPIH-Painterly-Image-Harmonization/assets/59011028/baa9bdec-288a-4210-8d00-6da1f4594f6c)
