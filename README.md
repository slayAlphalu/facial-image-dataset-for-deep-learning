# facial-image-dataset-for-deep-learning
### This repo contains my self-built dataset for deep learning-facial image rating track. Feel free to use the code or dataset if you want to train your own rater machine. 

### The final dataset contains 682 labeled(1-10) facial images and you can do data augmentation by yourself.

# Steps from sctrach:
**1.Crawled thousands of images from facebook and only keep the high quality white woman facial images.**

**2.run py document as follows:**

```bash
run find_faces_in_picture.py
```

```bash
run rename.py
```

```bash
run resize_image.py

```
**zip files specified:**

The original dataset [web_image](https://github.com/slayAlphalu/facial-image-dataset-for-deep-learning/blob/master/web_image.zip)

The datasets with logical names [full_data_rename.zip](https://github.com/slayAlphalu/facial-image-dataset-for-deep-learning/blob/master/full_data_rename.zip)

The final dataset after every image standardized to 224x224x3 [full_data_standardize.zip](https://github.com/slayAlphalu/facial-image-dataset-for-deep-learning/blob/master/full_data_standardize.zip)


### p.s. the dataset originally published in [kaggle](https://www.kaggle.com/datasets?sortBy=updated&group=my&page=1&pageSize=20&size=all&filetype=all&license=all&tagids=13300)
