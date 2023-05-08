# a universal pipeline for most basic segmentation attack types
This repository is the developing repository by H0xC0000005 with a title "a universal pipeline for white box segmentation attacks", which supports most primitive types of white box segmentation attacks.
\
This repository uses the forked repository as a backbone perturbation update method, so I would rather keep this as a fork. If you are more interested in original implementation of paper please just go to the forked repo.
\
If you are interested in more details or development/discussion about this repository, feel free to drop an email to zhaopeizhusg@gmail.com and I would be very happy to hear from you. :)
## General Information
Here are some brief descriptions about some important files:
\
adaptive_attack.py contains all attack functions and direct utilities for attack algorithm.
\
cityscape_dataset.py contains dataset implementation of Cityscapes, which is also used all the time in my final year project.
You can substitute it into other dataset implementations as long as you provide a valid model for that dataset.
\
cityscapes_single_attack.py, cityscape_static_universal_attack.py,
and cityscape_targeted_universal_attack.py are sample attack processes used 
in my final year project. Feel free to use these building blocks to construct your skycrapers :)
\
eye_dataset.py, unet_model.py,and main.py are from forked repo in case you still want to give them a visit.
\
ext_transforms are a set of borrowed from https://github.com/cc-ai/Deeplabv3.
the model is also borrowed from this repository, and I find them useful for transforming images 
for both adversarial attack and model training. Remember to give them this credit :)
\
helper_functions.py are some global utility functions such as saving images, transforms, statistics, and constraints on perturbation.
\
self_defined_loss.py, well, just as it name suggests. There are some conventions about
callable interface and you may check them within that file.
\
stats_logger is a simple logger to log statistics. Though it's not that ugly it is yet simple, so 
feel free to substitute it or inherit it with your own creative ideas :)
\
|\
If you want some existing experiment data on some attacks, please go to the "data" branch
where all experiment data I produced in my FYP is there.


## Requirements
Since you may run in different types of OS, I cannot suggest any versions for packages, but 
just install these:
```
python > 3.8
torch 
torchvision 
numpy 
PIL 
network
gc
```
