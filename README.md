# Scale-Iterative Upscaling Network for Image Deblurring
by Minyuan Ye, Dong Lyu and Gengsheng Chen
### One real example
![/comparisions/images_in_paper/real_building1_comparision.png](../master/comparisons/images_in_paper/Real_building1_comparison.png)<br>
(a) Result of Nah et al. (b) Result of Tao et al. (c) Result of Zhang et al. (d) Our result.
<br>
### Results on benchmark datasets
![/comparisions/images_in_paper/benchmark_comparison.png](../master/comparisons/images_in_paper/benchmark_comparison.png)<br>
From top to bottom are blurry input, deblurring results of Nah et al., Tao et al., Zhang et al. and ours.<br>
<br>
### Results on real-world blurred images
![/comparisions/images_in_paper/real_comparison.png](../master/comparisons/images_in_paper/real_comparison.png)<br>
From top to bottom are images restored by Pan et al., Nah et al., Tao et al., Zhang et al. and ours. As space limits, the original blurry images are omitted here. 
They can be viewed in Lai dataset with their names, from left to right: boy_statue, pietro, street4 and text1.
<br>
## Prerequisites
Please refer to "/code/requirements.txt".
<br>
## Installation

```
git clone https://github.com/minyuanye/SIUN.git
cd code
```

## Basic usage
You can always add '--gpu=<gpu_id>' to specify GPU ID, the default ID is 0.<br>

1. For deblurring an image:<br>
**python deblur.py --apply --file-path='</testpath/test.png>'**<br>


2. For deblurring all images in a folder:<br>
**python deblur.py --apply --file-path='</testpath/test.png>'**<br>
Add '--result-dir=</output_path>' to specify output path. If it is not specified, the default path is './output'.<br>

3. For testing the model:<br>
**python deblur.py --test**<br>
Note that this command can only be used to test GOPRO dataset. And it will load all images into memory first. We recommand to use '--apply'
as an alternative (Item 2).<br>
Please set value of 'test_directory_path' to specify the GOPRO dataset path in file 'config.py'.<br>

4. For training a new model:<br>
**python deblur.py --train**<br>
Please remove the model file in 'model' first and set value of 'train_directory_path' to specify the GOPRO dataset path in file 'config.py'.<br>
When it finishes, run:<br>
**python deblur.py --verify**<br>


## Advanced usage
Please refer to the source code. Most configuration parameters are listed in './src/config.py'.
