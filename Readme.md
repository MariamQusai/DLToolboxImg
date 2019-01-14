# DLToolboxImg



A set of helper functions that one repeatedly need to construct a dataset from raw images, visualise the performance of a neural network while it is getting trained, evaluate the performance of a model after training is completed. 
This is a work in pogress. The end product would be similar to the below animation. 



|  **Task**                                              |  **Completed**
|-------------------------------------------------------|--------------------
[Develop Toolbox](https://github.com/x110/DLToolboxImg/blob/master/DLToolboxImg.ipynb)            |:white_check_mark:
[Construct dataset](https://github.com/x110/DLToolboxImg/blob/master/NoduleSegmentation.ipynb)                                    |:white_check_mark:
[Train Model](https://github.com/x110/DLToolboxImg/blob/master/NoduleSegmentationModel.ipynb)                                    |:white_check_mark:
[Evaluate Trained Model](https://github.com/x110/DLToolboxImg/blob/master/Evaluate_trained_model.ipynb)                                    |:white_check_mark:
[Prototype of System](https://github.com/x110/DLToolboxImg/blob/master/Prototype.ipynb)                                    |:white_check_mark:
[Evaluate Model while Training](https://github.com/x110/DLToolboxImg/blob/master/monitor_performance_while_training.ipynb)                                    |:white_check_mark:
[Build interface to Configure Experiments]()                                    |

![Alt Text](https://raw.githubusercontent.com/x110/DLToolboxImg/master/Chest_Cavity.gif)


|  **Model architecture**  |  **epochs**  |  **Data augmentation** |  **optimization function** |**Dice score**
|--------------------------|--------------|------------------------|------------------------|-------------------|
[Modified UNet with Dropout=0.6](https://github.com/x110/DLToolboxImg/blob/master/NoduleSegmentationModel.ipynb)| 40  |  None |  dice loss |-0.64328116
[Modified UNet with Dropout=0.6](https://github.com/x110/DLToolboxImg/blob/master/NoduleSegmentationModel2.ipynb)| 80 |  Horizantal + vertical flips |  dice loss| -0.68878084
[Modified UNet with Dropout=0.4](https://github.com/x110/DLToolboxImg/blob/master/NoduleSegmentationModel3.ipynb)| 60 |  Horizantal + vertical flips |  dice loss| -0.602
[Modified UNet with Dropout=0.6](https://github.com/x110/DLToolboxImg/blob/master/NoduleSegmentationModel5.ipynb)|  |  Horizantal + vertical flips |  weighted cross-entropy w=[.9,.1]| 
[Modified UNet with Dropout=0.6, train only on positive data](https://github.com/x110/DLToolboxImg/blob/master/NoduleSegmentationModel8.ipynb)|  |  Horizantal + vertical flips |  dice loss| [-0.62](https://github.com/x110/DLToolboxImg/blob/master/models/model8)
[Modified UNet with Dropout=0.6, train only on positive data, cosine annealing lr schedule](https://github.com/x110/DLToolboxImg/blob/master/NoduleSegmentationModel10.ipynb)|  |  Horizantal + vertical flips |  dice loss| [-0.62](https://github.com/x110/DLToolboxImg/blob/master/models/model10)
[Modified UNet with Dropout=0.6, train only on positive data](https://github.com/x110/DLToolboxImg/blob/master/NoduleSegmentationModel9.ipynb)|  |  Horizantal + vertical flips + elastic transform |  dice loss|-.56
[Modified UNet with Dropout=0.6, train only on positive data, cosine annealing lr schedule](https://github.com/x110/DLToolboxImg/blob/master/NoduleSegmentationModel11.ipynb)|  |  Horizantal + vertical flips + random rotations [-10,10]|  dice loss| -.63
[Modified UNet with Dropout=0.6, train only on positive data, cosine annealing lr schedule](https://github.com/x110/DLToolboxImg/blob/master/NoduleSegmentationModel12.ipynb)|  |  Horizantal + vertical flips + random rotations [-20,20]|  dice loss| -.63
[Modified UNet with Dropout=0.6, train only on positive data, cosine annealing lr schedule](https://github.com/x110/DLToolboxImg/blob/master/NoduleSegmentationModel13.ipynb)|  |  Horizantal + vertical flips + random rotations [-20,20] + elastic distortion(alpha =34, sigma=4 )|  dice loss| 
[Modified UNet with Dropout=0.6, train only on positive data, cosine annealing lr schedule](https://github.com/x110/DLToolboxImg/blob/master/NoduleSegmentationModel14.ipynb)|  |  Horizantal + vertical flips + random rotations [-20,20] + elastic distortion(alpha =34, sigma=3 )|  dice loss| 
<!--
##  Check the most recent notebook [here](https://github.com/x110/DLToolboxImg/blob/master/DLToolboxImg_3.ipynb)                                    |:white_check_mark:
)
|  **Task**                                              |  **Completed**
|-------------------------------------------------------|--------------------
[Download dataset](https://github.com/x110/DLToolboxImg/blob/master/DL_02_PreProcessing/download_dataset.ipynb)            |:white_check_mark:
[Read DICOM data](https://github.com/x110/DLToolboxImg/blob/master/DL_002_load_data.ipynb)                                    |:white_check_mark:
[Split data to train, validate, and test](https://github.com/x110/DLToolboxImg/blob/master/DL_003_filter_nodules_by_diameter.ipynb)              |  :white_check_mark:
[Filter data by nodule size](https://github.com/x110/DLToolboxImg/blob/master/DL_003_filter_nodules_by_diameter.ipynb)              |  :white_check_mark:
[Distribution of nodule diameter](https://github.com/x110/DLToolboxImg/blob/master/DL_003_filter_nodules_by_diameter.ipynb)              |  :white_check_mark:
[Preprocessing:convert to HU units](https://github.com/x110/DLToolboxImg/blob/master/DL_004_Preprocessing_convert_to_Hounsfields_Unit.ipynb)              |  :white_check_mark:
[Preprocessing:Resample scans to uniform resolution](https://github.com/x110/DLToolboxImg/blob/master/DL_005_Preprocessing_resample_to_new_resolution.ipynb)              |  :white_check_mark:
[Distribution of original scan resolutions](https://github.com/x110/DLToolboxImg/blob/master/DL_005_Preprocessing_resample_to_new_resolution.ipynb)              |  :white_check_mark:
[Preprocessing: Normalization](https://github.com/x110/DLToolboxImg/blob/master/DL_006_Preprocessing_Normalization.ipynb)              |  :white_check_mark:
[Find center of nodules](https://github.com/x110/DLToolboxImg/blob/master/DL_008_find_nodule_center.ipynb)              |  :white_check_mark:
[Find center of nodules and bounding box](https://github.com/x110/DLToolboxImg/blob/master/DL_009_find_bbox.ipynb)              |  :white_check_mark:
[Find boolean mask for all nodules in scan](https://github.com/x110/DLToolboxImg/blob/master/DL_009_find_bbox.ipynb)              |  :white_check_mark:
[Find boolean mask for lung in scan](https://github.com/x110/DLToolboxImg/blob/master/DL_010_create_lung_mask.ipynb)              |  :white_check_mark:
[Extract small 3d patches from 3-D images](https://github.com/x110/DLToolboxImg/blob/master/DL_02_PatchesExtraction3DImage.ipynb)              |  :white_check_mark:
[Evaluate Model 1](https://github.com/x110/DLToolboxImg/blob/master/DL_01_EvaluateModel.ipynb)              |  :white_check_mark:
[Evaluate Model 2](https://github.com/x110/DLToolboxImg/blob/master/DL_02_EvaluateModel.ipynb)              |  :white_check_mark:
[Train from random weights initializatoin]()              |  
[Resume training from last check point]()              |  
[Visualize model every 50 epochs]()              |  
[Visualize performance on a single full CT-scan]()              |  

--> 

