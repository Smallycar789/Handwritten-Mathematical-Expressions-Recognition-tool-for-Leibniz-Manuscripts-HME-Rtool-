# HME-Rtool



This is an annotation and recognition tool for Leibniz' handwritten math expressions.
To begin the implementation of this tool on a machine with CPU, please follow the commands below:

## Implementation

Run the application :

```
conda create -y -n HMER python=3.7
conda activate HMER
pip install opencv-python==4.5.5.64
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
conda install -y matplotlib numpy pillow six pyqt=5
pip install pytorch-lightning==1.4.9 einops==0.6.1 torchmetrics==0.6.0
pip install SIP
```

## Functionality

HME-Rtool provides a user-friendly interface for users to interact with the CoMER model and recognize the expressions immediately. It doesn't require GPU resources on the user's machine, CPU resources are sufficient for running the tool.

**(1) Recognition of handwritten mathematical expressions**

HME-Rtool interface allows users to import the images from a local directory and annotate the formulas by adding the points of polygons, finally generating the bounding boxes. Then the model will process the cropped images and output the corresponding replicable recognized LaTeX sequences.

**(2) Cropped image processing**

HME-Rtool supports a automatic save function for the cropped images, which can be activated by the users. The cropped images are saved in a local folder, with the name corresponding to the source image name and the coordinates of the first point of the polygons, which is same as the presented format in LHdataset. These images can be collected for the further research and even enrich the LHdataset. At same time, the cropped image is shown in the interface beside the recognized result, bringing a clear check.

**(3) Recording and Exporting results**

The recognition results are automatically stored in a local JSON file, named identically to the corresponding image. A one-to-one mapping is maintained, with each image associated with a single JSON file. It contains all polygons' coordinates in the reference of the image, and its recognized LaTeX sequence. These JSON files can be easily accessed and utilized for further analysis or integration into other applications.

**(4) Polygon edit**

HME-Rtool presents all polygons made in the interface automatically once the image is loaded. A list of polygons are at the right side, where with a selected polygon, the user can rewrite the recognition result or delete the polygon, then the JSON file is refreshed as well. At the same time, the boxes with expressions will be shown on the original image at the left side.

**(5) Model invocation**

For the consideration of future model upgrades or changes, the connection architecture between the interface tools and the model is designed quite simple. Under the condition of ensuring the compatibility between the dictionary and the model, the model can be changed quickly by replacing the model package (checkpoints).

## Usage Steps

1. Import an image from your local by clicking "Import images".

2. Add the points of the polygon around the formula that you want to recognize, and finally a complete bounding box. If you want to reset the polygon, please press key "ESC".

3. Click "Recognize" or press key "Enter" to begin the recognition, this may take time for the first recognition, because it needs to load the model. The result will be shown below, allowing the copy action. The cropped images will show below in preview.

4. Polygons management: Please firstly choose the polygon that you want to manage in the list, and then you can delete the whole polygon or rewrite the recognized expression. All modifications will be refreshed automatically in the JSON file. The preview of the expression image will be presented once you click the polygon in the list.

5. For further studies, if the cropped images of expressions are needed, please open "Save cropped images auto". This will help you to save the formulas part with the name in format "original image name_first point x coordinate_first point y coordinate.png".

6. All polygons created and their recognition results are saved in the corresponding JSON file, with the same name as the original image. Don't hesitate to consult the records, if you want more information. Every time when you open the image, the ancient annotations will be imported automatically.

7. The recognition model is embedded in lightning_logs folder. In <main.py>, you can choose the version of model by changing parameter in function . In this phase, we implanted the best model version_28.





**If there is any question, please send e-mails to [Ze.Qian@eleves.ec-nantes.fr](mailto:Ze.Qian@eleves.ec-nantes.fr) or [sy2424114_qian@buaa.edu.cn](mailto:sy2424114_qian@buaa.edu.cn).**



Tool created by Ze QIAN, in the support of CNRS SPHERE & LS2N IPI.
