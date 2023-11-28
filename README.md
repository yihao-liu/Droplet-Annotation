# Droplet-Annotation

## Overview

The Droplet-Annotation is an interactive Python-based application designed for the annotation and refinement of droplet tracking data. It facilitates the detailed examination of tracked droplets across sequential frames, allowing users to annotate, verify, and refine tracking data with precision and ease. The refined results can be used to enhance the accuracy of droplet tracking analysis by correcting errors in automated tracking data.

## Features
- **Interactive Annotation Interface**: Facilitates manual annotation of droplet tracking, allowing users to verify and refine tracking accuracy.
- **Multi-Frame Visualization**: Enables users to inspect droplets in sequential frames, aiding in the accurate assessment of tracking data.

## Usage of annotation_widget.py
1. Clone the repository to your local machine.
```bash
git clone git@github.com:yihao-liu/Droplet-Annotation.git
```
2. Add the `annotation_widget.py` under the root directory of the `Droplet_Tracking Project`, then run the script from the command line with the required arguments:

```python
python annotation_widget.py <image_name> [options]
```
3. (Optional) If you want to use `all` mode, please add the `droplet_all_channels.py` under the root directory of the `Droplet_Tracking Project`, then run the script from the command line with the required arguments before running `annotation_widget.py`:

```python
python droplet_all_channels.py <image_name> [options]
```

### Command Line Arguments
The following arguments are required when running `annotation_widget.py`:
- **image_name**: Name of the image file for processing.
- **-v, --verbose**: (Optional) Sets the verbosity level of the script (0 for silent, 1 for verbose).
- **-m, --mode**: (Optional) Sets the visualization mode of the script (all for all channels, classical for DAPI and BF channels, by default classical).
- **-h, --help**: (Optional) Displays help message and exits.

The following arguments are only required when running `droplet_all_channels.py`:
- **image_name**: Name of the image file for processing.
- **-minr, --min_radius**: (Optional) Sets the minimum radius of the droplets to be tracked (by default 12).
- **-maxr, --max_radius**: (Optional) Sets the maximum radius of the droplets to be tracked (by default 25).
- **-h, --help**: (Optional) Displays help message and exits.


### Interactive Interface
The tool presents an interface with the following features:

- **Image Display**: Shows images of droplets across sequential frames based on tracking results.
- **Navigation Buttons**: Use 'Prev' and 'Next' buttons to navigate through frames.
- **Annotation Buttons**: 'True', 'Unsure', and 'False' buttons allow you to label the correctness of droplet tracking and the 'Delete' button removes the current annotation.
- **TextBox**: Jump to a specific tracked droplets result by entering the row number.
- **Row Number**: Displays the current tracked droplet's row number.

![Interface](/readme_image/interface.png)

### Keyboard Shortcuts
- **Right Arrow**: Move to the next frame.
- **Left Arrow**: Move to the previous frame.
- **1**: Mark the current frame as 'True'.
- **2**: Mark the current frame as 'Unsure'.
- **3**: Mark the current frame as 'False'.
- **4**: Delete the current annotation.

### Saving Annotations

When you close the tool, it automatically saves the annotations in a CSV file located in the `dslab/data/05_results` directory with the name `refined_results_<image_name>.csv`.


## Usage of annotation_widget_matlab.py


1. Clone the repository to your local machine.
```bash
git clone git@github.com:yihao-liu/Droplet-Annotation.git
```
2. Install the dependencies listed in the `requirements.txt` file. We recommend using a new virtual environment.
```bash
conda create -n <env_name> python=3.8
pip install -r requirements.txt
```
3. Place files under correct directory.
- The `.nd2` images to be annotated in the `data` directory. 
- Put the tracking results of format `.xlsx` in the `results` directory. 
- Make sure the image names and tracking result names are the same. 
- Please refer to the example files in the `results` directory. The first column should be the droplet id. For the other columns make sure the column names are the same as the ones in the example file. Anyother columns will be ignored.
4. Run the script from the command line with the required arguments.
```python
python annotation_widget_matlab.py <image_name> [options]
```

### Command Line Arguments
- **image_name**: Name of the image file for processing.
- **-v, --verbose**: (Optional) Sets the verbosity level of the script (0 for silent, 1 for verbose).
- **-h, --help**: (Optional) Displays help message and exits.

### Interactive Interface
The tool presents an interface with the following features:

- **Image Display**: Shows images of droplets across sequential frames based on tracking results.
- **Navigation Buttons**: Use 'Prev' and 'Next' buttons to navigate through frames.
- **Annotation Buttons**: 'True', 'Unsure', and 'False' buttons allow you to label the correctness of droplet tracking and the 'Delete' button removes the current annotation.
- **TextBox**: Jump to a specific tracked droplets result by entering the row number.
- **Row Number**: Displays the current tracked droplet's row number.

![Interface](/readme_image/interface.png)

### Saving Annotations

When you close the tool, it automatically saves the annotations in a CSV file located in the `results` directory with the name `refined_results_<image_name>.csv`.

## Enhancing Droplet Tracking Tasks

The results generated by this tool can be use for the enhancement of droplet tracking tasks by:

- Allowing manual verification and correction of automated tracking data.
- Providing a user-friendly interface for non-programmers in lab settings.
- Enabling researchers to fine-tune tracking data for more accurate scientific analysis. For tracking based on **Voting System**, the refined results can be used to adjust the weights of the parameter as well as the cost function. For tracking based on **Optimal Transport**, the refined results can be used to adjust the cost function, e.g., by assigning `inf` to the tracked droplet with label `False` and `0` to the tracked droplet with label `True`.

## Troubleshooting

- Issue: The script does not run.
Solution: Ensure all dependencies are installed and that the script is called with the correct arguments.
- Issue: Images are not displaying correctly.
Solution: Verify the image paths and formats are compatible with the script.
For further issues, please contact the author.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments
- Prof. Dr. Klaus Eyer and Dr. Ines Lüchtefeld for their advice and support.
- The tracking results and functions from [Droplet Tracking](https://github.com/antoinebasseto/dslab). 