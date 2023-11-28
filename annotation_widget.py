"""
Script Name: Droplet Tracking Annotation Tool
Description: This script provides an interactive tool to annotate and refine droplet tracking data.

Author: Yihao Liu
Email: yihao_work@outlook.com
Last Modified: 27-11-2023
"""

import matplotlib.pyplot as plt
import numpy as np
from data_creation.droplet_retriever import resize_patch
from matplotlib.widgets import Button
import pandas as pd
import matplotlib.widgets as widgets
import argparse
from pathlib import Path
import os

PROJECT_PATH = Path(os.getcwd())
DATA_PATH = Path(PROJECT_PATH / "data")
RAW_PATH = Path(DATA_PATH / "01_raw")
PREPROCESSED_PATH = Path(DATA_PATH / "02_preprocessed")
FEATURE_PATH = Path(DATA_PATH / "03_features")
DROPLET_PATH = Path(DATA_PATH / "04_droplet")
RESULT_PATH = Path(DATA_PATH / "05_results")
EXPERIMENT_PATH = Path(PROJECT_PATH / "experiments")

parser = argparse.ArgumentParser(
    description="Annotation widget for droplet tracking refinement.")
parser.add_argument("image_name", type=str,
                    help="Name of the image for processing")
parser.add_argument("-v", "--verbose", type=int, choices=[0, 1], default=0,
                    help="Control the verbosity of the script, 0 for silent, 1 for verbose. Default is 0.")
parser.add_argument("-m", "--mode", type=str, choices=['all', 'classic'], default='classic',
                    help="Control the mode of the script, 'all' for all channels, 'classic' for DAPI and BF only. Default is 'classic'. To use all channels, please make sure the droplet data is generated with all channels. You can run 'python droplets_all_channels.py' to generate the droplet data with all channels.")
args = parser.parse_args()

# Generate the refined tracking data


def refine_droplet_tracking(result_path, image_name):
    """
    Refine droplet tracking based on input tracking data.

    This function processes tracking data of droplets to refine the tracking 
    information by merging sequential frames. It works by taking the tracking 
    information of droplets in one frame and linking it to the next frame, 
    keeping only those droplets that appear in all frames.

    Parameters:
    - result_path (str): The path to the directory where the tracking CSV files are stored.
    - image_name (str): The name of the image which corresponds to the naming convention 
                        of the tracking files. The expected file format is 'tracking_{image_name}.csv'.

    Returns:
    - DataFrame: A DataFrame containing the refined tracking information. Each row of the 
                 DataFrame represents a droplet, and each column represents a frame. The 
                 values in the DataFrame are the droplet IDs.

    Example:
    --------
    RESULT_PATH = '/path/to/directory/'
    IMAGE_NAME = 'sample_image'
    refined_df = refine_droplet_tracking(RESULT_PATH, IMAGE_NAME)
    """

    # 1. Load the CSV file
    df = pd.read_csv(f'{result_path}/tracking_{image_name}.csv')

    all_frames = sorted(list(set(df['framePrev']).union(set(df['frameNext']))))

    # 2. Split the DataFrame based on framePrev
    grouped_dfs = {key: value for key, value in df.groupby('framePrev')}

    # 3. Start the result_df with the first frame's data
    result_df = pd.DataFrame(
        grouped_dfs[0][['dropletIdPrev', 'dropletIdNext']])
    result_df.rename(columns={'dropletIdPrev': 0,
                     'dropletIdNext': 1}, inplace=True)

    # 4. Gradually merge the grouped_dfs
    for frame in range(1, len(all_frames)-1):

        # Merge the current state of result_df with the next frame
        merged_df = pd.merge(result_df, grouped_dfs[frame],
                             left_on=frame,
                             right_on='dropletIdPrev',
                             how='inner')

        result_df = merged_df.drop(
            ['framePrev', 'frameNext', 'dropletIdPrev'], axis=1)
        result_df.rename(columns={'dropletIdNext': frame+1}, inplace=True)

        # Update the result_df
        result_df.dropna(how='any', inplace=True)

    # Iterate over columns in the DataFrame and apply zfill to each column
    for column in result_df.columns:
        result_df[column] = result_df[column].astype(str).str.zfill(4)

    # Save the result_df to CSV
    result_df.to_csv(
        f'{result_path}/refined_droplet_tracking_{image_name}.csv', index=False)

    return result_df

# Generate the refined tracked droplet data


def refine_droplet_info(IMAGE_NAME, RESULT_PATH, FEATURE_PATH, sort=False):
    """
    Process and compile droplet tracking and feature data into a single table.

    This function consolidates the tracking information from individual frames 
    and corresponding droplet features such as center coordinates, radius, and 
    cell count into one comprehensive table. The function allows for sorting 
    the columns so that each droplet's data across different frames can be 
    arranged sequentially.

    Parameters:
    - image_name (str): The unique identifier for the set of images being processed.
                        This name is used to locate the CSV files following the naming 
                        pattern 'refined_droplet_tracking_{image_name}.csv' and 
                        'droplets_{image_name}.csv'.
    - result_path (str): The directory path where the result CSV files will 
                                   be saved. Defaults to '/path/to/results/'.
    - feature_path (str): The directory path where the feature CSV files are 
                                    stored. Defaults to '/path/to/features/'.
    - sort (bool, optional): A flag that indicates whether the columns in the final 
                             table should be sorted so that each droplet's information 
                             for each frame is grouped together. Defaults to False.

    Returns:
    - DataFrame: A DataFrame where each row corresponds to a droplet across multiple 
                 frames, and each column contains the droplet's tracking and feature 
                 information for a specific frame. If sorting is applied, the columns 
                 are arranged to group each droplet's data sequentially by frame.

    Saves:
    - CSV File: Depending on the 'sort' parameter, this function saves one or two CSV 
                files at the 'result_path'. One with the compiled but unsorted data 
                ('properties_{image_name}.csv'), and if sorting is applied, another 
                with the sorted data ('properties_{image_name}_sorted.csv').

    Example:
    --------
    IMAGE_NAME = 'small_mvt_1'
    SORT_COLUMNS = True
    RESULT_PATH = 'path/to/results/'
    FEATURE_PATH = 'path/to/features/'

    processed_df = process_droplet_data(
        IMAGE_NAME, 
        sort=SORT_COLUMNS, 
        result_path=RESULT_PATH, 
        feature_path=FEATURE_PATH
    )
    """
    # Construct file paths
    table1_path = f"{RESULT_PATH}/refined_droplet_tracking_{IMAGE_NAME}.csv"
    table2_path = f"{FEATURE_PATH}/droplets_{IMAGE_NAME}.csv"

    # Read the CSV files
    table1 = pd.read_csv(table1_path)
    table2 = pd.read_csv(table2_path)

    # Process the data
    table1_long = table1.reset_index().melt(
        id_vars='index', var_name='frame', value_name='droplet_id')
    table1_long['frame'] = pd.to_numeric(table1_long['frame'])
    table2['frame'] = pd.to_numeric(table2['frame'])

    merged = pd.merge(table1_long, table2, on=[
                      'droplet_id', 'frame'], how='left')
    wide_format = merged.pivot(index='index', columns='frame',
                               values=['droplet_id', 'center_row', 'center_col', 'radius', 'nr_cells'])
    wide_format.columns = ['_'.join(map(str, col)).strip()
                           for col in wide_format.columns.values]
    final_table = wide_format.reset_index(drop=True)
    final_table['Label'] = np.nan

    # Save the unsorted final table
    # final_table.to_csv(f"{RESULT_PATH}properties_{IMAGE_NAME}.csv", index=False)

    if sort:
        # Sort the columns if requested
        final_table_sorted = pd.DataFrame(index=wide_format.index)

        # Get the list of frames
        frames = table1.columns.astype(int).tolist()

        # Arrange columns in the desired order
        for frame in frames:
            cols_for_frame = [f"{detail}_{frame}" for detail in [
                'droplet_id', 'center_row', 'center_col', 'radius', 'nr_cells']]
            cols_for_frame = [
                col for col in cols_for_frame if col in wide_format.columns]
            final_table_sorted[cols_for_frame] = wide_format[cols_for_frame]

        final_table_sorted.reset_index(drop=True, inplace=True)
        final_table_sorted['Label'] = np.nan
        # Save the sorted final table
        # final_table_sorted.to_csv(f"{RESULT_PATH}properties_{IMAGE_NAME}_sorted.csv", index=False)
        return final_table_sorted

    return final_table


# Load the refined tracking data
# Modify function to load images for a given row
def load_images_for_row(row):
    images = []
    if args.mode == 'all':
        for frame, dropletId in row.items():
            path = f'{DROPLET_PATH}/{IMAGE_NAME}/f{frame}_d{dropletId}.npy'
            image = np.load(path)
            images.append(image[0])
            images.append(image[1])
            images.append(image[2])
            images.append(image[3])
            images.append(image[4])
            flag_droplet = True
    elif args.mode == 'classic':
        for frame, dropletId in row.items():
            path = f'{DROPLET_PATH}/{IMAGE_NAME}/f{frame}_d{dropletId}.npy'
            image = np.load(path)
            if image.shape[0] == 5:
                images.append(image[0])
                images.append(image[4])
                flag_droplet = True
            else:
                images.append(image[1])  # DAPI image
                images.append(image[0])  # BF image
                flag_droplet = False
    return images, flag_droplet

# Update display for a new row
# TODO: two modes: 1. only DAPI and BF images; 2. all channels (remember to edit the figsize)


def update_display(row):
    images, flag_droplet = load_images_for_row(row)
    if args.mode == 'all':
        for col_idx, (img_0, img_1, img_2, img_3, img_4) in enumerate(zip(images[0::5], images[1::5], images[2::5], images[3::5], images[4::5])):
            axarr[0, col_idx].imshow(img_0, vmin=0)
            axarr[1, col_idx].imshow(img_1, vmin=0)
            axarr[2, col_idx].imshow(img_2, vmin=0)
            axarr[3, col_idx].imshow(img_3, vmin=0)
            axarr[4, col_idx].imshow(img_4, vmin=0)
            if col_idx == 0:
                # the original order is DAPI, FITC, TRITC, Cy5, BF
                axarr[0, col_idx].set_title("DAPI", fontsize=10)
                axarr[1, col_idx].set_title("FITC", fontsize=10)
                axarr[2, col_idx].set_title("TRITC", fontsize=10)
                axarr[3, col_idx].set_title("Cy5", fontsize=10)
                axarr[4, col_idx].set_title("BF", fontsize=10)
    elif args.mode == 'classic' and flag_droplet:
        for col_idx, (img_0, img_1) in enumerate(zip(images[0::2], images[1::2])):
            axarr[0, col_idx].imshow(img_0, vmin=0)
            axarr[1, col_idx].imshow(img_1, vmin=0)
            if col_idx == 0:
                # the original order is DAPI, BF
                axarr[0, col_idx].set_title("DAPI", fontsize=10)
                axarr[1, col_idx].set_title("BF", fontsize=10)
    elif args.mode == 'classic' and not flag_droplet:
        for col_idx, (img_0, img_1) in enumerate(zip(images[0::2], images[1::2])):
            axarr[0, col_idx].imshow(img_0, vmin=0)
            axarr[1, col_idx].imshow(img_1, vmin=0)
            if col_idx == 0:
                axarr[0, col_idx].set_title("DAPI", fontsize=10)
                axarr[1, col_idx].set_title("BF", fontsize=10)
    plt.draw()
    # put update_row_info() here
    update_row_info()
    # Check if current row has a label and display it
    update_label_text()
    # current_label = final_results.loc[refined_df.index[current_row_idx], 'Label']
    # label_text.set_text(current_label if not pd.isna(current_label) else '')


def on_true_clicked(event):
    global current_row_idx
    if args.verbose == 1:
        print(f"Row {current_row_idx} is marked true.")

    # Mark the current row for keeping, and add label 'True'
    rows_true.append(refined_df.index[current_row_idx])
    tmp = refined_df.index[current_row_idx]
    if tmp in rows_false:
        rows_false.remove(tmp)
    elif tmp in rows_unsure:
        rows_unsure.remove(tmp)

    if current_row_idx < len(refined_df) - 1:
        current_row_idx += 1
        update_display(refined_df.iloc[current_row_idx])
        # update_row_info()
        # label_text.set_text('True')  # Update label text


def on_unsure_clicked(event):
    global current_row_idx
    if args.verbose == 1:
        print(f"Row {current_row_idx} is marked unsure.")
    rows_unsure.append(refined_df.index[current_row_idx])
    tmp = refined_df.index[current_row_idx]
    if tmp in rows_false:
        rows_false.remove(tmp)
    elif tmp in rows_true:
        rows_true.remove(tmp)

    if current_row_idx < len(refined_df) - 1:
        current_row_idx += 1
        update_display(refined_df.iloc[current_row_idx])
        # update_row_info()
        # label_text.set_text('Unsure')  # Update label text


def on_false_clicked(event):
    global current_row_idx
    if args.verbose == 1:
        print(f"Row {current_row_idx} is marked false.")

    # Mark the current row for deletion, and add label 'False'
    rows_false.append(refined_df.index[current_row_idx])
    tmp = refined_df.index[current_row_idx]
    if tmp in rows_true:
        rows_true.remove(tmp)
    elif tmp in rows_unsure:
        rows_unsure.remove(tmp)

    if current_row_idx < len(refined_df) - 1:
        current_row_idx += 1
        update_display(refined_df.iloc[current_row_idx])
        # update_row_info()
        # label_text.set_text('False')  # Update label text


def on_delete_clicked(event):
    global current_row_idx
    if args.verbose == 1:
        print(f"Label of row {current_row_idx} has be removed.")
    # If the current row is marked for deletion, unmark it
    if current_row_idx in rows_false:
        rows_false.remove(current_row_idx)
    elif current_row_idx in rows_true:
        rows_true.remove(current_row_idx)
    elif current_row_idx in rows_unsure:
        rows_unsure.remove(current_row_idx)
    # if we need to update the label text
    update_display(refined_df.iloc[current_row_idx])


def on_prev_clicked(event):
    global current_row_idx
    if current_row_idx > 0:
        current_row_idx -= 1
        update_display(refined_df.iloc[current_row_idx])
        # update_row_info()


def on_next_clicked(event):
    global current_row_idx
    if current_row_idx < len(refined_df) - 1:
        current_row_idx += 1
        update_display(refined_df.iloc[current_row_idx])
        # update_row_info()


def on_close(event):
    # Label the rows in rows_false with "False", the rows in rows_true with "True", and the rows in rows_unsure with "Unsure"
    final_results.loc[rows_false, 'Label'] = "False"
    final_results.loc[rows_true, 'Label'] = "True"
    final_results.loc[rows_unsure, 'Label'] = "Unsure"

    # Save the refined_df to CSV
    final_results.to_csv(
        f"{RESULT_PATH}/refined_results_{IMAGE_NAME}.csv", index=False)
    print('Saved the refined tracked droplet data after labeling.')


def on_key(event):
    """Keyboard interaction."""
    if event.key == 'right':
        on_next_clicked(event)
    elif event.key == 'left':
        on_prev_clicked(event)
    elif event.key == '1':
        on_true_clicked(event)
    elif event.key == '2':
        on_unsure_clicked(event)
    elif event.key == '3':
        on_false_clicked(event)
    elif event.key == '4':
        on_delete_clicked(event)


def update_row_info():
    row_info_text.set_text(f"Row: {current_row_idx + 1} / {len(refined_df)}")


def update_label_text():
    if current_row_idx in rows_false:
        label_text.set_text('False')
    elif current_row_idx in rows_true:
        label_text.set_text('True')
    elif current_row_idx in rows_unsure:
        label_text.set_text('Unsure')
    else:
        label_text.set_text('')


def on_submit(text_box, text):
    """Jump to specified row when user submits a value in the TextBox."""
    global current_row_idx
    # Check if text is not empty
    if text.strip():  # This removes any leading/trailing whitespace and checks if text is not just whitespace
        try:
            # Convert text input to integer and adjust for 0-based indexing
            new_idx = int(text) - 1

            # Check if the new index is within the dataframe's bounds
            if 0 <= new_idx < len(refined_df):
                current_row_idx = new_idx
                update_display(refined_df.iloc[current_row_idx])
                # update_row_info()
            else:
                print(f"Row {text} is out of range!")
        except ValueError:
            print("Please enter a valid row number.")
    # Clear the contents of the TextBox after processing the input
    text_box.set_val('')


# Create a new figure with a specified size
# TODO: rows_true, rows_false, rows_unsure contains duplicated values, but it is ok for the current version
# Example usage:
# IMAGE_NAME = 'small_mvt_1'
IMAGE_NAME = args.image_name
refined_df = refine_droplet_tracking(RESULT_PATH, IMAGE_NAME)
if os.path.exists(f"{RESULT_PATH}/refined_results_{IMAGE_NAME}.csv"):
    final_results = pd.read_csv(
        f"{RESULT_PATH}/refined_results_{IMAGE_NAME}.csv")
else:
    final_results = refine_droplet_info(
        IMAGE_NAME, RESULT_PATH=RESULT_PATH, FEATURE_PATH=FEATURE_PATH, sort=True)

# Button callback functions
current_row_idx = 0

# Create a list to keep track of rows
rows_false = []  # Marked for deletion, and will be used for training by increasing the distance among them
rows_true = []  # Marked for keeping, and will be used for training by decrease the ditance among them
# Marked for further inspection and the ditance among them will be kept the same
rows_unsure = []

# Change to 2 rows and adjust the figsize
if args.mode == 'all':
    fig, axarr = plt.subplots(5, refined_df.shape[1], figsize=(15, 20))
elif args.mode == 'classic':
    fig, axarr = plt.subplots(2, refined_df.shape[1], figsize=(15, 8))

# Remove axis for each subplot
for ax_row in axarr:
    for ax in ax_row:
        ax.axis('off')

# Add annotation to display the current label
label_text = fig.text(0.5, 0.95, '', ha='center', transform=fig.transFigure)
# Add annotation to display the current row info
row_info_text = fig.text(0.15, 0.95, '', transform=fig.transFigure)
# Display the initial label
update_label_text()
# Display the initial row info
update_row_info()
# Define a TextBox widget for user input
# Adjust position & size as needed
ax_textbox = plt.axes([0.8, 0.925, 0.1, 0.05])  # [x, y , width, height]
text_box = widgets.TextBox(ax_textbox, 'Jump to Row:')
# Assuming text_box is created as mentioned in the previous response
text_box.on_submit(lambda text: on_submit(text_box, text))

# Display the first row
update_display(refined_df.iloc[0])  # start with the first row

# Middle position of the figure's width
middle = 0.5

# Define button widths and offsets from the middle
button_width = 0.1
offset = 0.1
button_height = 0.05
button_space = 0.01  # Vertical space between buttons and plots

# Compute bottom position based on button's height and space from plots
bottom_position = button_space + button_height

# Define the button axes uniformly distributed
ax_prev = plt.axes([middle - 3*offset, bottom_position,
                    button_width, button_height])
ax_true = plt.axes([middle - 2*offset, bottom_position,
                    button_width, button_height])
ax_unsure = plt.axes([middle - offset, bottom_position,
                      button_width, button_height])
ax_false = plt.axes([middle, bottom_position, button_width, button_height])
ax_delete = plt.axes([middle + 1*offset, bottom_position,
                      button_width, button_height])
ax_next = plt.axes([middle + 2*offset, bottom_position,
                    button_width, button_height])

# Create buttons
btn_prev = Button(ax_prev, 'Prev')
btn_true = Button(ax_true, 'True (1)')
btn_unsure = Button(ax_unsure, 'Unsure (2)')
btn_false = Button(ax_false, 'False (3)')
btn_delete = Button(ax_delete, 'Delete (4)')
btn_next = Button(ax_next, 'Next')

# Link buttons to their callback functions
btn_prev.on_clicked(on_prev_clicked)
btn_true.on_clicked(on_true_clicked)
btn_unsure.on_clicked(on_unsure_clicked)
btn_false.on_clicked(on_false_clicked)
btn_delete.on_clicked(on_delete_clicked)
btn_next.on_clicked(on_next_clicked)

# Link the key event to the callback function
plt.gcf().canvas.mpl_connect('key_press_event', on_key)

# Connect the close event to the on_close function
plt.connect('close_event', on_close)

plt.show()
