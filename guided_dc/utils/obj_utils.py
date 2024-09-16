import trimesh
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import yaml
from omegaconf import OmegaConf
from guided_dc.utils.randomization import randomize_uv
import os

ASSET_PATH = os.path.expanduser('~/.maniskill/data/scene_datasets/ai2thor/ai2thorhab-uncompressed/assets/objects')

def get_obj_asset(obj_name):
    # List to store the paths of matching .glb files
    matching_files = []

    # Convert the obj_name to lowercase for case-insensitive comparison
    obj_name_lower = obj_name.lower()

    # Loop through all files in the ASSET_PATH directory
    for file in os.listdir(ASSET_PATH):
        # Check if the file contains the obj_name (case-insensitive) and has a .glb extension
        if obj_name_lower in file.lower() and file.endswith('.glb'):
            # Append the full path of the matching file to the list
            matching_files.append(os.path.join(ASSET_PATH, file))
    
    return matching_files

def generate_table_obj(cfg_path='cfg/config.yaml', visualize=True):
    #########################
    #                       #
    #    Generate Meshes    #
    #                       #
    #########################

    config = OmegaConf.load(cfg_path)
    table_config = config.base.init.table
    table_top_thickness = table_config.thickness
    table_top_width = table_config.width
    table_top_length = table_config.length
    leg_length = table_config.leg_length
    leg_thickness = table_config.leg_thickness
    width_length_ratio = table_top_width/table_top_length

    # Create the table top
    top_mesh = trimesh.creation.box(extents=(table_top_length, table_top_width, table_top_thickness))
    # Create the table legs (4 legs)
    legs = []
    leg_positions = [
        (-table_top_length/2 + leg_thickness/2, -table_top_width/2 + leg_thickness/2),  # Front-left
        (-table_top_length/2 + leg_thickness/2, table_top_width/2 - leg_thickness/2),   # Back-left
        (table_top_length/2 - leg_thickness/2, -table_top_width/2 + leg_thickness/2),   # Front-right
        (table_top_length/2 - leg_thickness/2, table_top_width/2 - leg_thickness/2)    # Back-right
    ]
    for pos in leg_positions:
        leg = trimesh.creation.box(extents=(leg_thickness, leg_thickness, leg_length))
        leg.apply_translation([pos[0], pos[1], -table_top_thickness/2 - leg_length/2])
        legs.append(leg)

    table = trimesh.util.concatenate([top_mesh] + legs)

    #########################
    #                       #
    #      Attach images    #
    #                       #
    #########################

    image_path = 'assets/table/table_top_larger.jpg'
    im = Image.open(image_path)
    def crop_to_ratio(image, target_ratio):
        """
        Crop an image to the specified aspect ratio.

        Args:
        - image (str or PIL.Image.Image): The input image or its file path.
        - target_ratio (float): The desired width-to-height aspect ratio (e.g., 2/1 for 2:1).

        Returns:
        - PIL.Image.Image: The cropped image.
        """
        
        # Load the image if a path is given
        if isinstance(image, str):
            image = Image.open(image)
        
        # Get current image size
        width, height = image.size
        
        # Calculate the current aspect ratio of the image
        current_ratio = width / height
        
        if current_ratio > target_ratio:
            # Image is too wide, crop the width
            new_width = int(height * target_ratio)
            left = (width - new_width) // 2
            right = left + new_width
            top = 0
            bottom = height
        else:
            # Image is too tall, crop the height
            new_height = int(width / target_ratio)
            top = (height - new_height) // 2
            bottom = top + new_height
            left = 0
            right = width
        
        # Crop the image to the calculated bounds
        cropped_image = image.crop((left, top, right, bottom))
        
        return cropped_image

    cropped_image = crop_to_ratio(im, table_top_length/table_top_width)

    if visualize:
        plt.imshow(np.array(im))
        plt.axis('off')  # Hide axis
        plt.show()
        plt.imshow(np.array(cropped_image))
        plt.axis('off')  # Hide axis
        plt.show()

    # Define UV coordinates for the top face of the tabletop
    # Assuming that the top face vertices are known
    
    # Example usage
    top_uv = np.array([
        [0, 0],  # bottom left
        [0, 1],  # bottom right
        [1, 0],  # top right
        [1, 1]   # top left
    ])
    image_width, image_height = im.size

    # Randomize the UV coordinates
    randomized_uv = randomize_uv(top_uv, image_width/image_height, shift_range=0.1)

    # Assign the texture only to the top face of the tabletop
    # Identify the top face (in this case, the face with the highest z value)
    top_face_idx = np.where(np.isclose(top_mesh.vertices[:, 2], table_top_thickness / 2))[0]

    # Create a blank UV map for the entire table
    uv_map = np.zeros((len(table.vertices), 2))

    # Assign the UV coordinates to the top face vertices
    uv_map[top_face_idx] = top_uv

    # Assign the texture and UV map to the tabletop
    table.visual = trimesh.visual.TextureVisuals(uv=uv_map, image=cropped_image)

    # Apply the translation to the mesh
    table.apply_translation([0, 0, -table_top_thickness/2]
    )

    #########################
    #                       #
    # Export and visualize  #
    #                       #
    #########################


    # Export the table with texture

    table.export('assets/table/table_with_textured_top.obj')
    if visualize:
        table.show()

if __name__ == '__main__':
    generate_table_obj()