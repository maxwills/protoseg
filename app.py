import argparse
import os
import numpy as np
import tifffile as tiff
import nrrd
from skimage import measure, morphology
from stl import mesh
import pymeshlab
import time
import scipy.ndimage
import trimesh

def read_volumetric_data(file_path):
    if file_path.endswith('.tiff'):
        data = tiff.imread(file_path)
        spacing = (1.0, 1.0, 1.0)  # Default spacing for TIFF
    elif file_path.endswith('.nrrd'):
        data, header = nrrd.read(file_path)
        spacing = header.get('space directions', np.eye(3)).diagonal()
    else:
        raise ValueError("Unsupported file format")
    return data, spacing

def preprocess_volume(volume):
    denoised_volume = scipy.ndimage.median_filter(volume, size=3)
    return denoised_volume

def save_intermediate_stl(segmented, output_file):
    print(f"Saving intermediate STL to {output_file}...")
    verts, faces, _, _ = measure.marching_cubes(segmented, level=0)
    intermediate_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            intermediate_mesh.vectors[i][j] = verts[f[j], :]
    intermediate_mesh.save(output_file)
    print(f"Saved intermediate STL to {output_file}")

def convert_to_stl(segmented, output_file, spacing):
    print(f"Converting to STL and saving to {output_file}...")
    verts, faces, _, _ = measure.marching_cubes(segmented, level=0, spacing=spacing)
    skin_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            skin_mesh.vectors[i][j] = verts[f[j], :]
    skin_mesh.save(output_file)
    print(f"Converted to STL and saved to {output_file}")

def convert_to_glb(input_file, output_file):
    print(f"Converting {input_file} to GLB and saving to {output_file}...")
    mesh = trimesh.load(input_file)
    mesh.export(output_file)
    print(f"Converted to GLB and saved to {output_file}")

def smooth_and_decimate_stl(input_file, output_file, apply_decimation=True):
    print(f"Smoothing and decimating STL {input_file}...")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_file)
    
    # Apply Laplacian smoothing with specified parameters
    ms.apply_filter('apply_coord_laplacian_smoothing', stepsmoothnum=2, boundary=True, cotangentweight=True, selected=False)
    
    if apply_decimation:
        # Get the original number of faces
        original_face_count = ms.current_mesh().face_number()
        target_face_count = int(original_face_count * 0.2)  # Target 20% of the original number of faces
        
        # Apply decimation
        ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=target_face_count)
    
    ms.save_current_mesh(output_file)
    print(f"Smoothed and decimated STL saved to {output_file}")

def segment_skin(volume, params):
    segmented = np.logical_and(volume > params['lower_threshold'], volume < params['upper_threshold'])
    start_time = time.time()
    base_name = params['base_name']
    if params['save_intermediate']:
            save_intermediate_stl(segmented, f"{params['base_name']}_raw_skin.stl")
    if params['remove_small_objects']:
        print("Removing small objects...")
        cleaned = morphology.remove_small_objects(segmented, min_size=64)
        if params['save_intermediate']:
            save_intermediate_stl(cleaned, f"{base_name}_post_01_remove_small_objects_skin.stl")
    else:
        cleaned = segmented
    print(f"Small objects removal completed in {time.time() - start_time:.2f} seconds")

    # Create the shell
    shell = np.copy(cleaned)
    closing_value_shell = 8
    dilation_value_shell = 6

    print("Creating shell...")
    shell = morphology.binary_closing(shell, morphology.ball(closing_value_shell))
    shell = morphology.binary_dilation(shell, morphology.ball(dilation_value_shell))
    print("Shell created.")

    # Keep only the largest connected component in the shell
    print("Labeling connected components in shell...")
    labeled_shell, num_features_shell = measure.label(shell, return_num=True)
    print(f"Found {num_features_shell} connected components in shell.")

    print("Keeping only the largest connected component in shell...")
    regions_shell = measure.regionprops(labeled_shell)
    largest_region_shell = max(regions_shell, key=lambda r: r.area)
    shell = labeled_shell == largest_region_shell.label
    if params['save_intermediate']:
        save_intermediate_stl(shell, f"{base_name}_post_00_shell_skin.stl")
    print(f"Largest connected component in shell kept. Area: {largest_region_shell.area}")

    closing_value = 8  # Adjust this value as needed
    print("Applying dilation...")
    dilated = morphology.binary_dilation(cleaned, morphology.ball(closing_value))
    if params['save_intermediate']:
        save_intermediate_stl(dilated, f"{base_name}_post_02_dilation_skin.stl")
    print(f"Dilation completed in {time.time() - start_time:.2f} seconds")

    if params['use_mask']:
        print("Applying mask and erosion...")
        # Create a mask to exclude border regions
        border_mask = np.ones_like(dilated, dtype=bool)
        border_thickness = 1  # Adjust this value as needed
        border_mask[border_thickness:-border_thickness, border_thickness:-border_thickness, border_thickness:-border_thickness] = False
        
        # Apply erosion only to the interior
        interior = np.copy(dilated)
        interior[border_mask] = False
        eroded_interior = morphology.binary_erosion(interior, morphology.ball(closing_value))
        
        # Combine the processed interior with the dilated border regions
        cleaned[~border_mask] = eroded_interior[~border_mask]
    else:
        print("Applying erosion...")
        # Apply erosion to the entire volume
        cleaned = morphology.binary_erosion(dilated, morphology.ball(closing_value))
    print(f"Erosion completed in {time.time() - start_time:.2f} seconds")

    if params['save_intermediate']:
        save_intermediate_stl(cleaned, f"{base_name}_post_03_erosion_skin.stl")

    print("Labeling connected components...")
    labeled, num_features = measure.label(cleaned, return_num=True)
    print(f"Found {num_features} connected components.")

    print("Keeping only the largest connected component...")
    regions = measure.regionprops(labeled)
    largest_region = max(regions, key=lambda r: r.area)
    largest_component = labeled == largest_region.label
    if params['save_intermediate']:
        save_intermediate_stl(largest_component, f"{base_name}_post_04_largest_component_skin.stl")
    print(f"Largest connected component kept. Area: {largest_region.area}")

    # Intersect the processed volume with the shell
    final_result = np.logical_and(largest_component, shell)
    if params['save_intermediate']:
        save_intermediate_stl(final_result, f"{base_name}_post_05_final_result_skin.stl")

    return final_result

def segment_bone(volume, params):
    segmented = np.logical_and(volume > params['lower_threshold'], volume < params['upper_threshold'])
    start_time = time.time()
    base_name = params['base_name']

    if params['remove_small_objects']:
        print("Removing small objects...")
        cleaned = morphology.remove_small_objects(segmented, min_size=64)
        if params['save_intermediate']:
            save_intermediate_stl(cleaned, f"{base_name}_post_01_remove_small_objects_bone.stl")
    else:
        cleaned = segmented
    print(f"Small objects removal completed in {time.time() - start_time:.2f} seconds")

    closing_value = 2  # Adjust this value as needed
    print("Applying closing...")
    closed = morphology.binary_closing(cleaned, morphology.ball(closing_value))
    if params['save_intermediate']:
        save_intermediate_stl(closed, f"{base_name}_post_02_closing_bone.stl")
    print(f"Closing completed in {time.time() - start_time:.2f} seconds")

    print("Labeling connected components...")
    labeled, num_features = measure.label(closed, return_num=True)
    print(f"Found {num_features} connected components.")

    print("Keeping only the largest connected component...")
    regions = measure.regionprops(labeled)
    largest_region = max(regions, key=lambda r: r.area)
    largest_component = labeled == largest_region.label
    if params['save_intermediate']:
        save_intermediate_stl(largest_component, f"{base_name}_post_03_largest_component_bone.stl")
    print(f"Largest connected component kept. Area: {largest_region.area}")

    output_file_final = f"out_{base_name}_bone.stl"
    output_file_smoothed = f"out_{base_name}_smoothed_bone.stl"
    output_file_final_glb = f"out_{base_name}_bone.glb"
    output_file_smoothed_glb = f"out_{base_name}_smoothed_bone.glb"

    convert_to_stl(largest_component, output_file_final, params['spacing'])
    convert_to_glb(output_file_final, output_file_final_glb)
    
    print("Smoothing and decimating final STL...")
    smooth_and_decimate_stl(output_file_final, output_file_smoothed, params['apply_decimation'])
    convert_to_glb(output_file_smoothed, output_file_smoothed_glb)
    print("Process completed.")

def segment_bone_mri(volume, params):
    skin_params = params.copy()
    skin_params['lower_threshold'] = 38
    skin_params['upper_threshold'] = 270
    skin_volume = segment_skin(volume, skin_params)

    segmented = np.logical_and(volume > params['lower_threshold'], volume < params['upper_threshold'])
    start_time = time.time()
    base_name = params['base_name']

    if params['remove_small_objects']:
        print("Removing small objects...")
        cleaned = morphology.remove_small_objects(segmented, min_size=64)
        if params['save_intermediate']:
            save_intermediate_stl(cleaned, f"{base_name}_post_01_remove_small_objects_bone.stl")
    else:
        cleaned = segmented
    print(f"Small objects removal completed in {time.time() - start_time:.2f} seconds")

    closing_value = 2  # Adjust this value as needed
    print("Applying closing...")
    closed = morphology.binary_closing(cleaned, morphology.ball(closing_value))
    if params['save_intermediate']:
        save_intermediate_stl(closed, f"{base_name}_post_02_closing_bone.stl")
    print(f"Closing completed in {time.time() - start_time:.2f} seconds")

    print("Labeling connected components...")
    labeled, num_features = measure.label(closed, return_num=True)
    print(f"Found {num_features} connected components.")

    print("Keeping only the largest connected component...")
    regions = measure.regionprops(labeled)
    largest_region = max(regions, key=lambda r: r.area)
    largest_component = labeled == largest_region.label
    if params['save_intermediate']:
        save_intermediate_stl(largest_component, f"{base_name}_post_03_largest_component_bone.stl")
    print(f"Largest connected component kept. Area: {largest_region.area}")

    final_bone = np.logical_and(largest_component, skin_volume)
    output_file_final = f"out_{base_name}_bone.stl"
    output_file_smoothed = f"out_{base_name}_smoothed_bone.stl"
    output_file_final_glb = f"out_{base_name}_bone.glb"
    output_file_smoothed_glb = f"out_{base_name}_smoothed_bone.glb"

    convert_to_stl(final_bone, output_file_final, params['spacing'])
    convert_to_glb(output_file_final, output_file_final_glb)
    
    print("Smoothing and decimating final STL...")
    smooth_and_decimate_stl(output_file_final, output_file_smoothed, params['apply_decimation'])
    convert_to_glb(output_file_smoothed, output_file_smoothed_glb)
    print("Process completed.")

def main():
    parser = argparse.ArgumentParser(description="Process volumetric data and generate STL files.")
    parser.add_argument('input_file', type=str, help="Path to the input volumetric data file")
    parser.add_argument('lower_threshold', type=float, help="Lower threshold for segmentation")
    parser.add_argument('upper_threshold', type=float, help="Upper threshold for segmentation")
    parser.add_argument('segment', type=str, choices=['skin', 'bone'], help="Type of segment to extract: 'skin' or 'bone'")
    parser.add_argument('exam_type', type=str, choices=['MRI', 'CT'], help="Type of exam: 'MRI' or 'CT'")
    args = parser.parse_args()

    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]

    params = {
        'lower_threshold': args.lower_threshold,
        'upper_threshold': args.upper_threshold,
        'use_mask': True,  # Set to False to disable the mask
        'remove_small_objects': True,  # Set to False to disable removing small objects
        'padding': 12,  # Padding size
        'padding_value': args.lower_threshold - 1e-5,  # Epsilon less than the lower threshold
        'remove_padding': False,  # Set to False to keep the padding
        'apply_decimation': False,  # Set to True to enable decimation
        'base_name': input_basename,
        'save_intermediate': False,  # Set to True to save intermediate STL files
        'segment': args.segment,
        'exam_type': args.exam_type
    }

    input_file = args.input_file

    print("Reading volumetric data...")
    volume, spacing = read_volumetric_data(input_file)
    params['spacing'] = spacing
    print("Volumetric data read successfully.")

    print("Applying median filter...")
    denoised_volume = preprocess_volume(volume)
    print("Median filtering completed.")

    print("Applying padding...")
    padded_volume = np.pad(denoised_volume, pad_width=params['padding'], mode='constant', constant_values=params['padding_value'])
    print("Padding applied.")

    if params['segment'] == 'skin':
        print("Segmenting skin...")
        cleaned_skin = segment_skin(padded_volume, params)
        print("Skin segmentation completed.")

        output_file_final = f"out_{input_basename}_skin.stl"
        output_file_smoothed = f"out_{input_basename}_smoothed_skin.stl"
        output_file_final_glb = f"out_{input_basename}_skin.glb"
        output_file_smoothed_glb = f"out_{input_basename}_smoothed_skin.glb"

        if params['remove_padding']:
            print("Removing padding...")
            cleaned_skin = cleaned_skin[params['padding']:-params['padding'], params['padding']:-params['padding'], params['padding']:-params['padding']]  # Remove padding
            print("Padding removed.")
        
        convert_to_stl(cleaned_skin, output_file_final, spacing)
        convert_to_glb(output_file_final, output_file_final_glb)
        
        print("Smoothing and decimating final STL...")
        smooth_and_decimate_stl(output_file_final, output_file_smoothed, params['apply_decimation'])
        convert_to_glb(output_file_smoothed, output_file_smoothed_glb)
        print("Process completed.")
    elif params['segment'] == 'bone':
        if params['exam_type'] == 'MRI':
            segment_bone_mri(padded_volume, params)
        else:
            segment_bone(padded_volume, params)

if __name__ == "__main__":
    main()