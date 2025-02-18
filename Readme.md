# Segmentation application

input files are not in the repository, and need to be placed in the same folder as the app.
To run the app, use the batch files, or manually call the application with the same parameters as described in the batch files.

## Useful links:

Input files: https://drive.google.com/drive/folders/1sgLOBVay6pbVRaw1Ybg7oFqSraYDBBIv?usp=sharing  
Video: https://drive.google.com/file/d/19mXxeaImYbrDtY4UiuDfTBJJFeQ6beQo/view?usp=sharing  
Pregenerated output files: https://drive.google.com/drive/folders/1Co2MdXPQ4scoT-l1bwz8pjYf_Oit_Dt4?usp=sharing  

## High-Level Summary of Algorithm

1. **Input Handling**:
    - Read volumetric data from `.tiff` or `.nrrd` files.
    - Extract spacing information from the file header.

2. **Preprocessing**:
    - Apply a median filter to the input volume to denoise it.
    - Add padding to the denoised volume.

3. **Segmentation**:
    - If the segmentation type is "skin":
        - Segment the skin from the padded volume.
        - Optionally remove the padding from the segmented skin.
        - Convert the segmented skin to STL and GLB formats.
        - Smooth and optionally decimate the STL file, then convert it to GLB format.
    - If the segmentation type is "bone":
        - If the exam type is MRI, segment the bone using MRI-specific techniques.
        - Otherwise, segment the bone using general techniques.
        - Convert the segmented bone to STL and GLB formats.
        - Smooth and optionally decimate the STL file, then convert it to GLB format.

4. **Output**:
    - Save the final and smoothed STL and GLB files.

## Detailed Segmentation Steps

### Skin Segmentation

#### Segmenting Skin:
- Apply thresholding to isolate the skin tissue.
- Optionally remove small objects from the segmented volume.
- Create a shell around the segmented skin using morphological operations.
- Keep only the largest connected component in the shell.
- Apply dilation and erosion to refine the segmented skin.
- Keep only the largest connected component in the refined volume.
- Intersect the refined volume with the shell to get the final segmented skin.

#### Removing Padding:
- If padding removal is enabled, remove the padding from the segmented skin volume.

#### Conversion and Smoothing:
- Convert the cleaned skin volume to STL format.
- Convert the STL file to GLB format.
- Smooth and optionally decimate the STL file.
- Convert the smoothed STL file to GLB format.

### Bone Segmentation

#### Segmenting Bone:
- Apply thresholding to isolate the bone tissue.
- Optionally remove small objects from the segmented volume.
- Apply closing to fill small holes in the segmented bone.
- Keep only the largest connected component in the closed volume.
- If the exam type is MRI, intersect the segmented bone with the segmented skin volume.

#### Conversion and Smoothing:
- Convert the cleaned bone volume to STL format.
- Convert the STL file to GLB format.
- Smooth and optionally decimate the STL file.
- Convert the smoothed STL file to GLB format.
