import argparse
import trimesh
import os
import numpy as np

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Merge skin and bone 3D models.')
    parser.add_argument('skin_file', type=str, help='The 3D file with the skin (stl or gbl)')
    parser.add_argument('bone_file', type=str, help='The 3D file with the bone (stl or gbl)')
    parser.add_argument('output_file', type=str, help='The output 3D file (stl or gbl)')
    args = parser.parse_args()

    # Check if skin file exists
    if not os.path.isfile(args.skin_file):
        print(f"Error: Skin file '{args.skin_file}' does not exist.")
        return

    # Check if bone file exists
    if not os.path.isfile(args.bone_file):
        print(f"Error: Bone file '{args.bone_file}' does not exist.")
        return

    # Debugging: Print file paths
    print(f"Skin file path: {args.skin_file}")
    print(f"Bone file path: {args.bone_file}")
    print(f"Output file path: {args.output_file}")

    
    # Function to extract the bottom-most geometry from a scene or a single mesh
    def extract_bottom_geometry(mesh_or_scene):
        if isinstance(mesh_or_scene, trimesh.Scene):
            # If it's a scene, extract the first geometry (or the one you're interested in)
            if len(mesh_or_scene.geometry) > 0:
                # Extract the first geometry from the scene
                geometry_key = list(mesh_or_scene.geometry.keys())[0]
                return mesh_or_scene.geometry[geometry_key]
            else:
                raise ValueError("No geometries found in the scene.")
        elif isinstance(mesh_or_scene, trimesh.Trimesh):
            # If it's already a single Trimesh object, return it directly
            return mesh_or_scene
        else:
            raise TypeError("Unsupported object type. Expected Trimesh or Scene.")

    # Load the input GLB files (these files may have hierarchical structures)
    skin_model = trimesh.load(args.skin_file)
    bone_model = trimesh.load(args.bone_file)

    # Extract only the relevant geometry (either from a scene or a single mesh)
    skin = extract_bottom_geometry(skin_model)  # Extract the skin geometry
    bone = extract_bottom_geometry(bone_model)  # Extract the bone geometry

    # Create a new scene with just the skin and bone geometries
    print("Combining models...")
    skin_scene = trimesh.Scene()

    # Add the extracted skin and bone geometries to the new scene
    skin_node_name = skin_scene.add_geometry(skin)
    bone_node_name = skin_scene.add_geometry(bone)

    # Ensure normals are preserved
    if skin.vertex_normals is not None and len(skin.vertex_normals) > 0:
        skin_scene.geometry[skin_node_name].vertex_normals = skin.vertex_normals
    if bone.vertex_normals is not None and len(bone.vertex_normals) > 0:
        skin_scene.geometry[bone_node_name].vertex_normals = bone.vertex_normals

    # Create an identity matrix for no transformations
    identity_matrix = np.eye(4)

    # Update the scene graph to make bone a child of skin (establish hierarchy)
    skin_scene.graph.update(frame_to=bone_node_name, frame_from=skin_node_name, matrix=identity_matrix)

    # Export the scene
    skin_scene.export(args.output_file)
    print(f"Combined model exported to {args.output_file}.")

if __name__ == "__main__":
    main()