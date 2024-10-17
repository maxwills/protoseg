REM Combining CT STL
python mergeSkinAndBone.py out_skullct_skin.stl out_skullct_bone.stl out_combined_skullct.stl
REM Combining CT Smoothed STL
python mergeSkinAndBone.py out_skullct_smoothed_skin.stl out_skullct_smoothed_bone.stl out_combined_skullct_smoothed.stl
REM Combining CT GLB
python mergeSkinAndBone.py out_skullct_skin.glb out_skullct_bone.glb out_combined_skullct.glb
REM Combining CT Smoothed GLB
python mergeSkinAndBone.py out_skullct_smoothed_skin.glb out_skullct_smoothed_bone.glb out_combined_skullct_smoothed.glb
pause