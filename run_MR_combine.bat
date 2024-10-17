REM Combining CT STL
python mergeSkinAndBone.py out_MRHead_skin.stl out_MRHead_bone.stl out_combined_MRHead.stl
REM Combining CT Smoothed STL
python mergeSkinAndBone.py out_MRHead_smoothed_skin.stl out_MRHead_smoothed_bone.stl out_combined_MRHead_smoothed.stl
REM Combining CT GLB
python mergeSkinAndBone.py out_MRHead_skin.glb out_MRHead_bone.glb out_combined_MRHead.glb
REM Combining CT Smoothed GLB
python mergeSkinAndBone.py out_MRHead_smoothed_skin.glb out_MRHead_smoothed_bone.glb out_combined_MRHead_smoothed.glb
pause