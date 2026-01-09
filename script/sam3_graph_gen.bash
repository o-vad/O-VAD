# Text prompt mode (finds all instances of each concept)
python tubeletgraph_sam3_workflow.py \
    --video cooking.mp4 \
    --output_dir ./my_dataset \
    --text_prompts "apple" "knife" \
    --run_tubeletgraph

# Interactive clicking mode  
python tubeletgraph_sam3_workflow.py \
    --video cooking.mp4 \
    --output_dir ./my_dataset \
    --interactive