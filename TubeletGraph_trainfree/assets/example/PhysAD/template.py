{
    "prediction": {frame_idx: {obj_id: RLE_mask}},  # Frame-by-frame masks
    "supix_masks": {frame_idx: {obj_id: RLE_mask}}, # Super-pixel masks
    "obj_info": {
        obj_id: {
            "desc": str,           # Object description
            "initial_state": str,  # e.g., "intact"
            "material": str,       # e.g., "plastic tube"
            "state_changes": [...]  # List of change events
        }
    },
    "state_change_events": [
        {
            "start_frame": int,
            "end_frame": int,
            "change_type": str,    # deformation, surface_change, etc.
            "description": str,
            "severity": str,       # none, slight, moderate, severe
            "object_idx": str
        }
    ]
}