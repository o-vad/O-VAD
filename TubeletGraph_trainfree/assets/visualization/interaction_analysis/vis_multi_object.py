#!/usr/bin/env python3
"""
Multi-Object State Graph Visualization
=======================================

This script creates combined visualizations for multiple tracked objects,
showing their state changes and interactions on a unified timeline.

Handles the case where TubeletGraph generates separate JSON files per object:
- 0000_1.json → Object 1 (e.g., toothpaste tube)
- 0000_2.json → Object 2 (e.g., gripper)

Usage:
    python vis_multi_object.py --json_files 0000_1.json 0000_2.json --output_dir ./vis_combined
    python vis_multi_object.py --json_pattern "0000_*.json" --output_dir ./vis_combined
"""

import json
import os
import os.path as osp
import argparse
import re
import glob
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# Visualization imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, ConnectionPatch
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StateChangeEvent:
    """Represents a state change event."""
    start_frame: int
    end_frame: int
    change_type: str
    description: str
    severity: str
    object_idx: str  # Global object ID (from filename)
    
    @property
    def severity_score(self) -> int:
        mapping = {'none': 0, 'slight': 1, 'moderate': 2, 'severe': 3}
        severity_lower = self.severity.lower()
        for key in mapping:
            if key in severity_lower:
                return mapping[key]
        return 0
    
    @property
    def mid_frame(self) -> float:
        return (self.start_frame + self.end_frame) / 2


@dataclass
class ObjectInfo:
    """Represents tracked object information."""
    obj_id: str
    description: str
    initial_state: str
    material: str
    state_changes: List[StateChangeEvent] = field(default_factory=list)
    source_file: str = ""


@dataclass 
class InteractionEvent:
    """Represents an interaction between two objects."""
    frame_start: int
    frame_end: int
    obj1_id: str
    obj2_id: str
    interaction_type: str  # concurrent, causal, etc.
    description: str = ""


# =============================================================================
# Color Schemes
# =============================================================================

class ColorScheme:
    """Professional color scheme for multi-object visualization."""
    
    SEVERITY_COLORS = {
        'none': '#E8E8E8',
        'slight': '#FFE082',
        'moderate': '#FF9800',
        'severe': '#F44336',
    }
    
    CHANGE_TYPE_COLORS = {
        'deformation': '#2196F3',
        'surface_change': '#9C27B0',
        'material_release': '#4CAF50',
        'size_change': '#FF5722',
        'texture_change': '#795548',
        'none': '#9E9E9E',
        'unknown': '#607D8B',
    }
    
    # Distinct object colors
    OBJECT_COLORS = [
        '#E91E63',  # Pink (Object 1)
        '#00BCD4',  # Cyan (Object 2)
        '#8BC34A',  # Light Green (Object 3)
        '#FF9800',  # Orange (Object 4)
        '#673AB7',  # Deep Purple (Object 5)
        '#009688',  # Teal (Object 6)
    ]
    
    BG_COLOR = '#1A1A2E'
    BG_LIGHT = '#16213E'
    BG_PANEL = '#0F3460'
    TEXT_COLOR = '#EAEAEA'
    ACCENT_COLOR = '#E94560'
    GRID_COLOR = '#2A2A4E'
    
    @classmethod
    def get_severity_color(cls, severity: str) -> str:
        severity_lower = severity.lower()
        for key, color in cls.SEVERITY_COLORS.items():
            if key in severity_lower:
                return color
        return cls.SEVERITY_COLORS['none']
    
    @classmethod
    def get_change_type_color(cls, change_type: str) -> str:
        return cls.CHANGE_TYPE_COLORS.get(
            change_type.lower(), 
            cls.CHANGE_TYPE_COLORS['unknown']
        )
    
    @classmethod
    def get_object_color(cls, obj_idx: int) -> str:
        return cls.OBJECT_COLORS[obj_idx % len(cls.OBJECT_COLORS)]


# =============================================================================
# Data Loading & Merging
# =============================================================================

def extract_object_id_from_filename(filename: str) -> str:
    """
    Extract object ID from filename pattern like '0000_1.json' or 'video_obj2.json'.
    Returns the object identifier as a string.
    """
    basename = Path(filename).stem
    
    # Try pattern: name_N (e.g., 0000_1, 0000_2)
    match = re.search(r'_(\d+)$', basename)
    if match:
        return match.group(1)
    
    # Try pattern: nameN (e.g., obj1, obj2)
    match = re.search(r'(\d+)$', basename)
    if match:
        return match.group(1)
    
    # Fallback: use full basename
    return basename


def load_and_merge_predictions(json_files: List[str]) -> Tuple[Dict[str, ObjectInfo], List[StateChangeEvent], Tuple[int, int]]:
    """
    Load multiple prediction JSON files and merge them into a unified structure.
    
    Args:
        json_files: List of JSON file paths
        
    Returns:
        Tuple of (obj_info_dict, all_events, frame_range)
    """
    obj_info_dict = {}
    all_events = []
    min_frame = float('inf')
    max_frame = 0
    
    for json_path in sorted(json_files):
        print(f"Loading: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract global object ID from filename
        global_obj_id = extract_object_id_from_filename(json_path)
        
        # Get frame range
        if 'prediction' in data:
            frames = [int(f) for f in data['prediction'].keys()]
            if frames:
                min_frame = min(min_frame, min(frames))
                max_frame = max(max_frame, max(frames))
        
        # Parse obj_info (usually has key "0" internally)
        if 'obj_info' in data:
            for local_obj_id, info in data['obj_info'].items():
                state_changes = []
                
                if 'state_changes' in info:
                    for sc in info['state_changes']:
                        event = StateChangeEvent(
                            start_frame=sc.get('start_frame', 0),
                            end_frame=sc.get('end_frame', 0),
                            change_type=sc.get('change_type', 'unknown'),
                            description=sc.get('description', ''),
                            severity=sc.get('severity', 'none'),
                            object_idx=global_obj_id  # Use global ID
                        )
                        state_changes.append(event)
                        all_events.append(event)
                
                obj_info_dict[global_obj_id] = ObjectInfo(
                    obj_id=global_obj_id,
                    description=info.get('desc', f'Object {global_obj_id}'),
                    initial_state=info.get('initial_state', 'unknown'),
                    material=info.get('material', 'unknown'),
                    state_changes=state_changes,
                    source_file=json_path
                )
        
        # Also parse top-level state_change_events
        if 'state_change_events' in data:
            for sc in data['state_change_events']:
                event = StateChangeEvent(
                    start_frame=sc.get('start_frame', 0),
                    end_frame=sc.get('end_frame', 0),
                    change_type=sc.get('change_type', 'unknown'),
                    description=sc.get('description', ''),
                    severity=sc.get('severity', 'none'),
                    object_idx=global_obj_id
                )
                # Avoid duplicates
                is_dup = any(
                    e.start_frame == event.start_frame and 
                    e.end_frame == event.end_frame and 
                    e.object_idx == event.object_idx
                    for e in all_events
                )
                if not is_dup:
                    all_events.append(event)
    
    frame_range = (int(min_frame) if min_frame != float('inf') else 0, int(max_frame))
    return obj_info_dict, all_events, frame_range


def detect_interactions(
    obj_info_dict: Dict[str, ObjectInfo],
    events: List[StateChangeEvent],
    temporal_threshold: int = 15
) -> List[InteractionEvent]:
    """
    Detect potential interactions between objects based on temporal overlap of events.
    
    Args:
        obj_info_dict: Dictionary of object information
        events: List of all state change events
        temporal_threshold: Maximum frame gap to consider as concurrent
        
    Returns:
        List of detected interaction events
    """
    interactions = []
    obj_ids = list(obj_info_dict.keys())
    
    if len(obj_ids) < 2:
        return interactions
    
    # Group events by object
    events_by_obj = defaultdict(list)
    for event in events:
        events_by_obj[event.object_idx].append(event)
    
    # Find temporal overlaps between different objects
    for i, obj1_id in enumerate(obj_ids):
        for obj2_id in obj_ids[i+1:]:
            events1 = events_by_obj.get(obj1_id, [])
            events2 = events_by_obj.get(obj2_id, [])
            
            for e1 in events1:
                for e2 in events2:
                    # Check for temporal overlap or proximity
                    overlap_start = max(e1.start_frame, e2.start_frame)
                    overlap_end = min(e1.end_frame, e2.end_frame)
                    
                    if overlap_end >= overlap_start - temporal_threshold:
                        # Events are concurrent or near-concurrent
                        interaction = InteractionEvent(
                            frame_start=min(e1.start_frame, e2.start_frame),
                            frame_end=max(e1.end_frame, e2.end_frame),
                            obj1_id=obj1_id,
                            obj2_id=obj2_id,
                            interaction_type='concurrent',
                            description=f"{e1.change_type} + {e2.change_type}"
                        )
                        
                        # Avoid duplicate interactions
                        is_dup = any(
                            abs(inter.frame_start - interaction.frame_start) < temporal_threshold and
                            {inter.obj1_id, inter.obj2_id} == {obj1_id, obj2_id}
                            for inter in interactions
                        )
                        if not is_dup:
                            interactions.append(interaction)
    
    return interactions


# =============================================================================
# Multi-Object Visualization
# =============================================================================

def create_multi_object_state_graph(
    obj_info_dict: Dict[str, ObjectInfo],
    events: List[StateChangeEvent],
    interactions: List[InteractionEvent],
    frame_range: Tuple[int, int],
    output_path: str,
    title: str = "Multi-Object State Transformation Graph",
    fps: int = 30
):
    """
    Create a comprehensive multi-object state graph visualization.
    """
    num_objects = len(obj_info_dict)
    fig_height = max(10, 4 + num_objects * 2)
    
    fig = plt.figure(figsize=(18, fig_height), facecolor=ColorScheme.BG_COLOR)
    
    # Create grid layout
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.5, 0.3, num_objects + 1, 1], hspace=0.25)
    
    # =========================================================================
    # Top Panel: Object Cards
    # =========================================================================
    ax_top = fig.add_subplot(gs[0])
    ax_top.set_facecolor(ColorScheme.BG_LIGHT)
    ax_top.set_xlim(0, 10)
    ax_top.set_ylim(0, 2)
    ax_top.axis('off')
    
    # Title
    ax_top.text(5, 1.85, title, fontsize=18, fontweight='bold', 
                color=ColorScheme.TEXT_COLOR, ha='center', va='top',
                fontfamily='monospace')
    
    # Draw object cards
    if num_objects > 0:
        card_width = min(4, 9 / num_objects)
        spacing = 10 / (num_objects + 1)
        
        for i, (obj_id, obj_info) in enumerate(sorted(obj_info_dict.items())):
            x = spacing * (i + 1) - card_width / 2
            y = 0.2
            
            obj_color = ColorScheme.get_object_color(i)
            
            # Card background
            card = FancyBboxPatch(
                (x, y), card_width, 1.4,
                boxstyle="round,pad=0.03,rounding_size=0.1",
                facecolor=ColorScheme.BG_PANEL,
                edgecolor=obj_color,
                linewidth=3,
                alpha=0.95
            )
            ax_top.add_patch(card)
            
            # Color indicator bar
            indicator = Rectangle((x, y + 1.25), card_width, 0.15,
                                   facecolor=obj_color, edgecolor='none')
            ax_top.add_patch(indicator)
            
            # Object info text
            lines = [
                f"Object {obj_id}",
                obj_info.description[:30] + "..." if len(obj_info.description) > 30 else obj_info.description,
                f"State: {obj_info.initial_state[:20]}",
                f"Material: {obj_info.material[:20]}",
                f"Changes: {len(obj_info.state_changes)}"
            ]
            
            for j, line in enumerate(lines):
                fontsize = 10 if j == 0 else 7
                fontweight = 'bold' if j == 0 else 'normal'
                ax_top.text(
                    x + card_width / 2, y + 1.15 - j * 0.22,
                    line, fontsize=fontsize, color=ColorScheme.TEXT_COLOR,
                    ha='center', va='center', fontfamily='monospace',
                    fontweight=fontweight
                )
    
    # =========================================================================
    # Interaction Summary Panel
    # =========================================================================
    ax_inter = fig.add_subplot(gs[1])
    ax_inter.set_facecolor(ColorScheme.BG_LIGHT)
    ax_inter.axis('off')
    ax_inter.set_xlim(0, 10)
    ax_inter.set_ylim(0, 1)
    
    if interactions:
        ax_inter.text(0.5, 0.5, f"🔗 Detected {len(interactions)} interaction events between objects",
                      fontsize=10, color=ColorScheme.ACCENT_COLOR,
                      va='center', fontfamily='monospace', fontweight='bold')
    else:
        ax_inter.text(0.5, 0.5, "ℹ️ Objects tracked independently (no temporal interaction detected)",
                      fontsize=10, color=ColorScheme.TEXT_COLOR, alpha=0.7,
                      va='center', fontfamily='monospace')
    
    # =========================================================================
    # Main Timeline Panel
    # =========================================================================
    ax_main = fig.add_subplot(gs[2])
    ax_main.set_facecolor(ColorScheme.BG_LIGHT)
    
    min_frame, max_frame = frame_range
    total_frames = max_frame - min_frame + 1
    margin = total_frames * 0.05
    
    ax_main.set_xlim(min_frame - margin, max_frame + margin)
    ax_main.set_ylim(-1, num_objects + 0.5)
    
    # Draw time grid
    time_markers = np.linspace(min_frame, max_frame, 12).astype(int)
    for t in time_markers:
        ax_main.axvline(x=t, color=ColorScheme.GRID_COLOR, linewidth=0.5, 
                        alpha=0.5, linestyle='-')
    
    # Draw object lanes
    sorted_obj_ids = sorted(obj_info_dict.keys())
    obj_id_to_row = {obj_id: i for i, obj_id in enumerate(sorted_obj_ids)}
    
    for i, obj_id in enumerate(sorted_obj_ids):
        obj_color = ColorScheme.get_object_color(i)
        
        # Lane background
        ax_main.axhspan(i - 0.4, i + 0.4, alpha=0.15, color=obj_color)
        
        # Lane separator
        ax_main.axhline(y=i - 0.4, color=ColorScheme.GRID_COLOR, 
                        linewidth=1, alpha=0.3)
        
        # Object label
        obj_desc = obj_info_dict[obj_id].description
        label = f"Obj {obj_id}: {obj_desc[:15]}..." if len(obj_desc) > 15 else f"Obj {obj_id}: {obj_desc}"
        ax_main.text(
            min_frame - margin * 0.8, i,
            label,
            fontsize=9, color=obj_color,
            ha='right', va='center',
            fontfamily='monospace', fontweight='bold'
        )
    
    # Draw state change events
    for event in events:
        row = obj_id_to_row.get(event.object_idx, 0)
        obj_color = ColorScheme.get_object_color(row)
        
        bar_height = 0.65
        bar_y = row - bar_height / 2
        
        # Event bar with change type color
        bar_color = ColorScheme.get_change_type_color(event.change_type)
        severity_color = ColorScheme.get_severity_color(event.severity)
        
        bar = FancyBboxPatch(
            (event.start_frame, bar_y),
            max(event.end_frame - event.start_frame, 2),
            bar_height,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=bar_color,
            edgecolor=severity_color,
            linewidth=1.5 + event.severity_score,
            alpha=0.75 + event.severity_score * 0.08
        )
        ax_main.add_patch(bar)
        
        # Event label for wider bars
        if event.end_frame - event.start_frame > total_frames * 0.04:
            mid_x = (event.start_frame + event.end_frame) / 2
            ax_main.text(
                mid_x, row,
                event.change_type[:8],
                fontsize=6, color='white',
                ha='center', va='center',
                fontfamily='monospace', fontweight='bold'
            )
    
    # Draw interaction connectors
    for inter in interactions:
        row1 = obj_id_to_row.get(inter.obj1_id, 0)
        row2 = obj_id_to_row.get(inter.obj2_id, 1)
        mid_frame = (inter.frame_start + inter.frame_end) / 2
        
        # Draw connection line
        ax_main.plot([mid_frame, mid_frame], [row1, row2], 
                     color=ColorScheme.ACCENT_COLOR, linewidth=2, 
                     alpha=0.6, linestyle='--')
        
        # Draw interaction marker
        ax_main.scatter([mid_frame], [(row1 + row2) / 2], 
                        color=ColorScheme.ACCENT_COLOR, s=80, 
                        marker='D', zorder=10, edgecolor='white', linewidth=1.5)
    
    # Time axis
    ax_main.axhline(y=-0.6, color=ColorScheme.TEXT_COLOR, linewidth=1.5)
    for t in time_markers:
        time_sec = t / fps
        ax_main.text(t, -0.85, f"{time_sec:.1f}s", fontsize=7, 
                     color=ColorScheme.TEXT_COLOR, ha='center', va='top',
                     fontfamily='monospace')
    
    ax_main.set_xlabel("Frame / Time", fontsize=11, color=ColorScheme.TEXT_COLOR,
                       fontfamily='monospace', labelpad=25)
    ax_main.tick_params(colors=ColorScheme.TEXT_COLOR, length=0)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    
    # =========================================================================
    # Legend Panel
    # =========================================================================
    ax_leg = fig.add_subplot(gs[3])
    ax_leg.set_facecolor(ColorScheme.BG_LIGHT)
    ax_leg.set_xlim(0, 10)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis('off')
    
    # Change type legend
    ax_leg.text(0.3, 0.8, "Change Types:", fontsize=9, 
                color=ColorScheme.TEXT_COLOR, fontweight='bold',
                fontfamily='monospace')
    
    change_types = ['deformation', 'surface_change', 'material_release', 'size_change']
    for i, ct in enumerate(change_types):
        x = 0.3 + i * 2.3
        rect = Rectangle((x, 0.55), 0.25, 0.15, 
                          facecolor=ColorScheme.get_change_type_color(ct),
                          edgecolor='white', linewidth=1)
        ax_leg.add_patch(rect)
        ax_leg.text(x + 0.35, 0.62, ct, fontsize=7, 
                    color=ColorScheme.TEXT_COLOR, va='center',
                    fontfamily='monospace')
    
    # Severity legend
    ax_leg.text(0.3, 0.35, "Severity (border):", fontsize=9, 
                color=ColorScheme.TEXT_COLOR, fontweight='bold',
                fontfamily='monospace')
    
    severities = ['slight', 'moderate', 'severe']
    for i, sev in enumerate(severities):
        x = 0.3 + i * 2.3
        rect = Rectangle((x, 0.1), 0.25, 0.15, 
                          facecolor=ColorScheme.BG_PANEL,
                          edgecolor=ColorScheme.get_severity_color(sev),
                          linewidth=2 + i)
        ax_leg.add_patch(rect)
        ax_leg.text(x + 0.35, 0.17, sev, fontsize=7, 
                    color=ColorScheme.TEXT_COLOR, va='center',
                    fontfamily='monospace')
    
    # Interaction legend
    ax_leg.plot([7.5, 7.8], [0.62, 0.62], color=ColorScheme.ACCENT_COLOR, 
                linewidth=2, linestyle='--')
    ax_leg.scatter([7.65], [0.62], color=ColorScheme.ACCENT_COLOR, s=50, 
                   marker='D', edgecolor='white', linewidth=1)
    ax_leg.text(7.95, 0.62, "Interaction", fontsize=7, 
                color=ColorScheme.TEXT_COLOR, va='center',
                fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=ColorScheme.BG_COLOR,
                bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print(f"Saved multi-object state graph to: {output_path}")


def create_interaction_matrix(
    obj_info_dict: Dict[str, ObjectInfo],
    events: List[StateChangeEvent],
    frame_range: Tuple[int, int],
    output_path: str,
    fps: int = 30
):
    """
    Create a heatmap showing temporal correlation between object state changes.
    """
    num_objects = len(obj_info_dict)
    if num_objects < 2:
        print("Need at least 2 objects for interaction matrix")
        return
    
    min_frame, max_frame = frame_range
    num_bins = 20
    bin_edges = np.linspace(min_frame, max_frame, num_bins + 1)
    
    # Create activity matrix per object
    sorted_obj_ids = sorted(obj_info_dict.keys())
    activity_matrix = np.zeros((num_objects, num_bins))
    
    for event in events:
        obj_idx = sorted_obj_ids.index(event.object_idx) if event.object_idx in sorted_obj_ids else -1
        if obj_idx < 0:
            continue
        
        for bin_idx in range(num_bins):
            if bin_edges[bin_idx] <= event.mid_frame < bin_edges[bin_idx + 1]:
                activity_matrix[obj_idx, bin_idx] += event.severity_score + 1
                break
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=ColorScheme.BG_COLOR)
    
    # Left: Activity heatmap
    ax1 = axes[0]
    ax1.set_facecolor(ColorScheme.BG_LIGHT)
    
    im = ax1.imshow(activity_matrix, aspect='auto', cmap='YlOrRd',
                     extent=[min_frame, max_frame, -0.5, num_objects - 0.5])
    
    ax1.set_yticks(range(num_objects))
    ax1.set_yticklabels([f"Obj {oid}" for oid in sorted_obj_ids], 
                         color=ColorScheme.TEXT_COLOR, fontfamily='monospace')
    ax1.set_xlabel("Frame", color=ColorScheme.TEXT_COLOR, fontfamily='monospace')
    ax1.set_title("State Change Activity Over Time", color=ColorScheme.TEXT_COLOR,
                   fontfamily='monospace', fontweight='bold')
    ax1.tick_params(colors=ColorScheme.TEXT_COLOR)
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Activity Intensity', color=ColorScheme.TEXT_COLOR)
    cbar.ax.yaxis.set_tick_params(color=ColorScheme.TEXT_COLOR)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=ColorScheme.TEXT_COLOR)
    
    # Right: Correlation matrix
    ax2 = axes[1]
    ax2.set_facecolor(ColorScheme.BG_LIGHT)
    
    if num_objects >= 2:
        corr_matrix = np.corrcoef(activity_matrix)
        corr_matrix = np.nan_to_num(corr_matrix)  # Handle NaN
        
        im2 = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        ax2.set_xticks(range(num_objects))
        ax2.set_yticks(range(num_objects))
        ax2.set_xticklabels([f"Obj {oid}" for oid in sorted_obj_ids],
                            color=ColorScheme.TEXT_COLOR, fontfamily='monospace', rotation=45)
        ax2.set_yticklabels([f"Obj {oid}" for oid in sorted_obj_ids],
                            color=ColorScheme.TEXT_COLOR, fontfamily='monospace')
        ax2.set_title("Temporal Correlation Matrix", color=ColorScheme.TEXT_COLOR,
                       fontfamily='monospace', fontweight='bold')
        
        # Add correlation values
        for i in range(num_objects):
            for j in range(num_objects):
                ax2.text(j, i, f"{corr_matrix[i, j]:.2f}",
                         ha='center', va='center', color='white',
                         fontsize=10, fontweight='bold')
        
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Correlation', color=ColorScheme.TEXT_COLOR)
        cbar2.ax.yaxis.set_tick_params(color=ColorScheme.TEXT_COLOR)
        plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color=ColorScheme.TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=ColorScheme.BG_COLOR,
                bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print(f"Saved interaction matrix to: {output_path}")


def create_combined_timeline(
    obj_info_dict: Dict[str, ObjectInfo],
    events: List[StateChangeEvent],
    frame_range: Tuple[int, int],
    output_path: str,
    fps: int = 30
):
    """
    Create a detailed combined timeline showing all events from all objects.
    """
    sorted_events = sorted(events, key=lambda e: (e.start_frame, e.object_idx))
    
    fig_height = max(8, len(sorted_events) * 0.4)
    fig, ax = plt.subplots(figsize=(16, fig_height), facecolor=ColorScheme.BG_COLOR)
    ax.set_facecolor(ColorScheme.BG_LIGHT)
    
    min_frame, max_frame = frame_range
    sorted_obj_ids = sorted(obj_info_dict.keys())
    
    for i, event in enumerate(sorted_events):
        y = len(sorted_events) - i - 1
        
        obj_idx = sorted_obj_ids.index(event.object_idx) if event.object_idx in sorted_obj_ids else 0
        obj_color = ColorScheme.get_object_color(obj_idx)
        bar_color = ColorScheme.get_change_type_color(event.change_type)
        severity_color = ColorScheme.get_severity_color(event.severity)
        
        # Background bar
        ax.barh(y, max_frame - min_frame, left=min_frame, height=0.8,
                color=ColorScheme.BG_COLOR, alpha=0.3)
        
        # Event bar
        ax.barh(y, event.end_frame - event.start_frame, left=event.start_frame,
                height=0.7, color=bar_color, alpha=0.85,
                edgecolor=severity_color, linewidth=2)
        
        # Object indicator
        ax.barh(y, 3, left=min_frame - (max_frame - min_frame) * 0.02, height=0.7,
                color=obj_color, alpha=0.9)
        
        # Label
        time_start = event.start_frame / fps
        time_end = event.end_frame / fps
        label = f"Obj {event.object_idx} | [{time_start:.1f}s-{time_end:.1f}s] {event.change_type} ({event.severity})"
        
        ax.text(min_frame - (max_frame - min_frame) * 0.03, y, label,
                fontsize=7, color=ColorScheme.TEXT_COLOR,
                ha='right', va='center', fontfamily='monospace')
        
        # Description
        desc = event.description[:60] + "..." if len(event.description) > 60 else event.description
        ax.text(event.end_frame + (max_frame - min_frame) * 0.01, y, desc,
                fontsize=5, color=ColorScheme.TEXT_COLOR,
                ha='left', va='center', fontfamily='monospace', alpha=0.7)
    
    ax.set_xlim(min_frame - (max_frame - min_frame) * 0.25, 
                max_frame + (max_frame - min_frame) * 0.4)
    ax.set_ylim(-0.5, len(sorted_events) - 0.5)
    ax.set_xlabel("Frame", fontsize=10, color=ColorScheme.TEXT_COLOR,
                  fontfamily='monospace')
    ax.set_title("Combined Multi-Object Timeline", fontsize=14,
                 color=ColorScheme.TEXT_COLOR, fontfamily='monospace',
                 fontweight='bold', pad=20)
    
    ax.tick_params(colors=ColorScheme.TEXT_COLOR)
    ax.spines['bottom'].set_color(ColorScheme.TEXT_COLOR)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=ColorScheme.BG_COLOR,
                bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print(f"Saved combined timeline to: {output_path}")


def generate_combined_report(
    obj_info_dict: Dict[str, ObjectInfo],
    events: List[StateChangeEvent],
    interactions: List[InteractionEvent],
    frame_range: Tuple[int, int],
    output_path: str,
    fps: int = 30
) -> str:
    """Generate a combined text report for all objects."""
    
    min_frame, max_frame = frame_range
    duration = (max_frame - min_frame) / fps
    
    lines = [
        "=" * 80,
        "MULTI-OBJECT STATE TRANSFORMATION ANALYSIS REPORT",
        "=" * 80,
        "",
        f"Video Duration: {duration:.2f} seconds ({max_frame - min_frame + 1} frames)",
        f"Frame Rate: {fps} fps",
        f"Total Objects Tracked: {len(obj_info_dict)}",
        f"Total State Change Events: {len(events)}",
        f"Detected Interactions: {len(interactions)}",
        "",
    ]
    
    # Object summaries
    lines.extend([
        "-" * 80,
        "OBJECT SUMMARIES",
        "-" * 80,
    ])
    
    for obj_id in sorted(obj_info_dict.keys()):
        obj_info = obj_info_dict[obj_id]
        obj_events = [e for e in events if e.object_idx == obj_id]
        
        lines.extend([
            "",
            f"═══ Object {obj_id} ═══",
            f"  Description: {obj_info.description}",
            f"  Initial State: {obj_info.initial_state}",
            f"  Material: {obj_info.material}",
            f"  Source File: {obj_info.source_file}",
            f"  State Changes: {len(obj_events)}",
        ])
        
        # Event summary for this object
        type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        for e in obj_events:
            type_counts[e.change_type] += 1
            for key in ['none', 'slight', 'moderate', 'severe']:
                if key in e.severity.lower():
                    severity_counts[key] += 1
                    break
        
        lines.append(f"  Change Types: {dict(type_counts)}")
        lines.append(f"  Severities: {dict(severity_counts)}")
    
    # Interactions
    if interactions:
        lines.extend([
            "",
            "-" * 80,
            "DETECTED INTERACTIONS",
            "-" * 80,
        ])
        
        for i, inter in enumerate(interactions, 1):
            time_start = inter.frame_start / fps
            time_end = inter.frame_end / fps
            lines.extend([
                "",
                f"Interaction {i}:",
                f"  Objects: {inter.obj1_id} ↔ {inter.obj2_id}",
                f"  Time: {time_start:.2f}s - {time_end:.2f}s",
                f"  Type: {inter.interaction_type}",
                f"  Details: {inter.description}",
            ])
    
    # All events chronologically
    lines.extend([
        "",
        "-" * 80,
        "ALL STATE CHANGE EVENTS (Chronological)",
        "-" * 80,
    ])
    
    sorted_events = sorted(events, key=lambda e: e.start_frame)
    for i, event in enumerate(sorted_events, 1):
        time_start = event.start_frame / fps
        time_end = event.end_frame / fps
        lines.extend([
            "",
            f"Event {i} [Object {event.object_idx}]:",
            f"  Time: {time_start:.2f}s - {time_end:.2f}s (frames {event.start_frame}-{event.end_frame})",
            f"  Type: {event.change_type}",
            f"  Severity: {event.severity}",
            f"  Description: {event.description[:100]}{'...' if len(event.description) > 100 else ''}",
        ])
    
    lines.extend(["", "=" * 80])
    
    report = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Saved combined report to: {output_path}")
    return report


# =============================================================================
# Main Entry Point
# =============================================================================

def visualize_multi_object(
    json_files: List[str],
    output_dir: str,
    fps: int = 30,
    title: str = None
):
    """
    Main function to generate multi-object visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and merge data
    print("=" * 60)
    print("Loading and merging prediction files...")
    print("=" * 60)
    
    obj_info_dict, events, frame_range = load_and_merge_predictions(json_files)
    
    print(f"\nMerged data summary:")
    print(f"  Objects: {len(obj_info_dict)}")
    print(f"  Total events: {len(events)}")
    print(f"  Frame range: {frame_range[0]} - {frame_range[1]}")
    
    for obj_id, obj_info in sorted(obj_info_dict.items()):
        print(f"  Object {obj_id}: {obj_info.description} ({len(obj_info.state_changes)} events)")
    
    if len(events) == 0:
        print("Warning: No state change events found!")
        return
    
    # Detect interactions
    interactions = detect_interactions(obj_info_dict, events)
    print(f"  Detected interactions: {len(interactions)}")
    
    # Generate title
    if title is None:
        title = f"Multi-Object State Graph ({len(obj_info_dict)} objects)"
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)
    
    # 1. Multi-object state graph
    create_multi_object_state_graph(
        obj_info_dict, events, interactions, frame_range,
        osp.join(output_dir, "multi_object_state_graph.png"),
        title=title, fps=fps
    )
    
    # 2. Interaction matrix (if multiple objects)
    if len(obj_info_dict) >= 2:
        create_interaction_matrix(
            obj_info_dict, events, frame_range,
            osp.join(output_dir, "interaction_matrix.png"),
            fps=fps
        )
    
    # 3. Combined timeline
    create_combined_timeline(
        obj_info_dict, events, frame_range,
        osp.join(output_dir, "combined_timeline.png"),
        fps=fps
    )
    
    # 4. Combined report
    report = generate_combined_report(
        obj_info_dict, events, interactions, frame_range,
        osp.join(output_dir, "combined_report.txt"),
        fps=fps
    )
    
    print("\n" + "=" * 60)
    print("Multi-object visualization complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("Files generated:")
    print("  - multi_object_state_graph.png")
    if len(obj_info_dict) >= 2:
        print("  - interaction_matrix.png")
    print("  - combined_timeline.png")
    print("  - combined_report.txt")
    
    return {
        'obj_info': obj_info_dict,
        'events': events,
        'interactions': interactions,
        'frame_range': frame_range
    }


def get_parser():
    parser = argparse.ArgumentParser(
        description="Multi-object state graph visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize multiple JSON files
    python vis_multi_object.py --json_files 0000_1.json 0000_2.json --output_dir ./vis_combined
    
    # Use glob pattern
    python vis_multi_object.py --json_pattern "0000_*.json" --output_dir ./vis_combined
        """
    )
    parser.add_argument(
        "--json_files", "-j",
        nargs='+',
        help="List of JSON prediction files"
    )
    parser.add_argument(
        "--json_pattern", "-p",
        help="Glob pattern to match JSON files (e.g., '0000_*.json')"
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="./vis_multi_output",
        help="Output directory"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second"
    )
    parser.add_argument(
        "--title", "-t",
        default=None,
        help="Custom title"
    )
    
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    
    # Collect JSON files
    json_files = []
    if args.json_files:
        json_files.extend(args.json_files)
    if args.json_pattern:
        json_files.extend(glob.glob(args.json_pattern))
    
    if not json_files:
        print("Error: No JSON files specified. Use --json_files or --json_pattern")
        exit(1)
    
    json_files = list(set(json_files))  # Remove duplicates
    print(f"Found {len(json_files)} JSON files: {json_files}")
    
    visualize_multi_object(
        json_files=json_files,
        output_dir=args.output_dir,
        fps=args.fps,
        title=args.title
    )
