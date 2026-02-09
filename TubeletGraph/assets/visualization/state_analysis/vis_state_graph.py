#!/usr/bin/env python3
"""
State Graph Visualization for VLM-Enhanced TubeletGraph
========================================================

This script visualizes object-centric temporal dynamics including:
1. State transformation graphs (PDF/PNG)
2. Timeline visualization of state changes
3. Video with state change annotations

Compatible with the output format from prompt_vlm.py:
- prediction: frame-by-frame masks
- supix_masks: super-pixel masks
- obj_info: object information with state changes
- state_change_events: list of detected state changes

Usage:
    python vis_state_graph.py --json_path <PATH_TO_JSON> --frames_dir <PATH_TO_FRAMES>
    python vis_state_graph.py --json_path 0000_1.json --frames_dir ./frames --output_dir ./vis_out
"""

import json
import os
import os.path as osp
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Optional imports for video generation
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from pycocotools import mask as MaskUtils
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False


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
    object_idx: str
    
    @property
    def severity_score(self) -> int:
        """Convert severity to numeric score."""
        mapping = {'none': 0, 'slight': 1, 'moderate': 2, 'severe': 3}
        # Handle complex severity strings like "none (for the tool itself...)"
        severity_lower = self.severity.lower()
        for key in mapping:
            if key in severity_lower:
                return mapping[key]
        return 0

@dataclass
class ObjectInfo:
    """Represents tracked object information."""
    obj_id: str
    description: str
    initial_state: str
    material: str
    state_changes: List[StateChangeEvent]


# =============================================================================
# Data Loading
# =============================================================================

def load_prediction_data(json_path: str) -> dict:
    """Load prediction JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def parse_state_changes(data: dict) -> Tuple[Dict[str, ObjectInfo], List[StateChangeEvent]]:
    """Parse object info and state changes from prediction data."""
    obj_info_dict = {}
    all_events = []
    
    # Parse obj_info
    if 'obj_info' in data:
        for obj_id, info in data['obj_info'].items():
            state_changes = []
            if 'state_changes' in info:
                for sc in info['state_changes']:
                    event = StateChangeEvent(
                        start_frame=sc.get('start_frame', 0),
                        end_frame=sc.get('end_frame', 0),
                        change_type=sc.get('change_type', 'unknown'),
                        description=sc.get('description', ''),
                        severity=sc.get('severity', 'none'),
                        object_idx=str(obj_id)
                    )
                    state_changes.append(event)
                    all_events.append(event)
            
            obj_info_dict[obj_id] = ObjectInfo(
                obj_id=str(obj_id),
                description=info.get('desc', f'Object {obj_id}'),
                initial_state=info.get('initial_state', 'unknown'),
                material=info.get('material', 'unknown'),
                state_changes=state_changes
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
                object_idx=str(sc.get('object_idx', '0'))
            )
            # Avoid duplicates
            if event not in all_events:
                all_events.append(event)
    
    return obj_info_dict, all_events


def get_frame_range(data: dict) -> Tuple[int, int]:
    """Get the frame range from prediction data."""
    if 'prediction' in data:
        frames = [int(f) for f in data['prediction'].keys()]
        return min(frames), max(frames)
    return 0, 0


# =============================================================================
# Color Schemes
# =============================================================================

class ColorScheme:
    """Professional color scheme for visualization."""
    
    # Severity colors (warm progression)
    SEVERITY_COLORS = {
        'none': '#E8E8E8',      # Light gray
        'slight': '#FFE082',     # Amber light
        'moderate': '#FF9800',   # Orange
        'severe': '#F44336',     # Red
    }
    
    # Change type colors (categorical)
    CHANGE_TYPE_COLORS = {
        'deformation': '#2196F3',      # Blue
        'surface_change': '#9C27B0',   # Purple
        'material_release': '#4CAF50', # Green
        'size_change': '#FF5722',      # Deep orange
        'texture_change': '#795548',   # Brown
        'none': '#9E9E9E',             # Gray
        'unknown': '#607D8B',          # Blue gray
    }
    
    # Object colors
    OBJECT_COLORS = [
        '#E91E63',  # Pink
        '#3F51B5',  # Indigo
        '#009688',  # Teal
        '#FF5722',  # Deep orange
        '#673AB7',  # Deep purple
        '#00BCD4',  # Cyan
    ]
    
    # Background and text
    BG_COLOR = '#1A1A2E'
    BG_LIGHT = '#16213E'
    TEXT_COLOR = '#EAEAEA'
    ACCENT_COLOR = '#E94560'
    
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
# State Graph Visualization
# =============================================================================

def create_state_graph(
    obj_info_dict: Dict[str, ObjectInfo],
    events: List[StateChangeEvent],
    frame_range: Tuple[int, int],
    output_path: str,
    title: str = "Object State Transformation Graph",
    fps: int = 30
):
    """
    Create a state graph visualization showing object transformations over time.
    
    Layout:
    - Top: Object nodes with initial states
    - Middle: Timeline with state change events
    - Bottom: Legend
    """
    fig = plt.figure(figsize=(16, 10), facecolor=ColorScheme.BG_COLOR)
    
    # Create grid layout
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 5, 1], hspace=0.3)
    
    # =========================================================================
    # Top Panel: Object Overview
    # =========================================================================
    ax_top = fig.add_subplot(gs[0])
    ax_top.set_facecolor(ColorScheme.BG_LIGHT)
    ax_top.set_xlim(0, 10)
    ax_top.set_ylim(0, 2)
    ax_top.axis('off')
    
    # Title
    ax_top.text(5, 1.8, title, fontsize=16, fontweight='bold', 
                color=ColorScheme.TEXT_COLOR, ha='center', va='top',
                fontfamily='monospace')
    
    # Draw object boxes
    num_objects = len(obj_info_dict)
    if num_objects > 0:
        box_width = min(3, 8 / num_objects)
        spacing = 10 / (num_objects + 1)
        
        for i, (obj_id, obj_info) in enumerate(obj_info_dict.items()):
            x = spacing * (i + 1) - box_width / 2
            y = 0.3
            
            # Object box
            box = FancyBboxPatch(
                (x, y), box_width, 1.2,
                boxstyle="round,pad=0.05,rounding_size=0.1",
                facecolor=ColorScheme.get_object_color(i),
                edgecolor='white',
                linewidth=2,
                alpha=0.9
            )
            ax_top.add_patch(box)
            
            # Object label
            label_lines = [
                f"Object {obj_id}",
                obj_info.description[:25] + "..." if len(obj_info.description) > 25 else obj_info.description,
                f"State: {obj_info.initial_state}",
                f"Changes: {len(obj_info.state_changes)}"
            ]
            
            for j, line in enumerate(label_lines):
                ax_top.text(
                    x + box_width / 2, y + 1.0 - j * 0.25,
                    line, fontsize=8, color='white',
                    ha='center', va='center', fontfamily='monospace'
                )
    
    # =========================================================================
    # Middle Panel: Timeline
    # =========================================================================
    ax_mid = fig.add_subplot(gs[1])
    ax_mid.set_facecolor(ColorScheme.BG_LIGHT)
    
    min_frame, max_frame = frame_range
    total_frames = max_frame - min_frame + 1
    
    # Setup axes
    ax_mid.set_xlim(min_frame - total_frames * 0.02, max_frame + total_frames * 0.02)
    ax_mid.set_ylim(-0.5, num_objects + 0.5)
    
    # Timeline background
    for i in range(num_objects):
        ax_mid.axhspan(i - 0.4, i + 0.4, alpha=0.1, 
                       color=ColorScheme.get_object_color(i))
    
    # Draw timeline axis
    ax_mid.axhline(y=-0.3, color=ColorScheme.TEXT_COLOR, linewidth=1, alpha=0.5)
    
    # Time markers
    time_markers = np.linspace(min_frame, max_frame, 10).astype(int)
    for t in time_markers:
        ax_mid.axvline(x=t, color=ColorScheme.TEXT_COLOR, linewidth=0.5, 
                       alpha=0.2, linestyle='--')
        time_sec = t / fps
        ax_mid.text(t, -0.45, f"{time_sec:.1f}s", fontsize=7, 
                    color=ColorScheme.TEXT_COLOR, ha='center', va='top',
                    fontfamily='monospace')
    
    # Draw state change events
    obj_id_to_row = {obj_id: i for i, obj_id in enumerate(obj_info_dict.keys())}
    
    for event in events:
        obj_row = obj_id_to_row.get(event.object_idx, 0)
        
        # Event bar
        bar_height = 0.6
        bar_y = obj_row - bar_height / 2
        
        # Color by change type
        bar_color = ColorScheme.get_change_type_color(event.change_type)
        edge_color = ColorScheme.get_severity_color(event.severity)
        
        bar = FancyBboxPatch(
            (event.start_frame, bar_y),
            event.end_frame - event.start_frame,
            bar_height,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=bar_color,
            edgecolor=edge_color,
            linewidth=2 + event.severity_score,
            alpha=0.7 + event.severity_score * 0.1
        )
        ax_mid.add_patch(bar)
        
        # Event label (for wider events)
        if event.end_frame - event.start_frame > total_frames * 0.05:
            mid_x = (event.start_frame + event.end_frame) / 2
            ax_mid.text(
                mid_x, obj_row,
                event.change_type[:10],
                fontsize=6, color='white',
                ha='center', va='center',
                fontfamily='monospace',
                fontweight='bold'
            )
    
    # Y-axis labels (object names)
    for obj_id, row in obj_id_to_row.items():
        ax_mid.text(
            min_frame - total_frames * 0.01, row,
            f"Obj {obj_id}",
            fontsize=9, color=ColorScheme.get_object_color(row),
            ha='right', va='center',
            fontfamily='monospace', fontweight='bold'
        )
    
    ax_mid.set_xlabel("Frame", fontsize=10, color=ColorScheme.TEXT_COLOR,
                      fontfamily='monospace')
    ax_mid.tick_params(colors=ColorScheme.TEXT_COLOR)
    ax_mid.spines['bottom'].set_color(ColorScheme.TEXT_COLOR)
    ax_mid.spines['left'].set_visible(False)
    ax_mid.spines['top'].set_visible(False)
    ax_mid.spines['right'].set_visible(False)
    
    # =========================================================================
    # Bottom Panel: Legend
    # =========================================================================
    ax_bot = fig.add_subplot(gs[2])
    ax_bot.set_facecolor(ColorScheme.BG_LIGHT)
    ax_bot.set_xlim(0, 10)
    ax_bot.set_ylim(0, 1)
    ax_bot.axis('off')
    
    # Change type legend
    ax_bot.text(0.5, 0.85, "Change Types:", fontsize=9, 
                color=ColorScheme.TEXT_COLOR, fontweight='bold',
                fontfamily='monospace')
    
    change_types = ['deformation', 'surface_change', 'material_release', 'size_change']
    for i, ct in enumerate(change_types):
        x = 0.5 + i * 2.2
        rect = plt.Rectangle((x, 0.55), 0.3, 0.2, 
                              facecolor=ColorScheme.get_change_type_color(ct),
                              edgecolor='white', linewidth=1)
        ax_bot.add_patch(rect)
        ax_bot.text(x + 0.4, 0.65, ct, fontsize=7, 
                    color=ColorScheme.TEXT_COLOR, va='center',
                    fontfamily='monospace')
    
    # Severity legend
    ax_bot.text(0.5, 0.35, "Severity:", fontsize=9, 
                color=ColorScheme.TEXT_COLOR, fontweight='bold',
                fontfamily='monospace')
    
    severities = ['none', 'slight', 'moderate', 'severe']
    for i, sev in enumerate(severities):
        x = 0.5 + i * 2.2
        rect = plt.Rectangle((x, 0.05), 0.3, 0.2, 
                              facecolor=ColorScheme.BG_LIGHT,
                              edgecolor=ColorScheme.get_severity_color(sev),
                              linewidth=2 + i)
        ax_bot.add_patch(rect)
        ax_bot.text(x + 0.4, 0.15, sev, fontsize=7, 
                    color=ColorScheme.TEXT_COLOR, va='center',
                    fontfamily='monospace')
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=ColorScheme.BG_COLOR,
                bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print(f"Saved state graph to: {output_path}")


# =============================================================================
# Detailed Timeline Visualization
# =============================================================================

def create_detailed_timeline(
    events: List[StateChangeEvent],
    frame_range: Tuple[int, int],
    output_path: str,
    fps: int = 30
):
    """Create a detailed timeline showing all state change events with descriptions."""
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(events) * 0.5)), 
                           facecolor=ColorScheme.BG_COLOR)
    ax.set_facecolor(ColorScheme.BG_LIGHT)
    
    min_frame, max_frame = frame_range
    
    # Sort events by start frame
    sorted_events = sorted(events, key=lambda e: e.start_frame)
    
    # Draw events
    for i, event in enumerate(sorted_events):
        y = len(sorted_events) - i - 1
        
        # Event bar
        bar_color = ColorScheme.get_change_type_color(event.change_type)
        severity_color = ColorScheme.get_severity_color(event.severity)
        
        # Background bar (full width)
        ax.barh(y, max_frame - min_frame, left=min_frame, height=0.8,
                color=ColorScheme.BG_COLOR, alpha=0.3)
        
        # Event bar
        ax.barh(y, event.end_frame - event.start_frame, left=event.start_frame,
                height=0.7, color=bar_color, alpha=0.8,
                edgecolor=severity_color, linewidth=2)
        
        # Event label
        time_start = event.start_frame / fps
        time_end = event.end_frame / fps
        
        label = f"[{time_start:.1f}s-{time_end:.1f}s] {event.change_type} ({event.severity})"
        ax.text(min_frame - (max_frame - min_frame) * 0.01, y, label,
                fontsize=8, color=ColorScheme.TEXT_COLOR,
                ha='right', va='center', fontfamily='monospace')
        
        # Description (truncated)
        desc = event.description[:80] + "..." if len(event.description) > 80 else event.description
        ax.text(event.end_frame + (max_frame - min_frame) * 0.01, y, desc,
                fontsize=6, color=ColorScheme.TEXT_COLOR,
                ha='left', va='center', fontfamily='monospace', alpha=0.7)
    
    # Styling
    ax.set_xlim(min_frame - (max_frame - min_frame) * 0.3, 
                max_frame + (max_frame - min_frame) * 0.5)
    ax.set_ylim(-0.5, len(sorted_events) - 0.5)
    ax.set_xlabel("Frame", fontsize=10, color=ColorScheme.TEXT_COLOR,
                  fontfamily='monospace')
    ax.set_title("State Change Events Timeline", fontsize=14,
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
    
    print(f"Saved detailed timeline to: {output_path}")


# =============================================================================
# Severity Distribution Visualization
# =============================================================================

def create_severity_analysis(
    events: List[StateChangeEvent],
    output_path: str
):
    """Create analysis charts showing severity and change type distributions."""
    
    fig = plt.figure(figsize=(12, 8), facecolor=ColorScheme.BG_COLOR)
    
    # Severity distribution
    ax1 = fig.add_subplot(221)
    ax1.set_facecolor(ColorScheme.BG_LIGHT)
    
    severity_counts = defaultdict(int)
    for event in events:
        severity_lower = event.severity.lower()
        for key in ['none', 'slight', 'moderate', 'severe']:
            if key in severity_lower:
                severity_counts[key] += 1
                break
    
    severities = ['none', 'slight', 'moderate', 'severe']
    counts = [severity_counts.get(s, 0) for s in severities]
    colors = [ColorScheme.get_severity_color(s) for s in severities]
    
    ax1.bar(severities, counts, color=colors, edgecolor='white', linewidth=2)
    ax1.set_title("Severity Distribution", color=ColorScheme.TEXT_COLOR,
                  fontfamily='monospace', fontweight='bold')
    ax1.set_ylabel("Count", color=ColorScheme.TEXT_COLOR, fontfamily='monospace')
    ax1.tick_params(colors=ColorScheme.TEXT_COLOR)
    for spine in ax1.spines.values():
        spine.set_color(ColorScheme.TEXT_COLOR)
    
    # Change type distribution
    ax2 = fig.add_subplot(222)
    ax2.set_facecolor(ColorScheme.BG_LIGHT)
    
    type_counts = defaultdict(int)
    for event in events:
        type_counts[event.change_type] += 1
    
    types = list(type_counts.keys())
    counts = [type_counts[t] for t in types]
    colors = [ColorScheme.get_change_type_color(t) for t in types]
    
    ax2.barh(types, counts, color=colors, edgecolor='white', linewidth=2)
    ax2.set_title("Change Type Distribution", color=ColorScheme.TEXT_COLOR,
                  fontfamily='monospace', fontweight='bold')
    ax2.set_xlabel("Count", color=ColorScheme.TEXT_COLOR, fontfamily='monospace')
    ax2.tick_params(colors=ColorScheme.TEXT_COLOR)
    for spine in ax2.spines.values():
        spine.set_color(ColorScheme.TEXT_COLOR)
    
    # Severity over time
    ax3 = fig.add_subplot(212)
    ax3.set_facecolor(ColorScheme.BG_LIGHT)
    
    # Sort events by start frame
    sorted_events = sorted(events, key=lambda e: e.start_frame)
    frames = [e.start_frame for e in sorted_events]
    scores = [e.severity_score for e in sorted_events]
    colors = [ColorScheme.get_severity_color(e.severity) for e in sorted_events]
    
    ax3.scatter(frames, scores, c=colors, s=100, edgecolor='white', linewidth=1.5)
    ax3.plot(frames, scores, color=ColorScheme.ACCENT_COLOR, alpha=0.5, linewidth=2)
    
    ax3.set_title("Severity Over Time", color=ColorScheme.TEXT_COLOR,
                  fontfamily='monospace', fontweight='bold')
    ax3.set_xlabel("Frame", color=ColorScheme.TEXT_COLOR, fontfamily='monospace')
    ax3.set_ylabel("Severity Score", color=ColorScheme.TEXT_COLOR, fontfamily='monospace')
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(['none', 'slight', 'moderate', 'severe'])
    ax3.tick_params(colors=ColorScheme.TEXT_COLOR)
    for spine in ax3.spines.values():
        spine.set_color(ColorScheme.TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=ColorScheme.BG_COLOR,
                bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print(f"Saved severity analysis to: {output_path}")


# =============================================================================
# Summary Report
# =============================================================================

def generate_summary_report(
    obj_info_dict: Dict[str, ObjectInfo],
    events: List[StateChangeEvent],
    frame_range: Tuple[int, int],
    output_path: str,
    fps: int = 30
):
    """Generate a text summary report of the analysis."""
    
    min_frame, max_frame = frame_range
    duration = (max_frame - min_frame) / fps
    
    lines = [
        "=" * 70,
        "OBJECT STATE TRANSFORMATION ANALYSIS REPORT",
        "=" * 70,
        "",
        f"Video Duration: {duration:.2f} seconds ({max_frame - min_frame + 1} frames)",
        f"Frame Rate: {fps} fps",
        f"Total Objects Tracked: {len(obj_info_dict)}",
        f"Total State Change Events: {len(events)}",
        "",
        "-" * 70,
        "OBJECT SUMMARY",
        "-" * 70,
    ]
    
    for obj_id, obj_info in obj_info_dict.items():
        lines.extend([
            "",
            f"Object {obj_id}:",
            f"  Description: {obj_info.description}",
            f"  Initial State: {obj_info.initial_state}",
            f"  Material: {obj_info.material}",
            f"  State Changes: {len(obj_info.state_changes)}",
        ])
    
    lines.extend([
        "",
        "-" * 70,
        "STATE CHANGE EVENTS (Chronological)",
        "-" * 70,
    ])
    
    sorted_events = sorted(events, key=lambda e: e.start_frame)
    for i, event in enumerate(sorted_events, 1):
        time_start = event.start_frame / fps
        time_end = event.end_frame / fps
        lines.extend([
            "",
            f"Event {i}:",
            f"  Time: {time_start:.2f}s - {time_end:.2f}s (frames {event.start_frame}-{event.end_frame})",
            f"  Object: {event.object_idx}",
            f"  Type: {event.change_type}",
            f"  Severity: {event.severity}",
            f"  Description: {event.description[:100]}{'...' if len(event.description) > 100 else ''}",
        ])
    
    # Statistics
    type_counts = defaultdict(int)
    severity_counts = defaultdict(int)
    for event in events:
        type_counts[event.change_type] += 1
        for key in ['none', 'slight', 'moderate', 'severe']:
            if key in event.severity.lower():
                severity_counts[key] += 1
                break
    
    lines.extend([
        "",
        "-" * 70,
        "STATISTICS",
        "-" * 70,
        "",
        "Change Type Distribution:",
    ])
    for ct, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {ct}: {count} ({100*count/len(events):.1f}%)")
    
    lines.extend([
        "",
        "Severity Distribution:",
    ])
    for sev in ['severe', 'moderate', 'slight', 'none']:
        count = severity_counts.get(sev, 0)
        lines.append(f"  {sev}: {count} ({100*count/len(events):.1f}%)")
    
    lines.extend([
        "",
        "=" * 70,
    ])
    
    report = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Saved summary report to: {output_path}")
    return report


# =============================================================================
# Main Entry Point
# =============================================================================

def visualize_state_graph(
    json_path: str,
    output_dir: str,
    fps: int = 30,
    title: str = None
):
    """
    Main function to generate all visualizations from prediction JSON.
    
    Args:
        json_path: Path to prediction JSON file
        output_dir: Output directory for visualizations
        fps: Frames per second
        title: Optional title for visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {json_path}")
    data = load_prediction_data(json_path)
    
    # Parse data
    obj_info_dict, events = parse_state_changes(data)
    frame_range = get_frame_range(data)
    
    print(f"Found {len(obj_info_dict)} objects")
    print(f"Found {len(events)} state change events")
    print(f"Frame range: {frame_range[0]} - {frame_range[1]}")
    
    if len(events) == 0:
        print("Warning: No state change events found!")
        return
    
    # Generate base name from input
    base_name = Path(json_path).stem
    if title is None:
        title = f"State Graph: {base_name}"
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. State graph
    create_state_graph(
        obj_info_dict, events, frame_range,
        osp.join(output_dir, f"{base_name}_state_graph.png"),
        title=title, fps=fps
    )
    
    # 2. Detailed timeline
    create_detailed_timeline(
        events, frame_range,
        osp.join(output_dir, f"{base_name}_timeline.png"),
        fps=fps
    )
    
    # 3. Severity analysis
    create_severity_analysis(
        events,
        osp.join(output_dir, f"{base_name}_analysis.png")
    )
    
    # 4. Summary report
    report = generate_summary_report(
        obj_info_dict, events, frame_range,
        osp.join(output_dir, f"{base_name}_report.txt"),
        fps=fps
    )
    
    print("\n" + "=" * 50)
    print("Visualization complete!")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - {base_name}_state_graph.png")
    print(f"  - {base_name}_timeline.png")
    print(f"  - {base_name}_analysis.png")
    print(f"  - {base_name}_report.txt")
    
    return {
        'obj_info': obj_info_dict,
        'events': events,
        'frame_range': frame_range,
        'report': report
    }


def get_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Visualize state transformation graphs from VLM-enhanced TubeletGraph output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python vis_state_graph.py --json_path 0000_1.json --output_dir ./vis_out
    python vis_state_graph.py --json_path results/pred.json --output_dir ./vis --fps 30
        """
    )
    parser.add_argument(
        "--json_path", "-j",
        required=True,
        help="Path to prediction JSON file"
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="./vis_output",
        help="Output directory for visualizations (default: ./vis_output)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--title", "-t",
        default=None,
        help="Title for visualizations"
    )
    
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    
    visualize_state_graph(
        json_path=args.json_path,
        output_dir=args.output_dir,
        fps=args.fps,
        title=args.title
    )
