#!/usr/bin/env python3
"""
TubeletGraph-Style Complete State Graph Visualization
======================================================

Creates complete state graph visualizations showing ALL state transitions:
- No consolidation - every state change is shown
- Horizontal layout with multi-row wrapping
- Object nodes with timestamps
- Diamond edges showing actions
- Interaction graphs showing object relationships

Usage:
    python vis_state_graph_full.py --json_files 0000_1.json 0000_2.json --output_dir ./vis_full
"""

import json
import os
import os.path as osp
import argparse
import re
import glob
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, RegularPolygon, Rectangle, FancyArrowPatch
from matplotlib.path import Path as MplPath
import matplotlib.patheffects as path_effects


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StateNode:
    """Represents an object state at a specific time."""
    obj_id: str
    timestamp: float
    frame: int
    description: str
    state: str
    node_type: str = "object"  # "initial", "object", "final", "interacting"
    color: str = "#E91E63"


@dataclass
class TransitionEdge:
    """Represents a transformation between states."""
    source_idx: int
    target_idx: int
    action: str
    timestamp: float
    frame: int
    severity: str = "slight"
    change_type: str = "deformation"


@dataclass
class StateGraph:
    """Complete state graph."""
    nodes: List[StateNode] = field(default_factory=list)
    edges: List[TransitionEdge] = field(default_factory=list)
    title: str = "State Graph"
    obj_id: str = "0"


# =============================================================================
# Color Schemes
# =============================================================================

OBJECT_COLORS = [
    "#E040FB",  # Magenta
    "#FF5252",  # Red
    "#FFAB40",  # Orange
    "#69F0AE",  # Green
    "#40C4FF",  # Cyan
    "#FFFF00",  # Yellow
    "#7C4DFF",  # Purple
    "#64FFDA",  # Teal
    "#FF4081",  # Pink
    "#536DFE",  # Indigo
]

# Gradient colors for sequential states (lighter to darker)
def get_gradient_colors(base_color: str, n: int) -> List[str]:
    """Generate gradient colors from light to dark."""
    base = base_color.lstrip('#')
    r, g, b = int(base[0:2], 16), int(base[2:4], 16), int(base[4:6], 16)
    
    colors = []
    for i in range(n):
        # Progress from lighter (0.3) to base (1.0)
        factor = 0.4 + 0.6 * (i / max(n - 1, 1))
        nr = int(255 - (255 - r) * factor)
        ng = int(255 - (255 - g) * factor)
        nb = int(255 - (255 - b) * factor)
        colors.append(f'#{nr:02x}{ng:02x}{nb:02x}')
    
    return colors


SEVERITY_COLORS = {
    'none': '#B0BEC5',
    'slight': '#81D4FA',
    'moderate': '#FFB74D',
    'severe': '#EF5350',
}

CHANGE_TYPE_COLORS = {
    'deformation': '#42A5F5',
    'surface_change': '#AB47BC',
    'material_release': '#66BB6A',
    'size_change': '#FFA726',
    'interaction': '#EC407A',
    'unknown': '#78909C',
}


# =============================================================================
# Helper Functions
# =============================================================================

def extract_object_id(filename: str) -> str:
    """Extract object ID from filename."""
    basename = Path(filename).stem
    match = re.search(r'_(\d+)$', basename)
    if match:
        return match.group(1)
    match = re.search(r'(\d+)$', basename)
    if match:
        return match.group(1)
    return "0"


def extract_action(description: str, change_type: str) -> str:
    """Extract concise action from description."""
    desc_lower = description.lower()
    
    actions = {
        'compress': 'compress',
        'squeeze': 'squeeze',
        'press': 'press',
        'flatten': 'flatten',
        'deform': 'deform',
        'bend': 'bend',
        'indent': 'indent',
        'puncture': 'puncture',
        'tear': 'tear',
        'crack': 'crack',
        'release': 'release',
        'leak': 'leak',
        'bulge': 'bulge',
        'stretch': 'stretch',
        'lift': 'lift',
        'contact': 'contact',
        'grip': 'grip',
        'move': 'move',
    }
    
    for keyword, action in actions.items():
        if keyword in desc_lower:
            return action
    
    # Fallback
    if change_type == 'surface_change':
        return 'surface\nchange'
    return change_type.replace('_', '\n')


def is_dark_color(hex_color: str) -> bool:
    """Check if color is dark."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.5


def get_severity_key(severity: str) -> str:
    """Normalize severity string."""
    sev_lower = severity.lower()
    for key in ['severe', 'moderate', 'slight', 'none']:
        if key in sev_lower:
            return key
    return 'slight'


# =============================================================================
# Graph Building - Full (No Consolidation)
# =============================================================================

def build_full_state_graph(json_path: str, fps: int = 30) -> StateGraph:
    """
    Build complete state graph showing ALL state transitions.
    No consolidation - every event creates a node.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    obj_id = extract_object_id(json_path)
    base_color = OBJECT_COLORS[int(obj_id) % len(OBJECT_COLORS)]
    
    # Get object info
    obj_info = {}
    if 'obj_info' in data:
        for k, v in data['obj_info'].items():
            obj_info = v
            break
    
    obj_desc = obj_info.get('desc', f'Object {obj_id}')
    initial_state = obj_info.get('initial_state', 'initial')
    
    # Get all events
    events = data.get('state_change_events', [])
    events = sorted(events, key=lambda x: x.get('start_frame', 0))
    
    # Generate gradient colors for nodes
    n_nodes = len(events) + 1
    node_colors = get_gradient_colors(base_color, n_nodes)
    
    graph = StateGraph(
        title=f"State Graph: {obj_desc}",
        obj_id=obj_id
    )
    
    # Initial node
    graph.nodes.append(StateNode(
        obj_id=obj_id,
        timestamp=0,
        frame=0,
        description=obj_desc,
        state=initial_state,
        node_type="initial",
        color=node_colors[0]
    ))
    
    prev_idx = 0
    
    for i, event in enumerate(events):
        start_frame = event.get('start_frame', 0)
        end_frame = event.get('end_frame', start_frame)
        change_type = event.get('change_type', 'unknown')
        description = event.get('description', '')
        severity = event.get('severity', 'slight')
        
        timestamp = end_frame / fps
        action = extract_action(description, change_type)
        
        # Determine node type
        node_type = "final" if i == len(events) - 1 else "object"
        
        # State node
        graph.nodes.append(StateNode(
            obj_id=obj_id,
            timestamp=timestamp,
            frame=end_frame,
            description=obj_desc,
            state=f"{action}ed",
            node_type=node_type,
            color=node_colors[i + 1]
        ))
        
        # Edge
        graph.edges.append(TransitionEdge(
            source_idx=prev_idx,
            target_idx=len(graph.nodes) - 1,
            action=action,
            timestamp=start_frame / fps,
            frame=start_frame,
            severity=severity,
            change_type=change_type
        ))
        
        prev_idx = len(graph.nodes) - 1
    
    return graph


def build_full_interaction_graph(json_files: List[str], fps: int = 30) -> StateGraph:
    """
    Build complete interaction graph showing primary object states
    with interacting objects connected at relevant points.
    """
    sorted_files = sorted(json_files, key=lambda f: extract_object_id(f))
    
    # Load all data
    all_data = {}
    for path in sorted_files:
        oid = extract_object_id(path)
        with open(path, 'r') as f:
            all_data[oid] = json.load(f)
    
    if not all_data:
        return StateGraph()
    
    # Primary object
    primary_id = list(all_data.keys())[0]
    primary_data = all_data[primary_id]
    primary_color = OBJECT_COLORS[0]
    
    primary_info = {}
    if 'obj_info' in primary_data:
        for k, v in primary_data['obj_info'].items():
            primary_info = v
            break
    
    primary_desc = primary_info.get('desc', f'Object {primary_id}')
    
    # Get primary events
    primary_events = primary_data.get('state_change_events', [])
    primary_events = sorted(primary_events, key=lambda x: x.get('start_frame', 0))
    
    # Gradient colors
    n_primary = len(primary_events) + 1
    primary_colors = get_gradient_colors(primary_color, n_primary)
    
    graph = StateGraph(
        title=f"Interaction Graph: {primary_desc}",
        obj_id=primary_id
    )
    
    # Initial primary node
    graph.nodes.append(StateNode(
        obj_id=primary_id,
        timestamp=0,
        frame=0,
        description=primary_desc,
        state=primary_info.get('initial_state', 'initial'),
        node_type="initial",
        color=primary_colors[0]
    ))
    
    prev_primary_idx = 0
    
    for i, event in enumerate(primary_events):
        start_frame = event.get('start_frame', 0)
        end_frame = event.get('end_frame', start_frame)
        change_type = event.get('change_type', 'unknown')
        description = event.get('description', '')
        severity = event.get('severity', 'slight')
        
        timestamp = end_frame / fps
        action = extract_action(description, change_type)
        
        # Find concurrent interaction
        interacting_id = None
        interacting_desc = None
        
        for other_id, other_data in all_data.items():
            if other_id == primary_id:
                continue
            
            other_events = other_data.get('state_change_events', [])
            for oe in other_events:
                o_start = oe.get('start_frame', 0)
                o_end = oe.get('end_frame', 0)
                
                if o_start <= end_frame and o_end >= start_frame:
                    interacting_id = other_id
                    if 'obj_info' in other_data:
                        for k, v in other_data['obj_info'].items():
                            interacting_desc = v.get('desc', f'Object {other_id}')
                            break
                    break
            if interacting_id:
                break
        
        # Primary node
        node_type = "final" if i == len(primary_events) - 1 else "object"
        
        graph.nodes.append(StateNode(
            obj_id=primary_id,
            timestamp=timestamp,
            frame=end_frame,
            description=primary_desc,
            state=f"{action}ed",
            node_type=node_type,
            color=primary_colors[i + 1]
        ))
        primary_node_idx = len(graph.nodes) - 1
        
        # Edge to primary
        graph.edges.append(TransitionEdge(
            source_idx=prev_primary_idx,
            target_idx=primary_node_idx,
            action=action,
            timestamp=start_frame / fps,
            frame=start_frame,
            severity=severity,
            change_type=change_type
        ))
        
        # Add interacting node
        if interacting_id and interacting_desc:
            inter_color = OBJECT_COLORS[int(interacting_id) % len(OBJECT_COLORS)]
            
            graph.nodes.append(StateNode(
                obj_id=interacting_id,
                timestamp=timestamp,
                frame=end_frame,
                description=interacting_desc,
                state="interacting",
                node_type="interacting",
                color=inter_color
            ))
            
            graph.edges.append(TransitionEdge(
                source_idx=len(graph.nodes) - 1,
                target_idx=primary_node_idx,
                action="",
                timestamp=start_frame / fps,
                frame=start_frame,
                severity=severity,
                change_type="interaction"
            ))
        
        prev_primary_idx = primary_node_idx
    
    return graph


# =============================================================================
# Drawing Functions
# =============================================================================

def draw_state_node(ax, x, y, width, height, color, timestamp, label,
                    is_initial=False, is_final=False, fontsize=8):
    """Draw a state node with timestamp bubble."""
    
    # Main rounded box
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color,
        edgecolor='#333333',
        linewidth=2.5 if is_initial else (2.0 if is_final else 1.5),
        zorder=5
    )
    ax.add_patch(box)
    
    # Timestamp bubble on top
    bubble_w, bubble_h = 0.48, 0.20
    bubble = FancyBboxPatch(
        (x - bubble_w/2, y + height/2 - bubble_h/2 - 0.02), 
        bubble_w, bubble_h,
        boxstyle="round,pad=0.01,rounding_size=0.06",
        facecolor='white',
        edgecolor=color,
        linewidth=1.5,
        zorder=6
    )
    ax.add_patch(bubble)
    
    # Timestamp text
    ax.text(x, y + height/2 - 0.02, timestamp,
            fontsize=fontsize - 1, ha='center', va='center',
            color=color, fontweight='bold', fontfamily='monospace',
            zorder=7)
    
    # Label text
    text_color = 'white' if is_dark_color(color) else 'black'
    ax.text(x, y - 0.02, label,
            fontsize=fontsize, ha='center', va='center',
            color=text_color, fontweight='bold',
            zorder=7)


def draw_action_diamond(ax, x, y, size, color, text, fontsize=6):
    """Draw action diamond."""
    diamond = RegularPolygon(
        (x, y), numVertices=4, radius=size,
        orientation=np.pi/4,
        facecolor='white',
        edgecolor=color,
        linewidth=2,
        zorder=8
    )
    ax.add_patch(diamond)
    
    # Text
    ax.text(x, y, text, fontsize=fontsize, ha='center', va='center',
            color='#333333', fontweight='bold',
            zorder=9)


def draw_connection(ax, x1, y1, x2, y2, color='#555555', style='-', lw=2):
    """Draw connection line with arrow."""
    ax.annotate('',
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               linestyle=style, shrinkA=0, shrinkB=0),
                zorder=2)


# =============================================================================
# Main Visualization - Horizontal Layout
# =============================================================================

def visualize_full_horizontal(
    graph: StateGraph,
    output_path: str,
    nodes_per_row: int = 8,
    node_width: float = 1.4,
    node_height: float = 0.6,
    h_gap: float = 1.9,
    v_gap: float = 1.6,
    diamond_size: float = 0.25
):
    """
    Visualize complete state graph with horizontal layout.
    Wraps to multiple rows if needed.
    """
    if not graph.nodes:
        print("Empty graph")
        return
    
    # Separate primary and interacting nodes
    primary_nodes = [(i, n) for i, n in enumerate(graph.nodes) if n.node_type != "interacting"]
    inter_nodes = [(i, n) for i, n in enumerate(graph.nodes) if n.node_type == "interacting"]
    
    n_primary = len(primary_nodes)
    n_rows = (n_primary - 1) // nodes_per_row + 1
    has_inter = len(inter_nodes) > 0
    
    # Figure size
    cols = min(n_primary, nodes_per_row)
    fig_width = max(16, cols * h_gap + 2)
    fig_height = max(4, n_rows * v_gap * (2 if has_inter else 1) + 2)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
    ax.set_facecolor('white')
    
    # Position primary nodes
    positions = {}
    y_top = fig_height - 1.5
    
    for i, (idx, node) in enumerate(primary_nodes):
        row = i // nodes_per_row
        col = i % nodes_per_row
        
        x = 1.2 + col * h_gap
        y = y_top - row * v_gap * (2 if has_inter else 1)
        
        positions[idx] = (x, y, 'primary')
    
    # Position interacting nodes below their corresponding primary
    for idx, node in inter_nodes:
        # Find connected primary node
        target_pos = None
        for edge in graph.edges:
            if edge.source_idx == idx and edge.target_idx in positions:
                target_pos = positions[edge.target_idx]
                break
        
        if target_pos:
            positions[idx] = (target_pos[0], target_pos[1] - v_gap * 0.8, 'interacting')
    
    # Draw edges
    for edge in graph.edges:
        src_pos = positions.get(edge.source_idx)
        tgt_pos = positions.get(edge.target_idx)
        
        if not src_pos or not tgt_pos:
            continue
        
        src_type = src_pos[2]
        severity_key = get_severity_key(edge.severity)
        diamond_color = SEVERITY_COLORS.get(severity_key, '#81D4FA')
        line_color = '#555555'
        
        if src_type == 'interacting':
            # Vertical dashed line from interacting to primary
            edge_color = CHANGE_TYPE_COLORS.get('interaction', '#EC407A')
            
            ax.plot([src_pos[0], tgt_pos[0]], 
                   [src_pos[1] + node_height/2, tgt_pos[1] - node_height/2 - 0.05],
                   color=edge_color, lw=2, linestyle='--', zorder=1)
            
            draw_connection(ax, tgt_pos[0], tgt_pos[1] - node_height/2 - 0.05,
                           tgt_pos[0], tgt_pos[1] - node_height/2,
                           color=edge_color, lw=2)
        else:
            # Horizontal connection between primary nodes
            src_x, src_y = src_pos[0], src_pos[1]
            tgt_x, tgt_y = tgt_pos[0], tgt_pos[1]
            
            if abs(src_y - tgt_y) < 0.1:  # Same row
                mid_x = (src_x + tgt_x) / 2
                
                # Line from source to diamond
                ax.plot([src_x + node_width/2, mid_x - diamond_size * 1.1],
                       [src_y, src_y], color=line_color, lw=2, zorder=1)
                
                # Line from diamond to target
                ax.plot([mid_x + diamond_size * 1.1, tgt_x - node_width/2 - 0.08],
                       [tgt_y, tgt_y], color=line_color, lw=2, zorder=1)
                
                # Arrow
                draw_connection(ax, tgt_x - node_width/2 - 0.08, tgt_y,
                               tgt_x - node_width/2, tgt_y, line_color, lw=2)
                
                # Diamond
                draw_action_diamond(ax, mid_x, src_y, diamond_size, diamond_color, edge.action)
                
            else:  # Different rows - wrap around
                # Right from source
                ax.plot([src_x + node_width/2, src_x + h_gap * 0.6],
                       [src_y, src_y], color=line_color, lw=2, zorder=1)
                
                # Down
                ax.plot([src_x + h_gap * 0.6, src_x + h_gap * 0.6],
                       [src_y, tgt_y], color=line_color, lw=2, zorder=1)
                
                # Left to target
                ax.plot([src_x + h_gap * 0.6, tgt_x - node_width/2 - 0.08],
                       [tgt_y, tgt_y], color=line_color, lw=2, zorder=1)
                
                # Arrow
                draw_connection(ax, tgt_x - node_width/2 - 0.08, tgt_y,
                               tgt_x - node_width/2, tgt_y, line_color, lw=2)
                
                # Diamond at corner
                draw_action_diamond(ax, src_x + h_gap * 0.6, (src_y + tgt_y) / 2,
                                   diamond_size, diamond_color, edge.action)
    
    # Draw nodes
    for idx, (x, y, ntype) in positions.items():
        node = graph.nodes[idx]
        
        ts_str = f"t={node.timestamp:.1f}s"
        if node.timestamp == int(node.timestamp):
            ts_str = f"t={int(node.timestamp)}s"
        
        # Shorter label
        label = node.description
        if len(label) > 16:
            label = label[:14] + "..."
        
        draw_state_node(
            ax, x, y, node_width, node_height, node.color,
            ts_str, label,
            is_initial=(node.node_type == "initial"),
            is_final=(node.node_type == "final")
        )
    
    # Axis limits
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    
    ax.set_xlim(min(all_x) - 1.2, max(all_x) + 1.2)
    ax.set_ylim(min(all_y) - 1.0, max(all_y) + 1.0)
    
    # Title
    ax.set_title(graph.title, fontsize=14, fontweight='bold', pad=15)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', pad_inches=0.2)
    plt.close()
    
    print(f"Saved: {output_path}")


def visualize_single_row(
    graph: StateGraph,
    output_path: str,
    node_width: float = 1.2,
    node_height: float = 0.55,
    h_gap: float = 1.7,
    diamond_size: float = 0.22
):
    """
    Visualize state graph in a single horizontal row (for shorter sequences).
    Good for up to ~10-12 nodes.
    """
    if not graph.nodes:
        print("Empty graph")
        return
    
    primary_nodes = [(i, n) for i, n in enumerate(graph.nodes) if n.node_type != "interacting"]
    inter_nodes = [(i, n) for i, n in enumerate(graph.nodes) if n.node_type == "interacting"]
    
    n_primary = len(primary_nodes)
    has_inter = len(inter_nodes) > 0
    
    fig_width = max(16, n_primary * h_gap + 2)
    fig_height = 4.5 if has_inter else 3.0
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
    ax.set_facecolor('white')
    
    # Position all primary nodes in one row
    positions = {}
    y_primary = 2.0 if has_inter else 1.2
    
    for i, (idx, node) in enumerate(primary_nodes):
        x = 1.0 + i * h_gap
        positions[idx] = (x, y_primary, 'primary')
    
    # Interacting nodes below
    y_inter = 0.7
    for idx, node in inter_nodes:
        target_pos = None
        for edge in graph.edges:
            if edge.source_idx == idx and edge.target_idx in positions:
                target_pos = positions[edge.target_idx]
                break
        
        if target_pos:
            positions[idx] = (target_pos[0], y_inter, 'interacting')
    
    # Draw edges
    for edge in graph.edges:
        src_pos = positions.get(edge.source_idx)
        tgt_pos = positions.get(edge.target_idx)
        
        if not src_pos or not tgt_pos:
            continue
        
        src_type = src_pos[2]
        severity_key = get_severity_key(edge.severity)
        diamond_color = SEVERITY_COLORS.get(severity_key, '#81D4FA')
        line_color = '#555555'
        
        if src_type == 'interacting':
            edge_color = CHANGE_TYPE_COLORS['interaction']
            ax.plot([src_pos[0], tgt_pos[0]],
                   [src_pos[1] + node_height/2, tgt_pos[1] - node_height/2 - 0.03],
                   color=edge_color, lw=2, linestyle='--', zorder=1)
            draw_connection(ax, tgt_pos[0], tgt_pos[1] - node_height/2 - 0.03,
                           tgt_pos[0], tgt_pos[1] - node_height/2,
                           edge_color, lw=2)
        else:
            mid_x = (src_pos[0] + tgt_pos[0]) / 2
            mid_y = src_pos[1]
            
            ax.plot([src_pos[0] + node_width/2, mid_x - diamond_size * 1.1],
                   [mid_y, mid_y], color=line_color, lw=2, zorder=1)
            ax.plot([mid_x + diamond_size * 1.1, tgt_pos[0] - node_width/2 - 0.06],
                   [mid_y, mid_y], color=line_color, lw=2, zorder=1)
            draw_connection(ax, tgt_pos[0] - node_width/2 - 0.06, mid_y,
                           tgt_pos[0] - node_width/2, mid_y, line_color, lw=2)
            draw_action_diamond(ax, mid_x, mid_y, diamond_size, diamond_color, edge.action)
    
    # Draw nodes
    for idx, (x, y, ntype) in positions.items():
        node = graph.nodes[idx]
        
        ts_str = f"t={node.timestamp:.1f}s"
        if node.timestamp == int(node.timestamp):
            ts_str = f"t={int(node.timestamp)}s"
        
        label = node.description[:14] + "..." if len(node.description) > 14 else node.description
        
        draw_state_node(ax, x, y, node_width, node_height, node.color,
                       ts_str, label,
                       is_initial=(node.node_type == "initial"),
                       is_final=(node.node_type == "final"),
                       fontsize=7)
    
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    
    ax.set_xlim(min(all_x) - 1.0, max(all_x) + 1.0)
    ax.set_ylim(min(all_y) - 0.8, max(all_y) + 0.8)
    
    ax.set_title(graph.title, fontsize=13, fontweight='bold', pad=12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', pad_inches=0.15)
    plt.close()
    
    print(f"Saved: {output_path}")


# =============================================================================
# Main Functions
# =============================================================================

def generate_full_state_graphs(
    json_files: List[str],
    output_dir: str,
    fps: int = 30,
    nodes_per_row: int = 8
):
    """Generate complete state graphs for all objects."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Generating COMPLETE State Graphs (All States)")
    print("=" * 70)
    
    # Individual graphs
    for json_path in sorted(json_files):
        obj_id = extract_object_id(json_path)
        print(f"\nObject {obj_id}: {json_path}")
        
        graph = build_full_state_graph(json_path, fps)
        print(f"  Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")
        
        out_path = osp.join(output_dir, f"full_state_graph_obj{obj_id}.png")
        
        # Choose layout based on node count
        if len(graph.nodes) <= 12:
            visualize_single_row(graph, out_path)
        else:
            visualize_full_horizontal(graph, out_path, nodes_per_row=nodes_per_row)
    
    # Combined interaction graph
    if len(json_files) > 1:
        print(f"\nGenerating combined interaction graph...")
        combined = build_full_interaction_graph(json_files, fps)
        print(f"  Nodes: {len(combined.nodes)}, Edges: {len(combined.edges)}")
        
        out_path = osp.join(output_dir, "full_interaction_graph.png")
        
        primary_count = sum(1 for n in combined.nodes if n.node_type != "interacting")
        if primary_count <= 10:
            visualize_single_row(combined, out_path)
        else:
            visualize_full_horizontal(combined, out_path, nodes_per_row=nodes_per_row)
    
    print("\n" + "=" * 70)
    print(f"Complete! Output: {output_dir}")
    print("=" * 70)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate complete TubeletGraph-style state graphs (no consolidation)"
    )
    parser.add_argument("--json_files", "-j", nargs='+', help="JSON files")
    parser.add_argument("--json_pattern", "-p", help="Glob pattern")
    parser.add_argument("--output_dir", "-o", default="./vis_full_graphs")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--nodes_per_row", "-n", type=int, default=8,
                       help="Max nodes per row for wrapping")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    
    json_files = []
    if args.json_files:
        json_files.extend(args.json_files)
    if args.json_pattern:
        json_files.extend(glob.glob(args.json_pattern))
    
    if not json_files:
        print("No JSON files specified")
        exit(1)
    
    json_files = list(set(json_files))
    print(f"Processing {len(json_files)} files")
    
    generate_full_state_graphs(
        json_files, 
        args.output_dir, 
        args.fps,
        args.nodes_per_row
    )
