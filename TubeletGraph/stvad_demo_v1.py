#!/usr/bin/env python3
"""
ST-VAD Framework: Spatio-Temporal Video Anomaly Detection
A multi-stage pipeline for detecting anomalies in videos using object grounding,
tracking, and VLM-based reasoning.

Pipeline Stages:
1. Object Grounding: Identify and segment objects in the video using VLM
2. Object Tracking: Track objects across frames using TubeletGraph
3. State Change Analysis: Analyze object state changes and temporal patterns
4. Anomaly Detection: Detect and explain anomalies using step-by-step VLM reasoning

Usage:
    python stvad_demo.py analyze <video_path> -c <config_path> [options]
    python stvad_demo.py batch <video_dir> -c <config_path> [options]

File Structure:
    This framework expects the following structure:
    TubeletGraph/
    ├── stvad_demo.py (this file)
    ├── quick_run.py
    ├── annotate/
    │   └── vlm_mask_grounded.py
    └── TubeletGraph/
        └── vlm/
            ├── prompt_vlm.py      # Stage 2: State tracking
            └── prompt_vad.py      # Stage 3&4: Anomaly detection
"""

import os
import os.path as osp
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil


class STVADFramework:
    """Main framework for spatio-temporal video anomaly detection."""
    
    def __init__(self, config_path: str, output_dir: str = "output", verbose: bool = True):
        """
        Initialize ST-VAD Framework.
        
        Args:
            config_path: Path to TubeletGraph config file
            output_dir: Directory for intermediate outputs
            verbose: Enable verbose logging
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Get the base directory (where this script is located)
        self.base_dir = osp.dirname(osp.abspath(__file__))
        
        # Define script paths relative to base directory
        self.vlm_mask_script = osp.join(self.base_dir, "annotate", "vlm_mask_grounded.py")
        self.quick_run_script = osp.join(self.base_dir, "quick_run.py")
        # Stage 3&4: Use prompt_vad.py for anomaly detection with chain-of-thought reasoning
        self.prompt_vad_script = osp.join(self.base_dir, "TubeletGraph", "vlm", "prompt_vad.py")
        # Stage 2 (optional): Use prompt_vlm.py for state tracking only
        self.prompt_vlm_script = osp.join(self.base_dir, "TubeletGraph", "vlm", "prompt_vlm.py")
        
        # Verify scripts exist
        self.verify_scripts()
        
        # Create output directory structure
        self.setup_directories()
        
    def verify_scripts(self):
        """Verify that all required scripts exist."""
        scripts = {
            "vlm_mask_grounded.py": self.vlm_mask_script,
            "quick_run.py": self.quick_run_script,
            "prompt_vad.py": self.prompt_vad_script
        }
        
        missing = []
        for name, path in scripts.items():
            if not osp.isfile(path):
                missing.append(f"{name} (expected at: {path})")
        
        if missing:
            print(f"❌ Error: Missing required scripts:")
            for m in missing:
                print(f"   - {m}")
            print(f"\nCurrent base directory: {self.base_dir}")
            print(f"\nExpected structure:")
            print(f"  {self.base_dir}/")
            print(f"  ├── annotate/vlm_mask_grounded.py")
            print(f"  ├── quick_run.py")
            print(f"  └── TubeletGraph/vlm/prompt_vad.py")
            sys.exit(1)
    
    def setup_directories(self):
        """Create necessary output directories."""
        dirs = [
            self.output_dir,
            osp.join(self.output_dir, "JPEGImages"),
            osp.join(self.output_dir, "Annotations"),
            osp.join(self.output_dir, "tracking_results"),
            osp.join(self.output_dir, "anomaly_reports"),
            osp.join(self.output_dir, "visualizations")
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with formatting."""
        if self.verbose:
            symbols = {
                "INFO": "ℹ️",
                "SUCCESS": "✅",
                "ERROR": "❌",
                "WARNING": "⚠️",
                "STAGE": "🔄"
            }
            symbol = symbols.get(level, "•")
            print(f"{symbol} {message}")
    
    def run_command(self, cmd: List[str], description: str, check: bool = True) -> Tuple[bool, str]:
        """
        Run a shell command and capture output.
        
        Args:
            cmd: Command to run as list of strings
            description: Description for logging
            check: Whether to exit on failure
            
        Returns:
            Tuple of (success, output)
        """
        self.log(f"Running: {description}", "STAGE")
        if self.verbose:
            print(f"  Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                self.log(f"Failed: {description}", "ERROR")
                if self.verbose:
                    print(f"  Error output: {result.stderr}")
                if check:
                    sys.exit(1)
                return False, result.stderr
            
            self.log(f"Completed: {description}", "SUCCESS")
            return True, result.stdout
            
        except Exception as e:
            self.log(f"Exception in {description}: {str(e)}", "ERROR")
            if check:
                sys.exit(1)
            return False, str(e)
    
    def stage1_object_grounding(
        self,
        video_path: str,
        vlm_model: str = "gpt4v",
        target_fps: int = 10,
        auto_mode: bool = True
    ) -> Tuple[str, str, bool]:
        """
        Stage 1: Object Grounding and Segmentation
        
        Uses VLM to identify and segment objects in the video.
        
        Args:
            video_path: Path to input video
            vlm_model: VLM model to use (gpt4v, claude, etc.)
            target_fps: Target FPS for frame extraction
            auto_mode: Use automatic object detection
            
        Returns:
            Tuple of (frames_dir, mask_path, success)
        """
        self.log("="*60, "INFO")
        self.log("STAGE 1: Object Grounding", "STAGE")
        self.log("="*60, "INFO")
        
        video_name = Path(video_path).stem
        
        cmd = [
            "python", self.vlm_mask_script,
            "-i", video_path,
            "--vlm", vlm_model,
            "--frame", str(target_fps),
            "--output_dir", self.output_dir
        ]
        
        if auto_mode:
            cmd.append("--auto")
        
        success, output = self.run_command(cmd, "Object grounding with VLM")
        
        frames_dir = osp.join(self.output_dir, "JPEGImages", video_name)
        
        # Try multiple possible mask paths
        possible_mask_paths = [
            osp.join(self.output_dir, "Annotations", video_name, f"{video_name}000.png"),
            osp.join(self.output_dir, "Annotations", video_name, "0000000.png"),
            osp.join(self.output_dir, "Annotations", video_name, f"{video_name}_0000000.png"),
        ]
        
        mask_path = None
        for path in possible_mask_paths:
            if osp.isfile(path):
                mask_path = path
                break
        
        # Verify outputs exist
        if not osp.isdir(frames_dir):
            self.log(f"Frames directory not found: {frames_dir}", "ERROR")
            return frames_dir, mask_path or possible_mask_paths[0], False
        
        if mask_path is None:
            self.log(f"Mask file not found. Tried:", "ERROR")
            for path in possible_mask_paths:
                self.log(f"  - {path}", "ERROR")
            return frames_dir, possible_mask_paths[0], False
        
        self.log(f"Frames saved to: {frames_dir}", "SUCCESS")
        self.log(f"Mask saved to: {mask_path}", "SUCCESS")
        
        return frames_dir, mask_path, success
    
    def stage2_object_tracking(
        self,
        frames_dir: str,
        mask_path: str,
        fps: int = 10,
        method: str = "Ours"
    ) -> Tuple[str, str, bool]:
        """
        Stage 2: Object-Centric Tracking
        
        Tracks objects across frames using TubeletGraph pipeline.
        
        Args:
            frames_dir: Directory containing video frames
            mask_path: Path to object mask annotation
            fps: Frames per second
            method: Tracking method name
            
        Returns:
            Tuple of (prediction_name, video_name, success)
        """
        self.log("="*60, "INFO")
        self.log("STAGE 2: Object-Centric Tracking", "STAGE")
        self.log("="*60, "INFO")
        
        cmd = [
            "python", self.quick_run_script,
            "-c", self.config_path,
            "--input_dir", frames_dir,
            "--input_mask", mask_path,
            "--fps", str(fps),
            "--method", method
        ]
        
        success, output = self.run_command(cmd, "TubeletGraph tracking pipeline")
        
        # Extract prediction name from output or construct it
        video_name = Path(frames_dir).name
        # Prediction name format: custom-{video_name}-{method}_{vlm_model}
        # We'll need to parse config to get vlm model name, for now use generic
        prediction_name = f"custom-{video_name}-{method}"
        
        self.log(f"Tracking results: {prediction_name}", "SUCCESS")
        
        return prediction_name, video_name, success
    
    def stage3_state_analysis(
        self,
        video_path: str,
        prediction_name: str,
        sample_interval: int = 10,
        detect_anomalies: bool = True,
        vlm_model: str = "openai"
    ) -> Tuple[Dict, bool]:
        """
        Stage 3 & 4: State Change Analysis and Anomaly Detection
        
        Uses VLM with chain-of-thought reasoning to analyze object states 
        and detect anomalies through a 6-step reasoning process:
        1. Observation: What changes occurred?
        2. Expectation: What should have happened?
        3. Comparison: How do they differ?
        4. Causation: What caused the deviation?
        5. Classification: What type of anomaly?
        6. Severity: How serious is it?
        
        Args:
            video_path: Path to original video
            prediction_name: Name of tracking predictions
            sample_interval: Interval for sampling frames
            detect_anomalies: Enable anomaly detection
            vlm_model: VLM provider (openai, claude, ollama)
            
        Returns:
            Tuple of (anomaly_report, success)
        """
        self.log("="*60, "INFO")
        self.log("STAGE 3 & 4: State Analysis and Anomaly Detection", "STAGE")
        self.log("="*60, "INFO")
        
        # Build command for prompt_vad.py (the anomaly detection script)
        cmd = [
            "python", self.prompt_vad_script,
            "-c", self.config_path,
            "-p", prediction_name,
            "--sample_interval", str(sample_interval),
            "--video_path", video_path,
            "--vlm", vlm_model,
            "--output_dir", osp.join(self.output_dir, "anomaly_reports")
        ]
        
        if detect_anomalies:
            cmd.append("--detect_anomalies")
        
        if self.verbose:
            cmd.append("-v")
        
        success, output = self.run_command(cmd, "VLM-based state analysis and anomaly detection")
        
        # Parse anomaly report from output
        report = self.parse_anomaly_report(output, prediction_name)
        
        # Save report
        video_name = Path(video_path).stem
        report_path = osp.join(self.output_dir, "anomaly_reports", f"{video_name}_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log(f"Anomaly report saved to: {report_path}", "SUCCESS")
        
        return report, success
    
    def parse_anomaly_report(self, output: str, prediction_name: str) -> Dict:
        """
        Parse anomaly detection output into structured report.
        
        Args:
            output: Raw output from VLM analysis
            prediction_name: Name of the prediction
            
        Returns:
            Structured anomaly report dictionary
        """
        report = {
            "prediction_name": prediction_name,
            "anomaly_detected": False,
            "anomalies": [],
            "reasoning_trace": "",
            "identified_events": [],
            "anomalous_transitions": [],
            "num_anomalies": 0
        }
        
        # Parse output for key information
        if "ANOMALY DETECTION REPORT" in output:
            lines = output.split('\n')
            for i, line in enumerate(lines):
                # Parse anomaly detection status
                if "Anomaly Detected:" in line:
                    report["anomaly_detected"] = "True" in line
                
                # Parse number of anomalies
                elif "Number of Anomalies:" in line:
                    import re
                    match = re.search(r'(\d+)', line)
                    if match:
                        report["num_anomalies"] = int(match.group(1))
                
                # Parse overall severity
                elif "Overall Severity:" in line:
                    severity = line.split(":")[-1].strip()
                    report["overall_severity"] = severity
                
                # Capture reasoning trace
                elif "Reasoning Trace:" in line:
                    trace_lines = []
                    in_trace = True
                    for j in range(i+1, len(lines)):
                        if lines[j].startswith("=" * 10):
                            # Check if this is the end of reasoning section
                            if j + 1 < len(lines) and "Step 1:" in lines[j+1]:
                                break
                            in_trace = False
                            break
                        if in_trace and lines[j].strip():
                            trace_lines.append(lines[j].strip())
                    report["reasoning_trace"] = "\n".join(trace_lines)
                
                # Parse identified events (Step 1)
                elif "Step 1: Identified events" in line or "Step 1:" in line:
                    for j in range(i+1, len(lines)):
                        if lines[j].strip() and not lines[j].startswith("="):
                            if "Step 2" in lines[j]:
                                break
                            # Parse numbered events
                            event_match = re.match(r'\s*\d+\.\s*(.+)', lines[j])
                            if event_match:
                                report["identified_events"].append(event_match.group(1))
                        elif lines[j].startswith("="):
                            break
                
                # Parse anomalous transitions (Step 2)
                elif "anomalous transitions" in line.lower():
                    import re
                    match = re.search(r'(\d+)\s+anomalous', line)
                    if match:
                        num_anomalies = int(match.group(1))
                        if report["num_anomalies"] == 0:
                            report["num_anomalies"] = num_anomalies
                
                # Parse individual anomalies
                elif line.strip().startswith("[anomaly_"):
                    anomaly = {"id": line.strip().strip("[]")}
                    for j in range(i+1, min(i+8, len(lines))):
                        aline = lines[j].strip()
                        if aline.startswith("Type:"):
                            anomaly["type"] = aline.split(":", 1)[1].strip()
                        elif aline.startswith("Severity:"):
                            anomaly["severity"] = aline.split(":", 1)[1].strip()
                        elif aline.startswith("Description:"):
                            anomaly["description"] = aline.split(":", 1)[1].strip()
                        elif aline.startswith("Confidence:"):
                            anomaly["confidence"] = aline.split(":", 1)[1].strip()
                        elif aline.startswith("[anomaly_") or aline.startswith("="):
                            break
                    if "type" in anomaly:
                        report["anomalies"].append(anomaly)
                
                # Parse summary
                elif "Summary:" in line:
                    summary_lines = []
                    for j in range(i+1, len(lines)):
                        if lines[j].startswith("="):
                            break
                        if lines[j].strip():
                            summary_lines.append(lines[j].strip())
                    report["summary"] = " ".join(summary_lines)
        
        return report
    
    def generate_summary_report(self, report: Dict, video_path: str) -> str:
        """
        Generate human-readable summary report.
        
        Args:
            report: Anomaly report dictionary
            video_path: Path to video
            
        Returns:
            Formatted summary string
        """
        video_name = Path(video_path).stem
        
        summary = f"""
{'='*80}
ST-VAD ANOMALY DETECTION REPORT
{'='*80}
Video: {video_name}
Path: {video_path}
{'='*80}

DETECTION RESULT:
  Anomaly Detected: {'YES ❌' if report['anomaly_detected'] else 'NO ✅'}
  Number of Anomalies: {report.get('num_anomalies', 0)}
  Overall Severity: {report.get('overall_severity', 'N/A')}

{'='*80}
REASONING TRACE:
{'='*80}
{report.get('reasoning_trace', 'No reasoning trace available')}

{'='*80}
IDENTIFIED EVENTS:
{'='*80}
"""
        if report.get('identified_events'):
            for i, event in enumerate(report['identified_events'], 1):
                summary += f"{i}. {event}\n"
        else:
            summary += "No events identified\n"
        
        if report.get('anomalies'):
            summary += f"\n{'='*80}\nDETECTED ANOMALIES:\n{'='*80}\n"
            for anomaly in report['anomalies']:
                summary += f"\n  [{anomaly.get('id', 'unknown')}]\n"
                summary += f"    Type: {anomaly.get('type', 'unknown')}\n"
                summary += f"    Severity: {anomaly.get('severity', 'unknown')}\n"
                summary += f"    Description: {anomaly.get('description', 'N/A')}\n"
        
        summary += f"\n{'='*80}\n"
        
        if report.get('summary'):
            summary += f"\nSUMMARY: {report['summary']}\n"
        
        return summary
    
    def analyze_video(
        self,
        video_path: str,
        vlm_model: str = "openai",
        target_fps: int = 10,
        sample_interval: int = 10,
        method: str = "Ours",
        auto_mode: bool = True
    ) -> Dict:
        """
        Run complete ST-VAD pipeline on a single video.
        
        Args:
            video_path: Path to video file
            vlm_model: VLM model for object grounding
            target_fps: Target FPS for processing
            sample_interval: Sampling interval for analysis
            method: Tracking method
            auto_mode: Use automatic object detection
            
        Returns:
            Complete analysis report
        """
        self.log("="*80, "INFO")
        self.log(f"ST-VAD Framework: Analyzing {video_path}", "INFO")
        self.log("="*80, "INFO")
        
        video_name = Path(video_path).stem
        
        # Stage 1: Object Grounding
        frames_dir, mask_path, stage1_success = self.stage1_object_grounding(
            video_path, vlm_model, target_fps, auto_mode
        )
        
        if not stage1_success:
            self.log("Stage 1 failed, aborting pipeline", "ERROR")
            return {"success": False, "stage": 1, "error": "Object grounding failed"}
        
        # Stage 2: Object Tracking
        prediction_name, video_name, stage2_success = self.stage2_object_tracking(
            frames_dir, mask_path, target_fps, method
        )
        
        if not stage2_success:
            self.log("Stage 2 failed, aborting pipeline", "ERROR")
            return {"success": False, "stage": 2, "error": "Object tracking failed"}
        
        # Stage 3 & 4: State Analysis and Anomaly Detection
        anomaly_report, stage3_success = self.stage3_state_analysis(
            video_path, prediction_name, sample_interval, 
            detect_anomalies=True, vlm_model=vlm_model
        )
        
        if not stage3_success:
            self.log("Stage 3/4 failed", "WARNING")
        
        # Generate and display summary
        summary = self.generate_summary_report(anomaly_report, video_path)
        print(summary)
        
        # Save summary to file
        summary_path = osp.join(self.output_dir, "anomaly_reports", f"{video_name}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        return {
            "success": True,
            "video_path": video_path,
            "video_name": video_name,
            "frames_dir": frames_dir,
            "mask_path": mask_path,
            "prediction_name": prediction_name,
            "anomaly_report": anomaly_report,
            "summary": summary,
            "summary_path": summary_path
        }
    
    def batch_analyze(
        self,
        video_dir: str,
        video_extensions: List[str] = ['.mp4', '.avi', '.mov'],
        **kwargs
    ) -> List[Dict]:
        """
        Run ST-VAD pipeline on all videos in a directory.
        
        Args:
            video_dir: Directory containing videos
            video_extensions: List of valid video extensions
            **kwargs: Additional arguments for analyze_video
            
        Returns:
            List of analysis reports
        """
        self.log(f"Batch processing videos in: {video_dir}", "INFO")
        
        # Find all videos
        videos = []
        for ext in video_extensions:
            videos.extend(Path(video_dir).glob(f"*{ext}"))
        
        self.log(f"Found {len(videos)} videos to process", "INFO")
        
        results = []
        for i, video_path in enumerate(videos, 1):
            self.log(f"Processing video {i}/{len(videos)}: {video_path.name}", "INFO")
            try:
                result = self.analyze_video(str(video_path), **kwargs)
                results.append(result)
            except Exception as e:
                self.log(f"Error processing {video_path.name}: {str(e)}", "ERROR")
                results.append({
                    "success": False,
                    "video_path": str(video_path),
                    "error": str(e)
                })
        
        # Generate batch summary
        self.generate_batch_summary(results)
        
        return results
    
    def generate_batch_summary(self, results: List[Dict]):
        """Generate summary for batch processing."""
        total = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        with_anomalies = sum(
            1 for r in results 
            if r.get('success', False) and r.get('anomaly_report', {}).get('anomaly_detected', False)
        )
        
        summary = f"""
{'='*80}
BATCH PROCESSING SUMMARY
{'='*80}
Total Videos: {total}
Successfully Processed: {successful}
Failed: {total - successful}
Videos with Anomalies: {with_anomalies}
{'='*80}
"""
        print(summary)
        
        # Save batch summary
        summary_path = osp.join(self.output_dir, "batch_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
            f.write("\n\nDETAILED RESULTS:\n")
            f.write("="*80 + "\n")
            for r in results:
                video_name = Path(r['video_path']).name
                status = "✅ SUCCESS" if r.get('success', False) else "❌ FAILED"
                anomaly = "ANOMALY" if r.get('anomaly_report', {}).get('anomaly_detected', False) else "NORMAL"
                f.write(f"\n{video_name}: {status} | {anomaly}\n")
                if not r.get('success', False):
                    f.write(f"  Error: {r.get('error', 'Unknown error')}\n")


def get_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="ST-VAD Framework: Spatio-Temporal Video Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a single video
    python stvad_demo.py analyze video.mp4 -c configs/default.yaml
    
    # Analyze with custom settings
    python stvad_demo.py analyze video.mp4 -c configs/default.yaml --vlm claude --fps 15
    
    # Batch process videos in a directory
    python stvad_demo.py batch ./videos -c configs/default.yaml
    
    # Specify output directory
    python stvad_demo.py analyze video.mp4 -c configs/default.yaml -o ./results

File Structure:
    This script expects to be placed in the TubeletGraph directory with:
    TubeletGraph/
    ├── stvad_demo.py (this file)
    ├── quick_run.py
    ├── annotate/
    │   └── vlm_mask_grounded.py
    └── TubeletGraph/
        └── vlm/
            ├── prompt_vlm.py      # State tracking
            └── prompt_vad.py      # Anomaly detection (required)
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single video')
    analyze_parser.add_argument('video', help='Path to video file')
    analyze_parser.add_argument('-c', '--config', required=True, help='Path to TubeletGraph config')
    analyze_parser.add_argument('-o', '--output', default='output', help='Output directory')
    analyze_parser.add_argument('--vlm', default='openai', 
                               choices=['openai', 'claude', 'ollama'],
                               help='VLM provider (default: openai)')
    analyze_parser.add_argument('--fps', type=int, default=10, help='Target FPS')
    analyze_parser.add_argument('--sample_interval', type=int, default=10, help='Sampling interval')
    analyze_parser.add_argument('--method', default='Ours', help='Tracking method')
    analyze_parser.add_argument('--no-auto', action='store_true', help='Disable auto object detection')
    analyze_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process videos')
    batch_parser.add_argument('video_dir', help='Directory containing videos')
    batch_parser.add_argument('-c', '--config', required=True, help='Path to TubeletGraph config')
    batch_parser.add_argument('-o', '--output', default='output', help='Output directory')
    batch_parser.add_argument('--vlm', default='openai',
                             choices=['openai', 'claude', 'ollama'],
                             help='VLM provider (default: openai)')
    batch_parser.add_argument('--fps', type=int, default=10, help='Target FPS')
    batch_parser.add_argument('--sample_interval', type=int, default=10, help='Sampling interval')
    batch_parser.add_argument('--method', default='Ours', help='Tracking method')
    batch_parser.add_argument('--no-auto', action='store_true', help='Disable auto object detection')
    batch_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    return parser


def main():
    """Main entry point."""
    parser = get_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize framework
    framework = STVADFramework(
        config_path=args.config,
        output_dir=args.output,
        verbose=args.verbose
    )
    
    # Run appropriate command
    if args.command == 'analyze':
        result = framework.analyze_video(
            video_path=args.video,
            vlm_model=args.vlm,
            target_fps=args.fps,
            sample_interval=args.sample_interval,
            method=args.method,
            auto_mode=not args.no_auto
        )
        
        if not result.get('success', False):
            sys.exit(1)
    
    elif args.command == 'batch':
        results = framework.batch_analyze(
            video_dir=args.video_dir,
            vlm_model=args.vlm,
            target_fps=args.fps,
            sample_interval=args.sample_interval,
            method=args.method,
            auto_mode=not args.no_auto
        )
        
        # Exit with error if any video failed
        if any(not r.get('success', False) for r in results):
            sys.exit(1)


if __name__ == "__main__":
    main()