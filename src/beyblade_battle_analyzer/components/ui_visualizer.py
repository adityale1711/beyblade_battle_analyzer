import cv2
import numpy as np
from typing import Dict, Any


class UIVisualizer:
    """
    UI Visualization component for Beyblade Battle Analyzer.
    Handles all visual overlay rendering, panel creation, and UI elements.
    """
    
    def __init__(self, arena_bounds=None):
        """
        Initialize the UI visualizer.
        
        :param arena_bounds: Arena boundary coordinates (x1, y1, x2, y2)
        """
        self.arena_bounds = arena_bounds
        
        # Color palettes for different UI elements
        self.beyblade_colors = [
            (255, 100, 100),  # Light red
            (100, 255, 100),  # Light green  
            (100, 100, 255),  # Light blue
            (255, 255, 100),  # Light yellow
            (255, 100, 255),  # Light magenta
            (100, 255, 255),  # Light cyan
        ]
        
        self.tracker_colors = [
            (255, 100, 100),  # Light red
            (100, 255, 100),  # Light green  
            (100, 100, 255),  # Light blue
            (255, 255, 100),  # Light yellow
            (255, 100, 255),  # Light magenta
            (100, 255, 255),  # Light cyan
        ]

    def create_enhanced_detection_visualization(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Creates enhanced visualization for detected Beyblades with modern styling.
        
        :param frame: Input video frame
        :param detections: List of detected Beyblades
        :return: Frame with enhanced detection visualization
        """
        vis_frame = frame.copy()
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            center = detection['center']
            color = self.beyblade_colors[i % len(self.beyblade_colors)]
            
            # Draw enhanced bounding box with rounded corners effect
            thickness = 3
            corner_length = 15
            
            # Top-left corner
            cv2.line(vis_frame, (bbox[0], bbox[1]), (bbox[0] + corner_length, bbox[1]), color, thickness)
            cv2.line(vis_frame, (bbox[0], bbox[1]), (bbox[0], bbox[1] + corner_length), color, thickness)
            
            # Top-right corner  
            cv2.line(vis_frame, (bbox[2], bbox[1]), (bbox[2] - corner_length, bbox[1]), color, thickness)
            cv2.line(vis_frame, (bbox[2], bbox[1]), (bbox[2], bbox[1] + corner_length), color, thickness)
            
            # Bottom-left corner
            cv2.line(vis_frame, (bbox[0], bbox[3]), (bbox[0] + corner_length, bbox[3]), color, thickness)
            cv2.line(vis_frame, (bbox[0], bbox[3]), (bbox[0], bbox[3] - corner_length), color, thickness)
            
            # Bottom-right corner
            cv2.line(vis_frame, (bbox[2], bbox[3]), (bbox[2] - corner_length, bbox[3]), color, thickness)
            cv2.line(vis_frame, (bbox[2], bbox[3]), (bbox[2], bbox[3] - corner_length), color, thickness)
            
            # Draw enhanced center point with pulse effect
            center_radius = 8
            cv2.circle(vis_frame, center, center_radius, color, -1)
            cv2.circle(vis_frame, center, center_radius + 3, color, 2)
            
            # Enhanced label with background
            label = f'Beyblade {i + 1}'
            confidence_text = f'{confidence:.1%}'
            
            # Calculate text dimensions
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)[0]
            conf_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
            
            # Position label above bounding box
            label_x = bbox[0]
            label_y = bbox[1] - 35
            
            # Draw label background
            padding = 8
            bg_width = max(label_size[0], conf_size[0]) + padding * 2
            bg_height = label_size[1] + conf_size[1] + padding * 3
            
            # Semi-transparent background
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (label_x - 5, label_y - bg_height + 5), 
                         (label_x + bg_width, label_y + 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
            
            # Draw border for label background
            cv2.rectangle(vis_frame, (label_x - 5, label_y - bg_height + 5), 
                         (label_x + bg_width, label_y + 10), color, 2)
            
            # Draw label text
            cv2.putText(vis_frame, label, (label_x + padding, label_y - conf_size[1] - 5),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_frame, confidence_text, (label_x + padding, label_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
            
        return vis_frame

    def draw_arena_bounds(self, frame: np.ndarray) -> None:
        """
        Draws enhanced arena bounds on the frame with modern styling.

        :param frame: The frame to draw the arena bounds on.
        """
        if self.arena_bounds:
            x1, y1, x2, y2 = self.arena_bounds
            
            # Ensure coordinates are valid
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Create an overlay for semi-transparent arena area
            overlay = frame.copy()
            
            # Fill the arena area with a subtle semi-transparent highlight
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
            
            # Blend the overlay with the original frame for transparency
            alpha = 0.08  # Very subtle transparency
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw the arena boundary with modern dashed effect
            dash_length = 15
            gap_length = 8
            thickness = 3
            color = (0, 255, 255)  # Cyan
            
            # Top edge (dashed)
            self._draw_dashed_line(frame, (x1, y1), (x2, y1), color, thickness, dash_length, gap_length)
            # Bottom edge (dashed)
            self._draw_dashed_line(frame, (x1, y2), (x2, y2), color, thickness, dash_length, gap_length)
            # Left edge (dashed)
            self._draw_dashed_line(frame, (x1, y1), (x1, y2), color, thickness, dash_length, gap_length)
            # Right edge (dashed)
            self._draw_dashed_line(frame, (x2, y1), (x2, y2), color, thickness, dash_length, gap_length)
            
            # Enhanced corner markers with glowing effect
            corner_size = 25
            corner_thickness = 4
            
            # Top-left corner with glow
            cv2.line(frame, (x1, y1), (x1 + corner_size, y1), (255, 255, 255), corner_thickness + 2)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_size), (255, 255, 255), corner_thickness + 2)
            cv2.line(frame, (x1, y1), (x1 + corner_size, y1), color, corner_thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_size), color, corner_thickness)
            
            # Top-right corner with glow
            cv2.line(frame, (x2, y1), (x2 - corner_size, y1), (255, 255, 255), corner_thickness + 2)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_size), (255, 255, 255), corner_thickness + 2)
            cv2.line(frame, (x2, y1), (x2 - corner_size, y1), color, corner_thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_size), color, corner_thickness)
            
            # Bottom-left corner with glow
            cv2.line(frame, (x1, y2), (x1 + corner_size, y2), (255, 255, 255), corner_thickness + 2)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_size), (255, 255, 255), corner_thickness + 2)
            cv2.line(frame, (x1, y2), (x1 + corner_size, y2), color, corner_thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_size), color, corner_thickness)
            
            # Bottom-right corner with glow
            cv2.line(frame, (x2, y2), (x2 - corner_size, y2), (255, 255, 255), corner_thickness + 2)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_size), (255, 255, 255), corner_thickness + 2)
            cv2.line(frame, (x2, y2), (x2 - corner_size, y2), color, corner_thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_size), color, corner_thickness)
            
            # Enhanced arena label with modern styling
            arena_text = "BATTLE ARENA"
            text_size = cv2.getTextSize(arena_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
            
            # Position the text at the top center of the arena
            text_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_y = y1 - 15 if y1 - 15 > text_size[1] else y1 + text_size[1] + 15
            
            # Draw enhanced background for the text
            padding = 12
            bg_x1 = text_x - padding
            bg_y1 = text_y - text_size[1] - padding
            bg_x2 = text_x + text_size[0] + padding
            bg_y2 = text_y + padding
            
            # Background with gradient effect
            bg_overlay = frame.copy()
            cv2.rectangle(bg_overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(bg_overlay, 0.7, frame, 0.3, 0, frame)
            
            # Border for text background
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
            
            # Draw the text with shadow effect
            cv2.putText(frame, arena_text, (text_x + 2, text_y + 2), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 3)  # Shadow
            cv2.putText(frame, arena_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

    def _draw_dashed_line(self, frame: np.ndarray, pt1: tuple, pt2: tuple, color: tuple, thickness: int, dash_length: int, gap_length: int) -> None:
        """
        Draws a dashed line between two points.
        
        :param frame: Frame to draw on
        :param pt1: Start point (x, y)
        :param pt2: End point (x, y)
        :param color: Line color
        :param thickness: Line thickness
        :param dash_length: Length of each dash
        :param gap_length: Length of each gap
        """
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Calculate line length and direction
        dx = x2 - x1
        dy = y2 - y1
        line_length = np.sqrt(dx**2 + dy**2)
        
        if line_length == 0:
            return
            
        # Normalize direction
        dx_norm = dx / line_length
        dy_norm = dy / line_length
        
        # Draw dashed line
        current_length = 0
        cycle_length = dash_length + gap_length
        
        while current_length < line_length:
            # Start of current dash
            dash_start_x = int(x1 + current_length * dx_norm)
            dash_start_y = int(y1 + current_length * dy_norm)
            
            # End of current dash
            dash_end_length = min(current_length + dash_length, line_length)
            dash_end_x = int(x1 + dash_end_length * dx_norm)
            dash_end_y = int(y1 + dash_end_length * dy_norm)
            
            # Draw the dash
            cv2.line(frame, (dash_start_x, dash_start_y), (dash_end_x, dash_end_y), color, thickness)
            
            # Move to next cycle
            current_length += cycle_length

    def draw_modern_ui_overlay(self, frame: np.ndarray, analysis: Dict[str, Any], detections: list, battle_analyzer=None) -> None:
        """
        Draws a modern UI overlay with comprehensive battle information.
        
        :param frame: Frame to draw on
        :param analysis: Battle analysis data
        :param detections: Detection results
        :param battle_analyzer: Battle analyzer instance for tracker access
        """
        height, width = frame.shape[:2]
        
        # Create main UI panel
        self.draw_main_status_panel(frame, analysis, detections)
        
        # Create battle statistics panel
        self.draw_battle_stats_panel(frame, analysis, battle_analyzer)
        
        # Create Beyblade performance panel
        if battle_analyzer and hasattr(battle_analyzer, 'trackers') and battle_analyzer.trackers:
            self.draw_performance_panel(frame, analysis, battle_analyzer)
        
        # Draw frame counter and timestamp
        self.draw_frame_info(frame, analysis, battle_analyzer)

    def draw_main_status_panel(self, frame: np.ndarray, analysis: Dict[str, Any], detections: list) -> None:
        """
        Draws the main status panel with battle state and key information.
        """
        height, width = frame.shape[:2]
        
        # Panel dimensions
        panel_width = 320
        panel_height = 140
        panel_x = 20
        panel_y = 20
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Draw panel border with gradient effect
        border_color = self._get_battle_state_color(analysis['battle_state'])
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     border_color, 3)
        
        # Title section
        title_y = panel_y + 30
        cv2.putText(frame, "BEYBLADE BATTLE ANALYZER", (panel_x + 10, title_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
        
        # Battle state with icon
        state_y = title_y + 35
        state_text = f"Status: {analysis['battle_state'].upper()}"
        cv2.putText(frame, state_text, (panel_x + 10, state_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, border_color, 2)
        
        # Active Beyblades with visual indicator
        active_y = state_y + 30
        active_text = f"Active Beyblades: {analysis['active_beyblades']}"
        cv2.putText(frame, active_text, (panel_x + 10, active_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (100, 255, 100), 2)
        
        # Winner information
        if analysis.get('winner') is not None:
            winner_y = active_y + 30
            winner_text = f"Winner: Beyblade {analysis['winner']}"
            cv2.putText(frame, winner_text, (panel_x + 10, winner_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw crown icon next to winner
            crown_x = panel_x + 250
            crown_y = winner_y - 15
            self._draw_crown_icon(frame, crown_x, crown_y)

    def draw_battle_stats_panel(self, frame: np.ndarray, analysis: Dict[str, Any], battle_analyzer=None) -> None:
        """
        Draws battle statistics panel with progress bars and metrics.
        """
        height, width = frame.shape[:2]
        
        # Panel dimensions (right side)
        panel_width = 280
        panel_height = 200
        panel_x = width - panel_width - 20
        panel_y = 20
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (20, 40, 60), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw panel border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (100, 150, 200), 2)
        
        # Title
        title_y = panel_y + 25
        cv2.putText(frame, "BATTLE STATISTICS", (panel_x + 10, title_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw battle progress if battle analyzer has trackers
        if battle_analyzer and hasattr(battle_analyzer, 'trackers') and battle_analyzer.trackers:
            self._draw_battle_progress_bars(frame, panel_x, panel_y, panel_width, analysis, battle_analyzer)
        
        # Detection statistics
        detection_y = title_y + 40
        detection_text = f"Detections: {analysis['detections_count']}"
        cv2.putText(frame, detection_text, (panel_x + 10, detection_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def draw_performance_panel(self, frame: np.ndarray, analysis: Dict[str, Any], battle_analyzer) -> None:
        """
        Draws individual Beyblade performance metrics.
        """
        height, width = frame.shape[:2]
        
        # Panel dimensions (bottom)
        panel_width = width - 40
        panel_height = 120
        panel_x = 20
        panel_y = height - panel_height - 20
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (40, 20, 60), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Draw panel border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (150, 100, 200), 2)
        
        # Title
        title_y = panel_y + 25
        cv2.putText(frame, "BEYBLADE PERFORMANCE", (panel_x + 10, title_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw performance metrics for each tracker
        self._draw_tracker_performance_metrics(frame, panel_x, panel_y, panel_width, panel_height, battle_analyzer)

    def _draw_tracker_performance_metrics(self, frame: np.ndarray, panel_x: int, panel_y: int, panel_width: int, panel_height: int, battle_analyzer) -> None:
        """
        Draws performance metrics for individual trackers.
        """
        trackers = battle_analyzer.trackers
        if not trackers:
            return
            
        start_y = panel_y + 45
        col_width = (panel_width - 40) // min(len(trackers), 2)  # Max 2 columns

        # Skip to last 2 trackers if there are more than 2
        trackers_list = list(trackers.items())
        start_index = max(0, len(trackers_list) - 2)  # Start from 2nd last or earlier
        
        for i, (tracker_id, tracker) in enumerate(trackers.items()):
            if start_index + i >= len(trackers_list):
                break

            tracker_id, tracker = trackers_list[start_index + i]
            col_x = panel_x + 20 + (i * col_width)
            color = self.tracker_colors[i % len(self.tracker_colors)]

            # Tracker ID
            cv2.putText(frame, f"Beyblade {i + 1}", (col_x, start_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Spinning status
            status_text = "SPINNING" if tracker.is_spinning else "STOPPED"
            status_color = (100, 255, 100) if tracker.is_spinning else (255, 100, 100)
            cv2.putText(frame, status_text, (col_x, start_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
            
            # Velocity bar
            if tracker.velocities:
                avg_velocity = np.mean(tracker.velocities[-10:])  # Last 10 velocities
                max_velocity = 50  # Assume max velocity for normalization
                velocity_ratio = min(1.0, avg_velocity / max_velocity)
                
                bar_width = col_width - 20
                bar_height = 8
                bar_x = col_x
                bar_y = start_y + 35
                
                # Background bar
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                             (100, 100, 100), -1)
                
                # Velocity bar
                filled_width = int(bar_width * velocity_ratio)
                if filled_width > 0:
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                                 color, -1)
                
                # Velocity text
                cv2.putText(frame, f"Vel: {avg_velocity:.1f}", (col_x, start_y + 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    def _draw_battle_progress_bars(self, frame: np.ndarray, panel_x: int, panel_y: int, panel_width: int, analysis: Dict[str, Any], battle_analyzer) -> None:
        """
        Draws progress bars showing battle progression.
        """
        if not hasattr(battle_analyzer, 'trackers') or not battle_analyzer.trackers:
            return
            
        total_trackers = len(battle_analyzer.trackers)
        active_trackers = analysis['active_beyblades']
        
        # Battle progress bar
        progress_y = panel_y + 100
        cv2.putText(frame, "Battle Progress:", (panel_x + 10, progress_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Progress bar background
        bar_x = panel_x + 10
        bar_y = progress_y + 10
        bar_width = panel_width - 70
        bar_height = 12
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Calculate progress (inverse of active trackers ratio)
        if total_trackers > 0:
            progress_ratio = (total_trackers - active_trackers) / total_trackers
            filled_width = int(bar_width * progress_ratio)
            
            if filled_width > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                             (100, 255, 100), -1)
        
        # Progress percentage
        progress_percent = int(progress_ratio * 100) if total_trackers > 0 else 0
        cv2.putText(frame, f"{progress_percent}%", (bar_x + bar_width + 10, bar_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def draw_frame_info(self, frame: np.ndarray, analysis: Dict[str, Any], battle_analyzer=None) -> None:
        """
        Draws frame counter and timing information.
        """
        height, width = frame.shape[:2]
        
        # Frame info panel (bottom right)
        panel_width = 200
        panel_height = 60
        panel_x = width - panel_width - 20
        panel_y = height - panel_height - 150
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (100, 100, 100), 1)
        
        # Frame number
        frame_text = f"Frame: {analysis['frame_number']}"
        cv2.putText(frame, frame_text, (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Time calculation (approximate)
        if battle_analyzer and hasattr(battle_analyzer, 'fps'):
            time_seconds = analysis['frame_number'] / battle_analyzer.fps
            time_text = f"Time: {time_seconds:.1f}s"
            cv2.putText(frame, time_text, (panel_x + 10, panel_y + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _get_battle_state_color(self, battle_state: str) -> tuple:
        """
        Returns color based on battle state.
        """
        state_colors = {
            'starting': (255, 255, 100),    # Yellow
            'active': (100, 255, 100),      # Green
            'ending': (255, 150, 100),      # Orange
            'finished': (255, 100, 100)     # Red
        }
        return state_colors.get(battle_state.lower(), (255, 255, 255))

    def _draw_crown_icon(self, frame: np.ndarray, x: int, y: int) -> None:
        """
        Draws a simple crown icon for the winner.
        """
        # Crown base
        points = np.array([
            [x, y + 15],
            [x + 5, y + 5], 
            [x + 10, y + 10],
            [x + 15, y],
            [x + 20, y + 10],
            [x + 25, y + 5],
            [x + 30, y + 15],
            [x + 25, y + 20],
            [x + 5, y + 20]
        ], np.int32)
        
        cv2.fillPoly(frame, [points], (255, 215, 0))  # Gold color
        cv2.polylines(frame, [points], True, (255, 255, 255), 1)

    def create_annotated_frame(self, frame: np.ndarray, detections: list, analysis: Dict[str, Any], battle_analyzer=None) -> np.ndarray:
        """
        Creates a fully annotated frame with enhanced UI visualization.
        
        :param frame: The original video frame
        :param detections: List of detected Beyblades
        :param analysis: Analysis results for the frame
        :param battle_analyzer: Battle analyzer instance for tracker access
        :return: Fully annotated video frame
        """
        # Start with enhanced detection visualization
        annotated = self.create_enhanced_detection_visualization(frame, detections)

        # Draw arena bounds if available
        if self.arena_bounds:
            self.draw_arena_bounds(annotated)

        # Create modern UI overlay
        self.draw_modern_ui_overlay(annotated, analysis, detections, battle_analyzer)

        return annotated

    def set_arena_bounds(self, arena_bounds):
        """
        Set or update the arena bounds.
        
        :param arena_bounds: Arena boundary coordinates (x1, y1, x2, y2)
        """
        self.arena_bounds = arena_bounds

    def visualize_simple_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Creates simple detection visualization with corner markers and text overlays.
        This method provides the same styling as the original BeybladeDetector.visualize_detections.
        
        :param frame: Input video frame
        :param detections: List of detected Beyblades with bounding boxes and confidence scores
        :return: Frame with simple detection visualization
        """
        # Create a copy of the frame to visualize detections
        vis_frame = frame.copy()

        # Color palette for different Beyblades with better contrast
        colors = [
            (50, 255, 50),    # Bright green
            (255, 50, 50),    # Bright red
            (50, 50, 255),    # Bright blue
            (255, 255, 50),   # Bright yellow
            (255, 50, 255),   # Bright magenta
            (50, 255, 255),   # Bright cyan
        ]

        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            center = detection['center']
            color = colors[i % len(colors)]

            # Draw enhanced bounding box with corner markers
            thickness = 2
            corner_length = 12
            
            # Corner markers for better visibility
            # Top-left
            cv2.line(vis_frame, (bbox[0], bbox[1]), (bbox[0] + corner_length, bbox[1]), color, thickness)
            cv2.line(vis_frame, (bbox[0], bbox[1]), (bbox[0], bbox[1] + corner_length), color, thickness)
            
            # Top-right  
            cv2.line(vis_frame, (bbox[2], bbox[1]), (bbox[2] - corner_length, bbox[1]), color, thickness)
            cv2.line(vis_frame, (bbox[2], bbox[1]), (bbox[2], bbox[1] + corner_length), color, thickness)
            
            # Bottom-left
            cv2.line(vis_frame, (bbox[0], bbox[3]), (bbox[0] + corner_length, bbox[3]), color, thickness)
            cv2.line(vis_frame, (bbox[0], bbox[3]), (bbox[0], bbox[3] - corner_length), color, thickness)
            
            # Bottom-right
            cv2.line(vis_frame, (bbox[2], bbox[3]), (bbox[2] - corner_length, bbox[3]), color, thickness)
            cv2.line(vis_frame, (bbox[2], bbox[3]), (bbox[2], bbox[3] - corner_length), color, thickness)

            # Draw enhanced center point with double circle
            cv2.circle(vis_frame, center, 6, color, -1)
            cv2.circle(vis_frame, center, 9, (255, 255, 255), 2)
            
            # Enhanced label with better positioning and styling
            label = f'Beyblade {i + 1}'
            confidence_text = f'{confidence:.1%}'
            
            # Position label above bounding box
            label_x = bbox[0]
            label_y = bbox[1] - 25 if bbox[1] > 30 else bbox[3] + 25
            
            # Draw text with outline for better readability
            cv2.putText(vis_frame, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 3)  # Black outline
            cv2.putText(vis_frame, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)  # Colored text
            
            # Confidence text
            cv2.putText(vis_frame, confidence_text, (label_x, label_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
            cv2.putText(vis_frame, confidence_text, (label_x, label_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text

        return vis_frame
