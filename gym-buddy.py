import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import random
from datetime import timedelta
import sys

class GymBuddy:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        
        # Screen dimensions
        self.SCREEN_WIDTH = 1280
        self.SCREEN_HEIGHT = 720
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.PURPLE = (149, 53, 235)
        self.LIGHT_PURPLE = (187, 134, 252)
        self.DARK_PURPLE = (98, 0, 238)
        self.GRAY = (50, 50, 50)
        self.GREEN = (76, 175, 80)
        self.RED = (244, 67, 54)
        
        # Fonts
        pygame.font.init()
        self.title_font = pygame.font.SysFont('Arial', 48, bold=True)
        self.subtitle_font = pygame.font.SysFont('Arial', 36)
        self.button_font = pygame.font.SysFont('Arial', 24)
        self.counter_font = pygame.font.SysFont('Arial', 72, bold=True)
        self.message_font = pygame.font.SysFont('Arial', 28)
        
        # Create screen
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption('Gym Buddy - Your AI Workout Assistant')
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)  # Width
        self.cap.set(4, 720)   # Height
        
        # Initialize MediaPipe Pose with lower thresholds
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Custom drawing specs for landmarks
        self.bicep_curl_connections = frozenset([
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST)
        ])
        
        # Custom landmark drawing styles
        self.custom_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Green color
            thickness=3,
            circle_radius=5
        )
        
        # Exercise tracking variables
        self.angle_history = []  # Store recent angles for smoothing
        self.history_size = 5    # Number of frames to average
        self.curl_started = False
        self.max_angle = 0       # Track maximum angle in current rep
        self.min_angle = 180     # Track minimum angle in current rep
        self.rep_in_progress = False
        self.current_exercise = None
        self.exercise_count = 0
        self.target_count = 0
        self.plank_start_time = None
        self.plank_duration = 0
        self.plank_target_duration = 60  # Default 60 seconds
        self.is_in_position = False
        self.prev_position = False
        
        # Messaging system
        self.motivational_messages = [
            "You're doing great! Keep it up!",
            "Stay strong! You've got this!",
            "Almost there! Push through!",
            "Feel the burn! It's worth it!",
            "Amazing form! Keep going!",
            "You're stronger than you think!",
            "Every rep counts! Don't give up!",
            "Breathe and focus! You can do this!"
        ]
        self.current_message = ""
        self.message_timer = 0
        
        # App state
        self.current_screen = "main_menu"
        self.workout_active = False
        
    def run(self):
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(pygame.mouse.get_pos())
            
            # Update screen based on current state
            if self.current_screen == "main_menu":
                self.show_main_menu()
            elif self.current_screen == "setup":
                self.show_setup_screen()
            elif self.current_screen == "plank":
                self.run_plank_detector()
            elif self.current_screen == "squat":
                self.run_squat_counter()
            elif self.current_screen == "bicep_curl":
                self.run_bicep_curl_counter()
            elif self.current_screen == "ab_crunch":
                self.run_ab_crunch_counter()
            elif self.current_screen == "results":
                self.show_results_screen()
            
            pygame.display.flip()
            clock.tick(30)
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        sys.exit()
    
    def handle_click(self, pos):
        x, y = pos
        
        if self.current_screen == "main_menu":
            # Plank button
            if 200 <= x <= 500 and 200 <= y <= 300:
                self.current_exercise = "plank"
                self.current_screen = "setup"
            # Squat button
            elif 200 <= x <= 500 and 350 <= y <= 450:
                self.current_exercise = "squat"
                self.current_screen = "setup"
            # Bicep curl button
            elif 700 <= x <= 1000 and 200 <= y <= 300:
                self.current_exercise = "bicep_curl"
                self.current_screen = "setup"
            # Ab crunch button
            elif 700 <= x <= 1000 and 350 <= y <= 450:
                self.current_exercise = "ab_crunch"
                self.current_screen = "setup"
        
        elif self.current_screen == "setup":
            # Start button
            if 500 <= x <= 800 and 500 <= y <= 600:
                self.workout_active = True
                self.exercise_count = 0
                self.plank_duration = 0
                self.plank_start_time = None
                self.message_timer = 0
                
                if self.current_exercise == "plank":
                    self.current_screen = "plank"
                elif self.current_exercise == "squat":
                    self.current_screen = "squat"
                elif self.current_exercise == "bicep_curl":
                    self.current_screen = "bicep_curl"
                elif self.current_exercise == "ab_crunch":
                    self.current_screen = "ab_crunch"
            
            # Back button
            elif 50 <= x <= 150 and 50 <= y <= 100:
                self.current_screen = "main_menu"
            
            # Target input buttons
            if self.current_exercise == "plank":
                # Decrease target
                if 600 <= x <= 650 and 300 <= y <= 350:
                    if self.plank_target_duration > 10:
                        self.plank_target_duration -= 10
                # Increase target
                elif 850 <= x <= 900 and 300 <= y <= 350:
                    self.plank_target_duration += 10
            else:
                # Decrease target
                if 600 <= x <= 650 and 300 <= y <= 350:
                    if self.target_count > 1:
                        self.target_count -= 1
                # Increase target
                elif 850 <= x <= 900 and 300 <= y <= 350:
                    self.target_count += 1
        
        elif self.current_screen in ["plank", "squat", "bicep_curl", "ab_crunch"]:
            # End workout button
            if 50 <= x <= 250 and 650 <= y <= 700:
                self.workout_active = False
                self.current_screen = "results"
        
        elif self.current_screen == "results":
            # Return to main menu
            if 500 <= x <= 800 and 500 <= y <= 600:
                self.current_screen = "main_menu"
    
    def draw_button(self, x, y, width, height, text, color, hover_color=None):
        mouse_pos = pygame.mouse.get_pos()
        if x <= mouse_pos[0] <= x + width and y <= mouse_pos[1] <= y + height:
            if hover_color:
                button_color = hover_color
            else:
                button_color = color
        else:
            button_color = color
        
        pygame.draw.rect(self.screen, button_color, (x, y, width, height), border_radius=10)
        pygame.draw.rect(self.screen, self.WHITE, (x, y, width, height), 2, border_radius=10)
        
        text_surface = self.button_font.render(text, True, self.WHITE)
        text_rect = text_surface.get_rect(center=(x + width/2, y + height/2))
        self.screen.blit(text_surface, text_rect)
    
    def show_main_menu(self):
        # Background
        self.screen.fill(self.BLACK)
        
        # Title
        title_text = self.title_font.render("GYM BUDDY", True, self.WHITE)
        subtitle_text = self.subtitle_font.render("Your AI Workout Assistant", True, self.LIGHT_PURPLE)
        self.screen.blit(title_text, (self.SCREEN_WIDTH//2 - title_text.get_width()//2, 80))
        self.screen.blit(subtitle_text, (self.SCREEN_WIDTH//2 - subtitle_text.get_width()//2, 140))
        
        # Exercise buttons
        self.draw_button(200, 200, 300, 100, "PLANK DETECTOR", self.PURPLE, self.LIGHT_PURPLE)
        self.draw_button(200, 350, 300, 100, "SQUAT COUNTER", self.PURPLE, self.LIGHT_PURPLE)
        self.draw_button(700, 200, 300, 100, "BICEP CURL COUNTER", self.PURPLE, self.LIGHT_PURPLE)
        self.draw_button(700, 350, 300, 100, "AB CRUNCH COUNTER", self.PURPLE, self.LIGHT_PURPLE)
        
        # Instructions
        instructions = self.message_font.render("Select an exercise to begin your workout", True, self.WHITE)
        self.screen.blit(instructions, (self.SCREEN_WIDTH//2 - instructions.get_width()//2, 550))
    
    def show_setup_screen(self):
        # Background
        self.screen.fill(self.BLACK)
        
        # Back button
        self.draw_button(50, 50, 100, 50, "Back", self.GRAY)
        
        # Title
        title_text = self.title_font.render(f"{self.current_exercise.replace('_', ' ').title()}", True, self.WHITE)
        self.screen.blit(title_text, (self.SCREEN_WIDTH//2 - title_text.get_width()//2, 100))
        
        # Target setup
        if self.current_exercise == "plank":
            target_text = self.subtitle_font.render("Set your target duration (seconds):", True, self.WHITE)
            self.screen.blit(target_text, (self.SCREEN_WIDTH//2 - target_text.get_width()//2, 200))
            
            # Target controls
            self.draw_button(600, 300, 50, 50, "-", self.GRAY)
            target_value = self.subtitle_font.render(f"{self.plank_target_duration}", True, self.WHITE)
            self.screen.blit(target_value, (self.SCREEN_WIDTH//2 - target_value.get_width()//2, 310))
            self.draw_button(850, 300, 50, 50, "+", self.GRAY)
        else:
            target_text = self.subtitle_font.render("Set your target repetitions:", True, self.WHITE)
            self.screen.blit(target_text, (self.SCREEN_WIDTH//2 - target_text.get_width()//2, 200))
            
            # Target controls
            self.draw_button(600, 300, 50, 50, "-", self.GRAY)
            target_value = self.subtitle_font.render(f"{self.target_count}", True, self.WHITE)
            self.screen.blit(target_value, (self.SCREEN_WIDTH//2 - target_value.get_width()//2, 310))
            self.draw_button(850, 300, 50, 50, "+", self.GRAY)
        
        # Instructions
        instructions = self.message_font.render("Adjust your target and press Start when ready", True, self.WHITE)
        self.screen.blit(instructions, (self.SCREEN_WIDTH//2 - instructions.get_width()//2, 400))
        
        # Start button
        self.draw_button(500, 500, 300, 100, "START WORKOUT", self.GREEN)
    
    def run_plank_detector(self):
        # Read camera feed
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Convert frame to pygame surface
        frame = cv2.resize(frame, (640, 360))
        frame = pygame.surfarray.make_surface(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
        frame = pygame.transform.rotate(frame, 270)
        
        # Background
        self.screen.fill(self.BLACK)
        
        # Display camera feed
        self.screen.blit(frame, (self.SCREEN_WIDTH//2 - frame.get_width()//2, 150))
        
        # End workout button
        self.draw_button(50, 650, 200, 50, "End Workout", self.RED)
        
        # Title
        title_text = self.title_font.render("Plank Detector", True, self.WHITE)
        self.screen.blit(title_text, (self.SCREEN_WIDTH//2 - title_text.get_width()//2, 50))
        
        # Check plank position
        if results.pose_landmarks:
            self.check_plank_position(results.pose_landmarks.landmark)
            
            # Draw pose landmarks on frame
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        # Display timer
        if self.plank_start_time:
            current_time = time.time()
            self.plank_duration = current_time - self.plank_start_time
            
            # Display duration
            duration_text = self.counter_font.render(
                str(timedelta(seconds=int(self.plank_duration))).split('.')[0], True, self.GREEN)
            self.screen.blit(duration_text, 
                            (self.SCREEN_WIDTH//2 - duration_text.get_width()//2, 550))
            
            # Target progress
            target_text = self.message_font.render(
                f"Target: {timedelta(seconds=int(self.plank_target_duration))}", True, self.WHITE)
            self.screen.blit(target_text, 
                            (self.SCREEN_WIDTH//2 - target_text.get_width()//2, 630))
            
            # Display motivational message
            if self.plank_duration > 0 and int(self.plank_duration) % 15 == 0 and time.time() - self.message_timer > 5:
                self.current_message = random.choice(self.motivational_messages)
                self.message_timer = time.time()
            
            # Check if target reached
            if self.plank_duration >= self.plank_target_duration and not self.message_timer:
                self.current_message = "Target achieved! Amazing work!"
                self.message_timer = time.time()
        
        else:
            instruction_text = self.message_font.render(
                "Get into plank position to start the timer", True, self.WHITE)
            self.screen.blit(instruction_text, 
                            (self.SCREEN_WIDTH//2 - instruction_text.get_width()//2, 550))
        
        # Display current message
        if self.current_message and time.time() - self.message_timer < 5:
            message_text = self.message_font.render(self.current_message, True, self.LIGHT_PURPLE)
            self.screen.blit(message_text, 
                            (self.SCREEN_WIDTH//2 - message_text.get_width()//2, 600))
    
    def check_plank_position(self, landmarks):
        # Get key landmarks for plank detection
        shoulders = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]]
        hips = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]]
        ankles = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]]
        
        # Calculate body angle for plank position
        shoulder_point = [(shoulders[0].x + shoulders[1].x)/2, (shoulders[0].y + shoulders[1].y)/2]
        hip_point = [(hips[0].x + hips[1].x)/2, (hips[0].y + hips[1].y)/2]
        ankle_point = [(ankles[0].x + ankles[1].x)/2, (ankles[0].y + ankles[1].y)/2]
        
        # Simple vector analysis for plank detection
        # For plank, we expect shoulders, hips, and ankles to form a relatively straight line
        vector1 = [hip_point[0] - shoulder_point[0], hip_point[1] - shoulder_point[1]]
        vector2 = [ankle_point[0] - hip_point[0], ankle_point[1] - hip_point[1]]
        
        # Normalize vectors
        v1_norm = np.sqrt(vector1[0]**2 + vector1[1]**2)
        v2_norm = np.sqrt(vector2[0]**2 + vector2[1]**2)
        
        if v1_norm > 0 and v2_norm > 0:
            vector1 = [vector1[0]/v1_norm, vector1[1]/v1_norm]
            vector2 = [vector2[0]/v2_norm, vector2[1]/v2_norm]
            
            # Calculate dot product
            dot_product = vector1[0]*vector2[0] + vector1[1]*vector2[1]
            
            # If dot product is close to 1 or -1, vectors are aligned
            # Also check if body is roughly horizontal
            is_aligned = abs(dot_product) > 0.7
            is_horizontal = abs(shoulder_point[1] - ankle_point[1]) < 0.3
            
            current_position = is_aligned and is_horizontal
            
            # Start timer when person gets into plank position
            if current_position and not self.is_in_position:
                self.plank_start_time = time.time()
                self.is_in_position = True
            
            # Reset timer if person breaks plank position
            elif not current_position and self.is_in_position:
                self.plank_start_time = None
                self.is_in_position = False
                self.current_message = "Plank position broken. Get back in position!"
                self.message_timer = time.time()
    
    def run_squat_counter(self):
        # Read camera feed
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Draw pose landmarks on frame if detected
        if results.pose_landmarks:
            self.check_squat_position(results.pose_landmarks.landmark)
            self.mp_drawing.draw_landmarks(
                rgb_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        # Convert the frame for display (minimal processing)
        frame = cv2.resize(rgb_frame, (960, 540))  # Larger size
        frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        
        # Background
        self.screen.fill(self.BLACK)
        
        # Display camera feed
        self.screen.blit(frame, (self.SCREEN_WIDTH//2 - frame.get_width()//2, 150))
        
        # End workout button
        self.draw_button(50, 650, 200, 50, "End Workout", self.RED)
        
        # Title
        title_text = self.title_font.render("Squat Counter", True, self.WHITE)
        self.screen.blit(title_text, (self.SCREEN_WIDTH//2 - title_text.get_width()//2, 50))
        
        # Display counter
        counter_text = self.counter_font.render(str(self.exercise_count), True, self.GREEN)
        self.screen.blit(counter_text, (self.SCREEN_WIDTH//2 - counter_text.get_width()//2, 550))
        
        # Target progress
        target_text = self.message_font.render(f"Target: {self.target_count} reps", True, self.WHITE)
        self.screen.blit(target_text, (self.SCREEN_WIDTH//2 - target_text.get_width()//2, 630))
        
        # Display motivational message
        if self.exercise_count > 0 and self.exercise_count % 5 == 0 and time.time() - self.message_timer > 5:
            self.current_message = random.choice(self.motivational_messages)
            self.message_timer = time.time()
        
        # Check if target reached
        if self.exercise_count >= self.target_count and self.target_count > 0 and time.time() - self.message_timer > 5:
            self.current_message = "Target achieved! Amazing work!"
            self.message_timer = time.time()
        
        # Display current message
        if self.current_message and time.time() - self.message_timer < 5:
            message_text = self.message_font.render(self.current_message, True, self.LIGHT_PURPLE)
            self.screen.blit(message_text, (self.SCREEN_WIDTH//2 - message_text.get_width()//2, 600))

    def check_ab_crunch_position(self, landmarks):
        # Get key landmarks for ab crunch detection
        shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        
        # Calculate angle between shoulder, hip, and knee
        angle = self.calculate_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (knee.x, knee.y))
        
        # Ab crunch logic: When a person does a crunch, the angle between shoulder, hip, and knee decreases
        # We count a rep when the angle goes below the threshold and then back above
        current_position = angle < 130  # Crunched position
        
        if current_position and not self.prev_position:
            # Starting a crunch
            self.prev_position = True
        elif not current_position and self.prev_position:
            # Finishing a crunch
            self.exercise_count += 1
            self.prev_position = False
            
            # Show message at certain milestones
            if self.exercise_count == 5:
                self.current_message = "Great core work! Keep those abs engaged!"
                self.message_timer = time.time()
            elif self.exercise_count == 10:
                self.current_message = "Feel that core burning! You're crushing it!"
                self.message_timer = time.time()

    def calculate_angle(self, p1, p2, p3):
        # Calculate angle between three points
        # p1, p2, p3 are tuples of (x, y) coordinates
        # Returns angle in degrees
        
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    def show_results_screen(self):
        # Background
        self.screen.fill(self.BLACK)
        
        # Title
        title_text = self.title_font.render("Workout Complete!", True, self.WHITE)
        self.screen.blit(title_text, (self.SCREEN_WIDTH//2 - title_text.get_width()//2, 100))
        
        # Exercise results
        if self.current_exercise == "plank":
            result_text = self.subtitle_font.render(
                f"You held the plank for {timedelta(seconds=int(self.plank_duration))}", True, self.GREEN)
        else:
            result_text = self.subtitle_font.render(
                f"You completed {self.exercise_count} {self.current_exercise.replace('_', ' ')}s", True, self.GREEN)
        
        self.screen.blit(result_text, (self.SCREEN_WIDTH//2 - result_text.get_width()//2, 250))
        
        # Congratulatory message
        congrats_text = self.message_font.render(
            "Great work! Keep up the momentum!", True, self.LIGHT_PURPLE)
        self.screen.blit(congrats_text, (self.SCREEN_WIDTH//2 - congrats_text.get_width()//2, 350))
        
        # Return to main menu button
        self.draw_button(500, 500, 300, 100, "RETURN TO MENU", self.PURPLE, self.LIGHT_PURPLE)

    def check_squat_position(self, landmarks):
        # Get key landmarks for squat detection
        hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        # Calculate knee angle
        angle = self.calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
        
        # Squat logic: When a person squats, the knee angle decreases and then increases
        # We count a rep when the angle goes below the threshold and then back above
        current_position = angle < 120  # Squatting position
        
        if current_position and not self.prev_position:
            # Starting a squat
            self.prev_position = True
        elif not current_position and self.prev_position:
            # Finishing a squat
            self.exercise_count += 1
            self.prev_position = False
            
            # Show message at certain milestones
            if self.exercise_count == 5:
                self.current_message = "You're doing great! Keep it up!"
                self.message_timer = time.time()
            elif self.exercise_count == 10:
                self.current_message = "Halfway there! You've got this!"
                self.message_timer = time.time()
    
    def run_bicep_curl_counter(self):
        # Read camera feed
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Draw pose landmarks on frame if detected
        if results.pose_landmarks:
            self.check_bicep_curl_position(results.pose_landmarks.landmark)
            
            # Draw only right arm landmarks with custom style
            self.mp_drawing.draw_landmarks(
                rgb_frame,
                results.pose_landmarks,
                self.bicep_curl_connections,
                landmark_drawing_spec=self.custom_drawing_spec,
                connection_drawing_spec=self.custom_drawing_spec
            )
        
        # Convert the frame for display (minimal processing)
        frame = cv2.resize(rgb_frame, (960, 540))  # Larger size
        frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        
        # Background
        self.screen.fill(self.BLACK)
        
        # Display camera feed
        self.screen.blit(frame, (self.SCREEN_WIDTH//2 - frame.get_width()//2, 150))
        
        # End workout button
        self.draw_button(50, 650, 200, 50, "End Workout", self.RED)
        
        # Title
        title_text = self.title_font.render("Bicep Curl Counter", True, self.WHITE)
        self.screen.blit(title_text, (self.SCREEN_WIDTH//2 - title_text.get_width()//2, 50))
        
        # Display counter
        counter_text = self.counter_font.render(str(self.exercise_count), True, self.GREEN)
        self.screen.blit(counter_text, (self.SCREEN_WIDTH//2 - counter_text.get_width()//2, 550))
        
        # Target progress
        target_text = self.message_font.render(f"Target: {self.target_count} reps", True, self.WHITE)
        self.screen.blit(target_text, (self.SCREEN_WIDTH//2 - target_text.get_width()//2, 630))
        
        # Display motivational message
        if self.exercise_count > 0 and self.exercise_count % 5 == 0 and time.time() - self.message_timer > 5:
            self.current_message = random.choice(self.motivational_messages)
            self.message_timer = time.time()
        
        # Check if target reached
        if self.exercise_count >= self.target_count and self.target_count > 0 and time.time() - self.message_timer > 5:
            self.current_message = "Target achieved! Amazing work!"
            self.message_timer = time.time()
        
        # Display current message
        if self.current_message and time.time() - self.message_timer < 5:
            message_text = self.message_font.render(self.current_message, True, self.LIGHT_PURPLE)
            self.screen.blit(message_text, (self.SCREEN_WIDTH//2 - message_text.get_width()//2, 600))

    def check_ab_crunch_position(self, landmarks):
        # Get key landmarks for ab crunch detection
        shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        
        # Calculate angle between shoulder, hip, and knee
        angle = self.calculate_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (knee.x, knee.y))
        
        # Ab crunch logic: When a person does a crunch, the angle between shoulder, hip, and knee decreases
        # We count a rep when the angle goes below the threshold and then back above
        current_position = angle < 130  # Crunched position
        
        if current_position and not self.prev_position:
            # Starting a crunch
            self.prev_position = True
        elif not current_position and self.prev_position:
            # Finishing a crunch
            self.exercise_count += 1
            self.prev_position = False
            
            # Show message at certain milestones
            if self.exercise_count == 5:
                self.current_message = "Great core work! Keep those abs engaged!"
                self.message_timer = time.time()
            elif self.exercise_count == 10:
                self.current_message = "Feel that core burning! You're crushing it!"
                self.message_timer = time.time()

    def calculate_angle(self, p1, p2, p3):
        # Calculate angle between three points
        # p1, p2, p3 are tuples of (x, y) coordinates
        # Returns angle in degrees
        
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    def show_results_screen(self):
        # Background
        self.screen.fill(self.BLACK)
        
        # Title
        title_text = self.title_font.render("Workout Complete!", True, self.WHITE)
        self.screen.blit(title_text, (self.SCREEN_WIDTH//2 - title_text.get_width()//2, 100))
        
        # Exercise results
        if self.current_exercise == "plank":
            result_text = self.subtitle_font.render(
                f"You held the plank for {timedelta(seconds=int(self.plank_duration))}", True, self.GREEN)
        else:
            result_text = self.subtitle_font.render(
                f"You completed {self.exercise_count} {self.current_exercise.replace('_', ' ')}s", True, self.GREEN)
        
        self.screen.blit(result_text, (self.SCREEN_WIDTH//2 - result_text.get_width()//2, 250))
        
        # Congratulatory message
        congrats_text = self.message_font.render(
            "Great work! Keep up the momentum!", True, self.LIGHT_PURPLE)
        self.screen.blit(congrats_text, (self.SCREEN_WIDTH//2 - congrats_text.get_width()//2, 350))
        
        # Return to main menu button
        self.draw_button(500, 500, 300, 100, "RETURN TO MENU", self.PURPLE, self.LIGHT_PURPLE)


        
    def check_bicep_curl_position(self, landmarks):
        # Get key landmarks for bicep curl detection (use right arm)
        shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # Calculate elbow angle
        angle = self.calculate_angle((shoulder.x, shoulder.y), (elbow.x, elbow.y), (wrist.x, wrist.y))
        
        # Add angle to history and maintain history size
        self.angle_history.append(angle)
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)
        
        # Get smoothed angle
        smoothed_angle = sum(self.angle_history) / len(self.angle_history)
        
        # Define thresholds
        CURL_START_ANGLE = 150   # Arm must be this straight to start
        CURL_END_ANGLE = 50      # Arm must curl this much to count
        MIN_ROM = 70            # Minimum range of motion required
        
        # Update max and min angles during the rep
        if self.rep_in_progress:
            self.max_angle = max(self.max_angle, smoothed_angle)
            self.min_angle = min(self.min_angle, smoothed_angle)
        
        # State machine for rep counting
        if not self.rep_in_progress:
            # Check if starting a new rep
            if smoothed_angle > CURL_START_ANGLE:
                self.rep_in_progress = True
                self.max_angle = smoothed_angle
                self.min_angle = smoothed_angle
                self.current_message = "Starting curl..."
                self.message_timer = time.time()
        else:
            # Check if completing a rep
            if smoothed_angle > CURL_START_ANGLE:
                # Check if the range of motion was sufficient
                range_of_motion = self.max_angle - self.min_angle
                if range_of_motion > MIN_ROM and self.min_angle < CURL_END_ANGLE:
                    self.exercise_count += 1
                    if self.exercise_count == 5:
                        self.current_message = "Feel those biceps burn! Keep going!"
                    elif self.exercise_count == 10:
                        self.current_message = "Great form! You're getting stronger!"
                    else:
                        self.current_message = f"Good rep! Range of motion: {range_of_motion:.1f}°"
                else:
                    self.current_message = "Incomplete rep - curl deeper!"
                self.message_timer = time.time()
                self.rep_in_progress = False
    
    def run_ab_crunch_counter(self):
        # Read camera feed
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Draw pose landmarks on frame if detected
        if results.pose_landmarks:
            self.check_ab_crunch_position(results.pose_landmarks.landmark)
            self.mp_drawing.draw_landmarks(
                rgb_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        # Convert the frame for display (minimal processing)
        frame = cv2.resize(rgb_frame, (960, 540))  # Larger size
        frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        
        # Background
        self.screen.fill(self.BLACK)
        
        # Display camera feed
        self.screen.blit(frame, (self.SCREEN_WIDTH//2 - frame.get_width()//2, 150))
        
        # End workout button
        self.draw_button(50, 650, 200, 50, "End Workout", self.RED)
        
        # Title
        title_text = self.title_font.render("Ab Crunch Counter", True, self.WHITE)
        self.screen.blit(title_text, (self.SCREEN_WIDTH//2 - title_text.get_width()//2, 50))
        
        # Display counter
        counter_text = self.counter_font.render(str(self.exercise_count), True, self.GREEN)
        self.screen.blit(counter_text, (self.SCREEN_WIDTH//2 - counter_text.get_width()//2, 550))
        
        # Target progress
        target_text = self.message_font.render(f"Target: {self.target_count} reps", True, self.WHITE)
        self.screen.blit(target_text, (self.SCREEN_WIDTH//2 - target_text.get_width()//2, 630))
        
        # Display motivational message
        if self.exercise_count > 0 and self.exercise_count % 5 == 0 and time.time() - self.message_timer > 5:
            self.current_message = random.choice(self.motivational_messages)
            self.message_timer = time.time()
        
        # Check if target reached
        if self.exercise_count >= self.target_count and self.target_count > 0 and time.time() - self.message_timer > 5:
            self.current_message = "Target achieved! Amazing work!"
            self.message_timer = time.time()
        
        # Display current message
        if self.current_message and time.time() - self.message_timer < 5:
            message_text = self.message_font.render(self.current_message, True, self.LIGHT_PURPLE)
            self.screen.blit(message_text, (self.SCREEN_WIDTH//2 - message_text.get_width()//2, 600))

    def check_ab_crunch_position(self, landmarks):
        # Get key landmarks for ab crunch detection
        shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        
        # Calculate angle between shoulder, hip, and knee
        angle = self.calculate_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (knee.x, knee.y))
        
        # Ab crunch logic: When a person does a crunch, the angle between shoulder, hip, and knee decreases
        # We count a rep when the angle goes below the threshold and then back above
        current_position = angle < 130  # Crunched position
        
        if current_position and not self.prev_position:
            # Starting a crunch
            self.prev_position = True
        elif not current_position and self.prev_position:
            # Finishing a crunch
            self.exercise_count += 1
            self.prev_position = False
            
            # Show message at certain milestones
            if self.exercise_count == 5:
                self.current_message = "Great core work! Keep those abs engaged!"
                self.message_timer = time.time()
            elif self.exercise_count == 10:
                self.current_message = "Feel that core burning! You're crushing it!"
                self.message_timer = time.time()

    def calculate_angle(self, p1, p2, p3):
        # Calculate angle between three points
        # p1, p2, p3 are tuples of (x, y) coordinates
        # Returns angle in degrees
        
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    def show_results_screen(self):
        # Background
        self.screen.fill(self.BLACK)
        
        # Title
        title_text = self.title_font.render("Workout Complete!", True, self.WHITE)
        self.screen.blit(title_text, (self.SCREEN_WIDTH//2 - title_text.get_width()//2, 100))
        
        # Exercise results
        if self.current_exercise == "plank":
            result_text = self.subtitle_font.render(
                f"You held the plank for {timedelta(seconds=int(self.plank_duration))}", True, self.GREEN)
        else:
            result_text = self.subtitle_font.render(
                f"You completed {self.exercise_count} {self.current_exercise.replace('_', ' ')}s", True, self.GREEN)
        
        self.screen.blit(result_text, (self.SCREEN_WIDTH//2 - result_text.get_width()//2, 250))
        
        # Congratulatory message
        congrats_text = self.message_font.render(
            "Great work! Keep up the momentum!", True, self.LIGHT_PURPLE)
        self.screen.blit(congrats_text, (self.SCREEN_WIDTH//2 - congrats_text.get_width()//2, 350))
        
        # Return to main menu button
        self.draw_button(500, 500, 300, 100, "RETURN TO MENU", self.PURPLE, self.LIGHT_PURPLE)

if __name__ == "__main__":
    app = GymBuddy()
    app.run()
