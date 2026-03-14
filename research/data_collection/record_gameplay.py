import time
import cv2
import mss
import numpy as np
import keyboard
import pygetwindow as gw
from pathlib import Path

# ========== CAPTURE REGION CONFIG ==========
# Mode 1: Auto-detect game window (may be inaccurate)
USE_AUTO_DETECT = False
WINDOW_TITLE = "Elsword"

# Mode 2: Manual capture region (RECOMMENDED)
# To get exact coordinates:
# 1. Open game at desired position
# 2. Run: python -c "import mss; print(mss.mss().monitors)"
# 3. Or use screenshot tool with pixel coordinate display
MANUAL_REGION = {
    "top": 210,      # Y coordinate from screen top
    "left": 560,     # X coordinate from screen left
    "width": 800,    # Capture region width
    "height": 600    # Capture region height
}
# ===========================================

# Recording configuration
TARGET_SIZE = (256, 192)  # Storage resolution (W, H) for training
OUTPUT_DIR = "../training/recorded"
GAMEPLAY_DURATION = 180   # 3 minutes gameplay
BUFFER_TIME = 15          # 15 seconds buffer
RECORDING_DURATION = GAMEPLAY_DURATION + BUFFER_TIME  # 195s = 3min15s
FPS = 20                  # 20 frames/second
MAX_FRAMES = RECORDING_DURATION * FPS  # 3,900 frames

NUM_GAMEPLAYS = 5         # Number of gameplays to record
STOP_KEY = 'f9'           # Key to stop current recording
START_KEY = 'f10'         # Key to start next recording

KEYS_TO_LOG = [
    'f8', 'up', 'down', 'left', 'right', 'z', 'x',  # Basic commands
    '1', '3',                                       # Potion commands (only 1 and 3)
    'q', 'w', 'e', 'r', 't',                        # Skill commands
    'a', 's', 'd', 'c', 'f',                        # Skill commands
    'enter', 'ctrl', 'esc'                          # Enter, Ctrl, Esc (skip scene)
]

def get_action_vector():
    """
    Get current state of all keys
    Returns: numpy array - each element 0 or 1
    Supports multi-key combos (e.g., up+z = [1,0,0,0,1,0,0,0])
    
    FIX: Ghost keys from numpad removed (left and down affected)
    """
    raw_vector = [1 if keyboard.is_pressed(k) else 0 for k in KEYS_TO_LOG]
    
    return np.array(raw_vector, dtype=np.uint8)

def action_vector_to_string(action_vec):
    """
    Convert action vector to readable string for debugging
    Example: [1,0,1,0,1,0,0,0] -> "up+left+z"
    """
    pressed_keys = [KEYS_TO_LOG[i] for i, val in enumerate(action_vec) if val == 1]
    return '+'.join(pressed_keys) if pressed_keys else 'none'

def get_game_region():
    """Return capture region - auto or manual"""
    
    # Use manual mode
    if not USE_AUTO_DETECT:
        print("🎯 Using manual capture region:")
        print(f"   Top: {MANUAL_REGION['top']}, Left: {MANUAL_REGION['left']}")
        print(f"   Size: {MANUAL_REGION['width']}x{MANUAL_REGION['height']}")
        return MANUAL_REGION
    
    # Auto mode (may be inaccurate)
    try:
        windows = gw.getWindowsWithTitle(WINDOW_TITLE)
        
        if not windows:
            print(f"❌ Window '{WINDOW_TITLE}' not found!")
            print("💡 Set USE_AUTO_DETECT = False and use MANUAL_REGION")
            return None
        
        # List all found windows
        print(f"🔍 Found {len(windows)} window(s):")
        for i, w in enumerate(windows):
            print(f"   [{i}] {w.title} - Pos({w.left},{w.top}) Size({w.width}x{w.height})")
        
        game_window = windows[0]
        
        if not game_window.isActive:
            try:
                game_window.activate()
            except:
                pass

        print(f"\n✅ Selected window: {game_window.title}")
        print(f"\n✅ Selected window: {game_window.title}")
        
        region = {
            "top": game_window.top, 
            "left": game_window.left, 
            "width": game_window.width, 
            "height": game_window.height
        }
        return region

    except Exception as e:
        print(f"❌ Error finding window: {e}")
        print("💡 Set USE_AUTO_DETECT = False and use MANUAL_REGION")
        return None

def collect_single_gameplay(gameplay_num, max_steps=None):
    """Record a single gameplay session"""
    if max_steps is None:
        max_steps = MAX_FRAMES
    
    # RAM estimation
    ram_needed_mb = (max_steps * TARGET_SIZE[0] * TARGET_SIZE[1] * 3) / (1024 * 1024)
    ram_needed_gb = ram_needed_mb / 1024
    duration_sec = max_steps / FPS
    
    print("\n" + "=" * 60)
    print(f"🎮 GAMEPLAY #{gameplay_num}")
    print(f"📊 Recording info:")
    print(f"   Resolution: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"   Max duration: {duration_sec:.1f}s ({duration_sec/60:.1f} min)")
    print(f"   FPS: {FPS} | Max frames: {max_steps}")
    print(f"   RAM needed: ~{ram_needed_gb:.2f} GB")
    print(f"⌨️  Controls:")
    print(f"   [{STOP_KEY.upper()}] - Stop recording early")
    print(f"   [{START_KEY.upper()}] - Start next gameplay")
    print("=" * 60)
    
    observations = []
    actions = []
    
    print(f"\n🚀 Press [{START_KEY.upper()}] to start recording gameplay #{gameplay_num}...")
    
    # Wait for user to press START key
    keyboard.wait(START_KEY)
    print("✅ Recording started!")
    time.sleep(0.5)  # Small delay to avoid capturing START key
    
    # Get capture region
    monitor_region = get_game_region()
    if not monitor_region:
        return None, None
    
    frame_time = 1.0 / FPS
    stopped_early = False
    
    with mss.mss() as sct:
        start_recording = time.time()
        
        for i in range(max_steps):
            frame_start = time.time()
            
            # Check stop key
            if keyboard.is_pressed(STOP_KEY):
                print(f"\n⏸️  Stopped early at frame {i+1}")
                stopped_early = True
                break
            
            # Capture frame
            sct_img = sct.grab(monitor_region)
            frame = np.array(sct_img)
            
            # Preprocessing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            resized_frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            action = get_action_vector()
            
            observations.append(resized_frame)
            actions.append(action)
            
            # Maintain consistent FPS
            elapsed = time.time() - frame_start
            if elapsed < frame_time: 
                time.sleep(frame_time - elapsed)
                
            # Progress update
            if i % 5 == 0 or i == max_steps - 1:
                elapsed_total = time.time() - start_recording
                fps_actual = (i + 1) / elapsed_total if elapsed_total > 0 else 0
                remaining_sec = (max_steps - i - 1) / FPS
                action_str = action_vector_to_string(action)
                print(f"Frame {i+1}/{max_steps} | FPS: {fps_actual:.1f} | Remaining: {remaining_sec:.0f}s | Action: {action_str}")
    
    total_frames = len(observations)
    actual_duration = total_frames / FPS
    
    # Analyze action stats
    actions_array = np.array(actions)
    unique_combos = len(np.unique(actions_array, axis=0))
    total_actions = np.sum(actions_array)
    
    print(f"\n✅ Completed! Recorded {total_frames} frames ({actual_duration:.1f}s)")
    print(f"📊 Action stats:")
    print(f"   Total key presses: {total_actions}")
    print(f"   Unique combos: {unique_combos}")
    
    return observations, actions

def collect_multiple_gameplays():
    """Record multiple gameplay sessions consecutively"""
    
    print("=" * 60)
    print("🎮 RECORD MULTIPLE GAMEPLAYS")
    print(f"   Count: {NUM_GAMEPLAYS} gameplays")
    print(f"   Duration per gameplay: ~{GAMEPLAY_DURATION}s + {BUFFER_TIME}s buffer")
    print("=" * 60)
    
    # Check capture region once
    print("\n🔍 Checking capture region...")
    monitor_region = get_game_region()
    if not monitor_region:
        print("❌ Cannot determine capture region. Exiting.")
        return
    
    all_observations = []
    all_actions = []
    
    for gameplay_idx in range(1, NUM_GAMEPLAYS + 1):
        obs, acts = collect_single_gameplay(gameplay_idx)
        
        if obs is None or acts is None:
            print(f"❌ Error recording gameplay #{gameplay_idx}")
            continue
        
        all_observations.extend(obs)
        all_actions.extend(acts)
        
        print(f"\n📊 Total recorded: {len(all_observations)} frames")
        
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = output_path / f"elsword_gameplay_{gameplay_idx:02d}.npz"
        print(f"💾 Saving {filename}...")
        np.savez_compressed(filename, obs=np.array(obs), act=np.array(acts))
        print(f"✅ Saved {filename}")
        
        if gameplay_idx < NUM_GAMEPLAYS:
            print(f"\n⏳ Preparing for gameplay #{gameplay_idx + 1}...")
            print(f"   Load game and press [{START_KEY.upper()}] when ready")
    
    total_duration = len(all_observations) / FPS
    total_ram_gb = (len(all_observations) * TARGET_SIZE[0] * TARGET_SIZE[1] * 3) / (1024**3)
    
    print(f"\n🎉 ALL COMPLETED!")
    print(f"   Total frames: {len(all_observations)}")
    print(f"   Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"   RAM used: ~{total_ram_gb:.2f} GB")
    print(f"   Saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    collect_multiple_gameplays()