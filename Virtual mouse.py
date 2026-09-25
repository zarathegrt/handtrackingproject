import cv2
import mediapipe as mp
import time
import os
import math
import ctypes
from datetime import datetime
from collections import deque

import numpy as np
import pyautogui


try:
    from winotify import Notification
    WINOTIFY_AVAILABLE = True
except ImportError:
    WINOTIFY_AVAILABLE = False

# ========================
# CONFIG
# ========================
SCREEN_W, SCREEN_H = pyautogui.size()
pyautogui.FAILSAFE = False

DEBUG = False  

CAM_W, CAM_H = 640, 480

# Multiple smoothing techniques combined for buttery-smooth cursor movement
SMOOTHENING = 5              # Higher = smoother but more lag (3-8 recommended)
SMOOTHING_HISTORY = 8        # Number of recent positions to average
SMOOTHING_WEIGHT = 0.6       # Weight for exponential moving average (0.3-0.7)
JITTER_THRESHOLD = 3.0       # Minimum movement to register (pixels)
PREDICTION_WEIGHT = 0.15     # Slight movement prediction (0.05-0.25)

# --- pinch-family thresholds (all in camera pixels, thumb-tip to fingertip) ---
CLICK_DIST = 40          # thumb + index  -> left click
RIGHT_CLICK_DIST = 40    # thumb + middle -> right click
DOUBLE_CLICK_DIST = 40   # thumb + ring   -> double click
DRAG_START_DIST = 40     # thumb + pinky  -> start drag
DRAG_RELEASE_DIST = 55   # bigger than DRAG_START_DIST on purpose: this hysteresis gap
                          
PINCH_SEPARATION = 12    # the winning fingertip must be at least this much closer to the
                          # thumb than every other fingertip, or the pinch is rejected as
                          # ambiguous (returns None) instead of guessing which one you meant

# --- thumb-direction geometry (Volume Up/Down) ---
THUMB_EXTENSION_MIN = 35     # thumb tip must be at least this far from its MCP joint
                              # to count as "extended" rather than curled into the fist
THUMB_VERTICAL_RATIO = 1.15  # vertical component of the thumb vector must exceed the
                              # horizontal component by this factor to call it "up"/"down"
                              # rather than "sideways" (ambiguous -> NEUTRAL)

# --- volume pending/lock state machine ---
VOLUME_CONFIRM_FRAMES = 3    # consecutive frames a candidate must hold before it LOCKS in
VOLUME_RELEASE_FRAMES = 5    # consecutive non-matching frames needed to release the lock

# --- thumb-folded geometry (Screenshot vs. Move) ---
THUMB_FOLDED_MAX = 40        # thumb tip within this distance of its MCP joint counts as
                              # "tucked across the palm" rather than openly extended

# --- zoom (middle+ring fingertip distance, tracked over time) ---
ZOOM_HISTORY_WINDOW = 0.35   # seconds of recent samples considered
ZOOM_MIN_DELTA = 18          # px change within the window needed to count as intentional
ZOOM_COOLDOWN = 0.35         # seconds between successive zoom hotkey fires

# --- NEW: One-hand pinch zoom using thumb-index distance ---
PINCH_ZOOM_DEADZONE = 0.03       # Minimum normalized distance change to trigger zoom
PINCH_ZOOM_MIN_INTERVAL = 0.08   # Seconds between zoom updates (throttle)
PINCH_ZOOM_STEP_SCALE = 8.0      # Steps per unit of normalized distance change
PINCH_ZOOM_MIN_RATIO = 0.05      # Minimum thumb-index distance ratio
PINCH_ZOOM_MAX_RATIO = 0.80      # Maximum thumb-index distance ratio

# --- Alt+Tab (three-finger horizontal swipe) ---
SWIPE_HISTORY_WINDOW = 0.5   # seconds — a deliberate swipe should complete within this
SWIPE_MIN_DISTANCE = 90      # px of horizontal index-tip movement needed to confirm a swipe

# --- confirmation frames (require the pose to hold steady before firing) ---
RIGHT_CLICK_CONFIRM_FRAMES = 2
DOUBLE_CLICK_CONFIRM_FRAMES = 2
SCREENSHOT_STABLE_FRAMES = 3
MUTE_CONFIRM_FRAMES = 2      # SIMPLIFIED: Only need confirmation frames
PLAYPAUSE_CONFIRM_FRAMES = 2

# --- COPY / CUT / PASTE confirmation frames ---
COPY_CONFIRM_FRAMES = 3     # require 3 stable frames before executing
CUT_CONFIRM_FRAMES = 3
PASTE_CONFIRM_FRAMES = 1    # OPTIMIZED: Require only 1 frame for Paste (more responsive)

# --- cooldowns (seconds) — throttle continuous/repeatable actions, guard one-shot ones ---
CLICK_COOLDOWN = 0.3
RIGHT_CLICK_COOLDOWN = 0.4
DOUBLE_CLICK_COOLDOWN = 0.5
SCROLL_COOLDOWN = 0.12
SCROLL_NOTCHES = 1
VOLUME_COOLDOWN = 0.3
SCREENSHOT_COOLDOWN = 1.5
MUTE_COOLDOWN = 0.15   # Kept as a safety net, but mute_latched now handles the edge trigger
COPY_COOLDOWN = 0.5    # cooldown for copy/cut/paste to prevent accidental repeats
CUT_COOLDOWN = 0.5
PASTE_COOLDOWN = 0.3   # OPTIMIZED: Slightly reduced cooldown for Paste


# ========================
# WINDOWS WHEEL EVENT (real hardware-level scroll)
# ========================
_MOUSEEVENTF_WHEEL = 0x0800
_WHEEL_DELTA = 120

def windows_scroll(notches):
    """notches: positive = scroll up, negative = scroll down."""
    ctypes.windll.user32.mouse_event(_MOUSEEVENTF_WHEEL, 0, 0, int(notches * _WHEEL_DELTA), 0)


# ========================
# HAND DETECTOR
# ========================
class HandDetector:
    def __init__(self, detectionCon=0.7, trackCon=0.7):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    img, handLms, self.mpHands.HAND_CONNECTIONS
                )
        return img

    def findPosition(self, img):
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            h, w, _ = img.shape
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList


# ========================
# LOW-LEVEL HELPERS
# ========================
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def fingersUp(lmList):
    """
    Returns [thumb, index, middle, ring, pinky] as 1 (extended) / 0 (curled).
    NOTE: fingers[0] (thumb) is x-coordinate based and flips with hand rotation —
    kept only for debug display. No gesture uses it as a gate; thumb-based
    gestures use detectThumbDirection()/isThumbFolded() (geometry) instead.
    """
    fingers = []
    fingers.append(lmList[4][1] > lmList[3][1])  # thumb (debug display only)

    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(lmList[tip][2] < lmList[tip - 2][2])

    return [int(f) for f in fingers]


def detectThumbDirection(lmList):
    """
    Returns "UP", "DOWN", or "NEUTRAL" using the full thumb landmark chain
    (wrist -> MCP -> IP -> TIP), not just a single joint pair — this rejects a
    thumb that's merely bent/rotated in frame rather than genuinely pointing
    up or down, and stays correct regardless of hand rotation.
    Landmarks: 0 = wrist, 2 = thumb MCP, 3 = thumb IP, 4 = thumb tip.
    """
    wrist = lmList[0]
    mcp = lmList[2]
    ip = lmList[3]
    tip = lmList[4]

    vx = tip[1] - mcp[1]
    vy = tip[2] - mcp[2]
    extension = math.hypot(vx, vy)

    if extension < THUMB_EXTENSION_MIN:
        return "NEUTRAL"  # tucked in, not extended enough to judge direction

    if abs(vy) < abs(vx) * THUMB_VERTICAL_RATIO:
        return "NEUTRAL"  # pointing too sideways to call it clearly up or down

    # Image y grows downward. Require the WHOLE chain to agree on direction
    # (tip clearly past ip, past mcp, past wrist) rather than just one joint
    # pair, so a bent-but-not-really-pointing thumb doesn't slip through.
    if vy > 0 and tip[2] > ip[2] and tip[2] > mcp[2] and tip[2] > wrist[2]:
        return "DOWN"
    if vy < 0 and tip[2] < ip[2] and tip[2] < mcp[2] and tip[2] < wrist[2]:
        return "UP"
    return "NEUTRAL"


def isThumbFolded(lmList):
    """
    True when the thumb tip sits close to its own MCP joint — i.e. tucked
    across the palm rather than extended outward. Used to tell MOVE (open
    hand, thumb out) apart from SCREENSHOT (open hand, thumb tucked in).
    """
    mcp = lmList[2]
    tip = lmList[4]
    extension = math.hypot(tip[1] - mcp[1], tip[2] - mcp[2])
    return extension < THUMB_FOLDED_MAX


def classifyPinch(d_index, d_middle, d_ring, d_pinky):
    """
    Returns "index" / "middle" / "ring" / "pinky" / None.

    Picks whichever fingertip is closest to the thumb, but only commits to it
    if (a) that distance is under ITS OWN gesture threshold, AND (b) it's
    clearly closer than every other fingertip by at least PINCH_SEPARATION.
    If two fingertips are nearly tied, this returns None rather than guessing —
    which is exactly the case that previously let "thumb near middle" also
    register as a left click just because the index happened to be nearby too.
    """
    candidates = {
        "index": (d_index, CLICK_DIST),
        "middle": (d_middle, RIGHT_CLICK_DIST),
        "ring": (d_ring, DOUBLE_CLICK_DIST),
        "pinky": (d_pinky, DRAG_START_DIST),
    }
    raw = {name: val[0] for name, val in candidates.items()}

    closest_finger = min(raw, key=raw.get)
    closest_distance = raw[closest_finger]
    threshold = candidates[closest_finger][1]

    if closest_distance >= threshold:
        return None

    for finger, dist in raw.items():
        if finger != closest_finger and dist < closest_distance + PINCH_SEPARATION:
            return None  # another fingertip is too close a competitor — ambiguous

    return closest_finger


def takeScreenshot():
    """
    Captures the full screen and saves it to an absolute screenshots/ folder.
    Returns (success: bool, filepath_or_None) so the caller can tell a real
    save apart from a failure and react accordingly.
    """
    try:
        screenshot_dir = os.path.abspath("screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join(screenshot_dir, f"screenshot_{timestamp}.png")

        screenshot = pyautogui.screenshot()
        screenshot.save(filepath)

        print(f"SCREENSHOT SAVED: {filepath}")
        return True, filepath

    except Exception as e:
        print(f"SCREENSHOT ERROR: {e}")
        return False, None


def showScreenshotNotification(filepath):
    """
    Shows a Windows toast notification after a screenshot is successfully
    saved. Safe to call even if winotify isn't installed or the toast fails
    for any reason — it just prints a message instead of crashing SilentFlow.
    """
    if not WINOTIFY_AVAILABLE:
        print("Notification skipped: winotify not installed (pip install winotify)")
        return

    try:
        toast = Notification(
            app_id="SilentFlow",
            title="Screenshot Captured",
            msg=f"Screenshot saved successfully!\n{filepath}",
            duration="short"
        )
        toast.show()
    except Exception as e:
        print(f"Notification error: {e}")


# ========================
# GESTURE CLASSIFICATION
# ========================
def detectGesture(lmList, fingers, state):
    """
    Returns (gesture_label, distances) where distances =
    (d_index, d_middle, d_ring, d_pinky, d_mid_ring) — for debug display.

    Exactly ONE gesture label is returned per call, in this explicit priority
    order (first match wins):
      1. Active DRAG (must resolve before anything else)
      2. Volume LOCK already engaged -> keep firing Volume Up/Down
         Volume PENDING (forming) -> suppress everything else this frame
      3. Mute
      4. Scroll
      5. Confirmed pinch classification (LEFT/RIGHT/DOUBLE click, DRAG start)
      6. Screenshot
      7. COPY / CUT / PASTE
      8. Play/Pause
      9. Zoom (middle+ring spread)
      10. Window switch (Alt+Tab)
      11. Move
      12. NONE
    """
    now = time.time()

    thumb_tip = (lmList[4][1], lmList[4][2])
    index_tip = (lmList[8][1], lmList[8][2])
    middle_tip = (lmList[12][1], lmList[12][2])
    ring_tip = (lmList[16][1], lmList[16][2])
    pinky_tip = (lmList[20][1], lmList[20][2])

    d_index = distance(thumb_tip, index_tip)
    d_middle = distance(thumb_tip, middle_tip)
    d_ring = distance(thumb_tip, ring_tip)
    d_pinky = distance(thumb_tip, pinky_tip)
    d_mid_ring = distance(middle_tip, ring_tip)
    dists = (d_index, d_middle, d_ring, d_pinky, d_mid_ring)

    f = fingers[1:]  # [index, middle, ring, pinky] — thumb excluded from shape checks

    # ---- shared geometry, computed once ----
    thumb_direction = detectThumbDirection(lmList)
    thumb_folded = isThumbFolded(lmList)
    state["last_thumb_direction"] = thumb_direction

    # ---- DETECT COPY/CUT/PASTE SHAPES ----
    # COPY: Only Pinky up -> [thumb, index, middle, ring, pinky] = [0, 0, 0, 0, 1]
    # CUT: Ring + Pinky up -> [thumb, index, middle, ring, pinky] = [0, 0, 0, 1, 1]
    # PASTE: Index + Middle + Ring + Pinky up -> [thumb, index, middle, ring, pinky] = [0, 1, 1, 1, 1]
    
    copy_shape = (fingers == [0, 0, 0, 0, 1])  # Only pinky up
    cut_shape = (fingers == [0, 0, 0, 1, 1])   # Ring + Pinky up
    paste_shape = (fingers == [0, 1, 1, 1, 1]) # Index + Middle + Ring + Pinky up
    
    # Check if any COPY/CUT/PASTE gesture is active
    is_copy_paste_gesture = copy_shape or cut_shape or paste_shape

    # A "protected" pose: while the hand clearly matches one of these shapes,
    # pinch classification is skipped entirely. This is the core fix for both
    # scroll and fist — previously classifyPinch() ran unconditionally every
    # frame, so a closed fist (or ☝️/✌️) could still have its thumb "win" the
    # pinch classification and get reported as a click/drag instead.
    scroll_shape = f in ([1, 0, 0, 0], [1, 1, 0, 0])
    fist_shape = (f == [0, 0, 0, 0])

    if scroll_shape or fist_shape or is_copy_paste_gesture:
        pinch = None
        if scroll_shape:
            if DEBUG and not state["scroll_lock_active"]:
                print("SCROLL POSE LOCKED - PINCH DETECTION DISABLED")
            state["scroll_lock_active"] = True
        else:
            state["scroll_lock_active"] = False
    else:
        pinch = classifyPinch(d_index, d_middle, d_ring, d_pinky)
        state["scroll_lock_active"] = False

    state["last_pinch"] = pinch                      # debug: "Closest pinch: ..."

    # ---- bookkeeping: confirmation counters (always updated, regardless of
    #      what ultimately gets returned this frame) ----

    # ---- VOLUME pending/lock state machine ----
    # Use exact finger matching for volume detection
    if is_copy_paste_gesture:
        # Volume detection is completely disabled for COPY/CUT/PASTE
        volume_candidate = None
        state["volume_candidate"] = None
        # Don't modify volume lock state during COPY/CUT/PASTE
    else:
        # Check for exact volume pose: all four non-thumb fingers must be folded
        four_fingers_folded = (f == [0, 0, 0, 0])
        
        volume_candidate = None
        if four_fingers_folded and thumb_direction == "UP":
            volume_candidate = "UP"
        elif four_fingers_folded and thumb_direction == "DOWN":
            volume_candidate = "DOWN"
        
        state["volume_candidate"] = volume_candidate  # debug: "Volume candidate: ..."

        # Release bookkeeping — only relevant while a lock is already engaged.
        if state["volume_lock"] is not None:
            if volume_candidate == state["volume_lock"]:
                state["volume_release_count"] = 0
            else:
                state["volume_release_count"] += 1
                if state["volume_release_count"] >= VOLUME_RELEASE_FRAMES:
                    if DEBUG:
                        print(f"VOLUME {state['volume_lock']} RELEASED")
                    state["volume_lock"] = None
                    state["volume_release_count"] = 0

        # Pending/confirm bookkeeping — only while nothing is currently locked.
        if state["volume_lock"] is None:
            if volume_candidate is not None:
                if state["volume_pending"] == volume_candidate:
                    state["volume_pending_count"] += 1
                else:
                    state["volume_pending"] = volume_candidate
                    state["volume_pending_count"] = 1
                    if DEBUG:
                        print(f"VOLUME {volume_candidate} PENDING")

                if state["volume_pending_count"] >= VOLUME_CONFIRM_FRAMES:
                    state["volume_lock"] = volume_candidate
                    state["volume_pending"] = None
                    state["volume_pending_count"] = 0
                    state["volume_release_count"] = 0
                    if DEBUG:
                        print(f"VOLUME {volume_candidate} CONFIRMED")
            else:
                state["volume_pending"] = None
                state["volume_pending_count"] = 0

    # Scroll is allowed only when the thumb is genuinely neutral AND no volume
    # gesture is forming or locked AND not in COPY/CUT/PASTE.
    scroll_allowed = (
        thumb_direction == "NEUTRAL"
        and state["volume_lock"] is None
        and state["volume_pending"] is None
        and not is_copy_paste_gesture  # Disable scroll during COPY/CUT/PASTE
    )
    state["scroll_allowed"] = scroll_allowed  # debug: "Scroll allowed: ..."

    # Right click / double click — only counts up while THAT specific pinch
    # is the confirmed classification (not just "close enough").
    if pinch == "middle":
        state["right_click_frame_count"] += 1
    else:
        state["right_click_frame_count"] = 0

    if pinch == "ring":
        state["double_click_frame_count"] += 1
    else:
        state["double_click_frame_count"] = 0

    # ---- SIMPLIFIED MUTE: FIST / MUTE ----
    # Now works like a button: press on fist close, release on fist open
    # No more complex release frame counting
    
    if fist_shape:
        state["fist_confirm_count"] += 1
    else:
        state["fist_confirm_count"] = 0
        # SIMPLIFIED: Immediately re-arm when fist is released
        if state["mute_latched"]:
            state["mute_latched"] = False
            if DEBUG:
                print("MUTE RE-ARMED (fist released)")

    confirmed_fist = state["fist_confirm_count"] >= MUTE_CONFIRM_FRAMES

    # ---- COPY / CUT / PASTE detection ----
    # Update COPY confirmation counter
    if copy_shape:
        state["copy_confirm_count"] += 1
        if DEBUG and state["copy_confirm_count"] == COPY_CONFIRM_FRAMES:
            print("COPY GESTURE DETECTED")
    else:
        state["copy_confirm_count"] = 0

    # Update CUT confirmation counter
    if cut_shape:
        state["cut_confirm_count"] += 1
        if DEBUG and state["cut_confirm_count"] == CUT_CONFIRM_FRAMES:
            print("CUT GESTURE DETECTED")
    else:
        state["cut_confirm_count"] = 0

    # Update PASTE confirmation counter
    if paste_shape:
        state["paste_confirm_count"] += 1
        if DEBUG and state["paste_confirm_count"] >= PASTE_CONFIRM_FRAMES:
            print("PASTE GESTURE DETECTED")
    else:
        state["paste_confirm_count"] = 0

    # COPY / CUT / PASTE release tracking
    # Reset latches when pose is released
    if not copy_shape and state["copy_latched"]:
        state["copy_release_count"] += 1
        if state["copy_release_count"] >= 3:  # 3 frames to re-arm
            state["copy_latched"] = False
            state["copy_release_count"] = 0
            if DEBUG:
                print("COPY RE-ARMED")
    elif copy_shape:
        state["copy_release_count"] = 0

    if not cut_shape and state["cut_latched"]:
        state["cut_release_count"] += 1
        if state["cut_release_count"] >= 3:
            state["cut_latched"] = False
            state["cut_release_count"] = 0
            if DEBUG:
                print("CUT RE-ARMED")
    elif cut_shape:
        state["cut_release_count"] = 0

    if not paste_shape and state["paste_latched"]:
        state["paste_release_count"] += 1
        if state["paste_release_count"] >= 3:
            state["paste_latched"] = False
            state["paste_release_count"] = 0
            if DEBUG:
                print("PASTE RE-ARMED")
    elif paste_shape:
        state["paste_release_count"] = 0

    # ---- SCREENSHOT: four fingers open, thumb folded across the palm ----
    # **FIX: Skip screenshot detection during COPY/CUT/PASTE**
    if not is_copy_paste_gesture:
        screenshot_shape = (f == [1, 1, 1, 1]) and thumb_folded
        if screenshot_shape:
            state["screenshot_frame_count"] += 1
            if DEBUG and state["screenshot_frame_count"] == SCREENSHOT_STABLE_FRAMES:
                print("SCREENSHOT GESTURE DETECTED")
        else:
            state["screenshot_frame_count"] = 0
    else:
        # Reset screenshot frame count during COPY/CUT/PASTE to prevent accidental triggers
        state["screenshot_frame_count"] = 0

    # Play/Pause
    playpause_shape = (f == [1, 0, 0, 1])
    if playpause_shape:
        state["playpause_frame_count"] += 1
    else:
        state["playpause_frame_count"] = 0

    # Zoom (middle+ring spread, tracked over a short time window)
    zoom_candidate = (f == [0, 1, 1, 0])
    if zoom_candidate:
        state["zoom_history"].append((now, d_mid_ring))
        while state["zoom_history"] and now - state["zoom_history"][0][0] > ZOOM_HISTORY_WINDOW:
            state["zoom_history"].popleft()
    else:
        state["zoom_history"].clear()

    # ---- NEW: One-hand pinch zoom using thumb and index ----
    # This is a SEPARATE feature that doesn't interfere with any existing gestures
    # It uses the thumb-index distance normalized by palm width
    # Only activates when thumb and index are the ONLY fingers extended
    
    # Calculate palm width for normalization (landmark 5 = index MCP, landmark 17 = pinky MCP)
    palm_width = distance(
        (lmList[5][1], lmList[5][2]),
        (lmList[17][1], lmList[17][2])
    )
    
    # Normalize thumb-index distance
    if palm_width > 0:
        pinch_zoom_ratio = d_index / palm_width
    else:
        pinch_zoom_ratio = 0.0
    
    # Clamp ratio
    pinch_zoom_ratio = max(PINCH_ZOOM_MIN_RATIO, min(PINCH_ZOOM_MAX_RATIO, pinch_zoom_ratio))
    
    # Detect pinch-zoom gesture: ONLY thumb and index extended (all others folded)
    # Pattern: [thumb, index, middle, ring, pinky] = [1, 1, 0, 0, 0]
    is_pinch_zoom_gesture = fingers == [1, 1, 0, 0, 0]
    
    # Initialize or update pinch zoom state
    if is_pinch_zoom_gesture:
        if not state["pinch_zoom_initialized"]:
            state["pinch_zoom_prev_ratio"] = pinch_zoom_ratio
            state["pinch_zoom_initialized"] = True
        
        # Calculate change in normalized distance
        ratio_change = pinch_zoom_ratio - state["pinch_zoom_prev_ratio"]
        
        # Only trigger if change exceeds deadzone
        if abs(ratio_change) > PINCH_ZOOM_DEADZONE:
            if now - state["last_pinch_zoom_time"] > PINCH_ZOOM_MIN_INTERVAL:
                # Calculate zoom steps based on ratio change
                zoom_steps = int(abs(ratio_change) * PINCH_ZOOM_STEP_SCALE)
                zoom_steps = max(1, min(zoom_steps, 8))  # Limit to 1-8 steps per trigger
                
                if ratio_change > 0:
                    # Fingers spreading apart -> Zoom In
                    for _ in range(zoom_steps):
                        pyautogui.hotkey("ctrl", "+")
                    if DEBUG:
                        print(f"PINCH ZOOM IN (steps: {zoom_steps})")
                else:
                    # Fingers coming together -> Zoom Out
                    for _ in range(zoom_steps):
                        pyautogui.hotkey("ctrl", "-")
                    if DEBUG:
                        print(f"PINCH ZOOM OUT (steps: {zoom_steps})")
                
                state["last_pinch_zoom_time"] = now
                state["pinch_zoom_prev_ratio"] = pinch_zoom_ratio
        
        # Smoothly track the ratio
        state["pinch_zoom_prev_ratio"] = state["pinch_zoom_prev_ratio"] * 0.6 + pinch_zoom_ratio * 0.4
        
    else:
        # Reset zoom tracking when gesture ends
        if state["pinch_zoom_initialized"]:
            state["pinch_zoom_initialized"] = False
            state["pinch_zoom_prev_ratio"] = 0.0

    # Window switch (three-finger horizontal swipe, tracked over a short time window)
    swipe_candidate = (f == [1, 1, 1, 0])
    if swipe_candidate:
        state["swipe_history"].append((now, index_tip[0]))
        while state["swipe_history"] and now - state["swipe_history"][0][0] > SWIPE_HISTORY_WINDOW:
            state["swipe_history"].popleft()
    else:
        state["swipe_history"].clear()
        state["swipe_fired"] = False

    # ============================================================
    # 1. Active DRAG must resolve before anything else is considered
    # ============================================================
    if state["is_dragging"]:
        if d_pinky > DRAG_RELEASE_DIST:
            return "DRAG_END", dists
        return "DRAGGING", dists

    # ============================================================
    # 2. Volume — a LOCKED gesture keeps firing; a PENDING one suppresses
    #    absolutely everything else (scroll, mute, clicks included) until
    #    it either confirms into a lock or the candidate disappears.
    # ============================================================
    if not is_copy_paste_gesture:
        if state["volume_lock"] is not None:
            return ("VOLUME_UP" if state["volume_lock"] == "UP" else "VOLUME_DOWN"), dists

        if state["volume_pending"] is not None:
            return "NONE", dists

    # ============================================================
    # 3. SIMPLIFIED MUTE — Fist
    # ============================================================
    if fist_shape:
        if confirmed_fist:
            return "MUTE", dists
        return "NONE", dists

    # ============================================================
    # 4. Scroll — requires scroll_allowed (neutral thumb, no volume
    #    candidate forming or locked, AND not in COPY/CUT/PASTE)
    # ============================================================
    if f == [1, 0, 0, 0] and scroll_allowed:
        return "SCROLL_UP", dists

    if f == [1, 1, 0, 0] and scroll_allowed:
        return "SCROLL_DOWN", dists

    # ============================================================
    # 5. Confirmed pinch classification
    # ============================================================
    if pinch == "index":
        # Left click stays immediately responsive (no confirmation-frame
        # delay) — it already requires a clean, unambiguous pinch via
        # classifyPinch()'s separation check, and handleGesture()'s edge
        # trigger requires a full release before it can fire again.
        return "LEFT_CLICK", dists

    if pinch == "middle":
        if state["right_click_frame_count"] >= RIGHT_CLICK_CONFIRM_FRAMES:
            return "RIGHT_CLICK", dists
        return "NONE", dists  # still confirming — do not fall through to anything else

    if pinch == "ring":
        if state["double_click_frame_count"] >= DOUBLE_CLICK_CONFIRM_FRAMES:
            return "DOUBLE_CLICK", dists
        return "NONE", dists

    if pinch == "pinky":
        return "DRAG_START", dists  # responsive on purpose — see LEFT_CLICK note above

    # ============================================================
    # 6. Screenshot (open hand, thumb folded — no pinch involved)
    # ============================================================
    if not is_copy_paste_gesture:
        if screenshot_shape and state["screenshot_frame_count"] >= SCREENSHOT_STABLE_FRAMES:
            return "SCREENSHOT", dists
        # If screenshot not confirmed yet, return NONE to allow other gestures
        if screenshot_shape:
            return "NONE", dists

    # ============================================================
    # 7. COPY / CUT / PASTE (with proper confirmation)
    # ============================================================
    # COPY: Only Pinky up (unchanged, requires 3 frames)
    if copy_shape and state["copy_confirm_count"] >= COPY_CONFIRM_FRAMES:
        return "COPY", dists
    
    # CUT: Ring + Pinky up (unchanged, requires 3 frames)
    if cut_shape and state["cut_confirm_count"] >= CUT_CONFIRM_FRAMES:
        return "CUT", dists
    
    # PASTE: Index + Middle + Ring + Pinky up (requires only 1 frame now)
    if paste_shape and state["paste_confirm_count"] >= PASTE_CONFIRM_FRAMES:
        return "PASTE", dists

    # ============================================================
    # 8. Play/Pause
    # ============================================================
    if playpause_shape:
        if state["playpause_frame_count"] >= PLAYPAUSE_CONFIRM_FRAMES:
            return "PLAY_PAUSE", dists
        return "NONE", dists

    # ============================================================
    # 9. Zoom (middle+ring spread)
    # ============================================================
    if zoom_candidate:
        if len(state["zoom_history"]) >= 2:
            delta = state["zoom_history"][-1][1] - state["zoom_history"][0][1]
            if delta > ZOOM_MIN_DELTA:
                return "ZOOM_IN", dists
            if delta < -ZOOM_MIN_DELTA:
                return "ZOOM_OUT", dists
        return "NONE", dists

    # ============================================================
    # 10. Window switch (Alt+Tab)
    # ============================================================
    if swipe_candidate:
        if len(state["swipe_history"]) >= 2 and not state["swipe_fired"]:
            displacement = state["swipe_history"][-1][1] - state["swipe_history"][0][1]
            if abs(displacement) > SWIPE_MIN_DISTANCE:
                state["swipe_fired"] = True
                return "WINDOW_SWITCH", dists
        return "NONE", dists

    # ============================================================
    # 11. Move — open hand with the thumb NOT folded (distinguishes it from Screenshot)
    # ============================================================
    if f == [1, 1, 1, 1] and not thumb_folded:
        return "MOVE", dists

    # ============================================================
    # 12. Nothing matched
    # ============================================================
    return "NONE", dists


# ========================
# IMPROVED CURSOR MOVEMENT - Smooth & Glitch-Free
# ========================
def moveCursor(lmList, state):
    """
    Advanced cursor smoothing combining multiple techniques:
    1. Exponential moving average (EMA)
    2. History averaging
    3. Jitter threshold
    4. Movement prediction
    5. Velocity-based adaptive smoothing
    """
    x1, y1 = lmList[8][1], lmList[8][2]  # index fingertip drives the cursor

    # X range reversed (SCREEN_W -> 0) so cursor direction matches hand direction
    screenX = np.interp(x1, (0, CAM_W), (SCREEN_W, 0))
    screenY = np.interp(y1, (0, CAM_H), (0, SCREEN_H))

    # Calculate movement since last frame
    if state["prevX"] != 0 or state["prevY"] != 0:
        dx = screenX - state["prevX"]
        dy = screenY - state["prevY"]
        movement = math.hypot(dx, dy)
        
        # Jitter threshold - ignore tiny movements
        if movement < JITTER_THRESHOLD:
            # Don't update position - keep cursor stable
            return
        
        # Adaptive smoothing based on movement speed
        # Fast movements = less smoothing (more responsive)
        # Slow movements = more smoothing (less jitter)
        speed_factor = min(movement / 50.0, 1.0)
        adaptive_smooth = SMOOTHENING * (1.0 - speed_factor * 0.4)
        adaptive_smooth = max(3, adaptive_smooth)  # Clamp to reasonable range
    else:
        adaptive_smooth = SMOOTHENING

    # Store in history for averaging
    state["pos_history"].append((screenX, screenY))
    if len(state["pos_history"]) > SMOOTHING_HISTORY:
        state["pos_history"].popleft()

    # Calculate moving average from history
    if len(state["pos_history"]) >= 3:
        avg_x = sum(p[0] for p in state["pos_history"]) / len(state["pos_history"])
        avg_y = sum(p[1] for p in state["pos_history"]) / len(state["pos_history"])
    else:
        avg_x = screenX
        avg_y = screenY

    # Exponential moving average (EMA) - smooths current position toward average
    if state["prevX"] != 0 or state["prevY"] != 0:
        # Blend: 60% current average, 40% previous position
        smoothX = avg_x * SMOOTHING_WEIGHT + state["prevX"] * (1 - SMOOTHING_WEIGHT)
        smoothY = avg_y * SMOOTHING_WEIGHT + state["prevY"] * (1 - SMOOTHING_WEIGHT)
    else:
        smoothX = avg_x
        smoothY = avg_y

    # Apply slight prediction for smoother motion during consistent movement
    if len(state["pos_history"]) >= 3 and movement > 10:
        # Predict next position based on recent movement trend
        pred_x = smoothX + (smoothX - state["prevX"]) * PREDICTION_WEIGHT
        pred_y = smoothY + (smoothY - state["prevY"]) * PREDICTION_WEIGHT
        smoothX = smoothX * (1 - PREDICTION_WEIGHT) + pred_x * PREDICTION_WEIGHT
        smoothY = smoothY * (1 - PREDICTION_WEIGHT) + pred_y * PREDICTION_WEIGHT

    # Clamp to screen bounds
    currX = float(np.clip(smoothX, 0, SCREEN_W - 1))
    currY = float(np.clip(smoothY, 0, SCREEN_H - 1))

    # Only move if the change is significant (reduces micro-movements)
    if math.hypot(currX - state["prevX"], currY - state["prevY"]) > JITTER_THRESHOLD:
        pyautogui.moveTo(currX, currY)
        state["prevX"], state["prevY"] = currX, currY


# ========================
# GESTURE -> ACTION
# ========================
def handleGesture(gesture, lmList, state):
    now = time.time()
    entered = (gesture != state["previous_gesture"])  # true on the first frame of this gesture

    if gesture == "MOVE":
        moveCursor(lmList, state)

    elif gesture == "DRAG_START":
        pyautogui.mouseDown()
        state["is_dragging"] = True
        moveCursor(lmList, state)
        print("DRAG STARTED")

    elif gesture == "DRAGGING":
        moveCursor(lmList, state)

    elif gesture == "DRAG_END":
        pyautogui.mouseUp()
        state["is_dragging"] = False
        print("DRAG ENDED")

    elif gesture == "LEFT_CLICK":
        if entered and now - state["last_click_time"] > CLICK_COOLDOWN:
            pyautogui.click()
            state["last_click_time"] = now
            print("LEFT CLICK")

    elif gesture == "RIGHT_CLICK":
        if entered and now - state["last_right_click_time"] > RIGHT_CLICK_COOLDOWN:
            pyautogui.rightClick()
            state["last_right_click_time"] = now
            print("RIGHT CLICK")

    elif gesture == "DOUBLE_CLICK":
        if entered and now - state["last_double_click_time"] > DOUBLE_CLICK_COOLDOWN:
            pyautogui.doubleClick()
            state["last_double_click_time"] = now
            print("DOUBLE CLICK")

    # ============================================================
    # SIMPLIFIED MUTE — Works like a button
    # ============================================================
    elif gesture == "MUTE":
        # Simple edge trigger: execute once when fist becomes confirmed
        if not state["mute_latched"]:
            pyautogui.press("volumemute")
            state["mute_latched"] = True
            state["last_mute_time"] = now
            print("MUTE / UNMUTE TOGGLED")
        else:
            if DEBUG:
                print("MUTE LOCKED - Holding fist")

    elif gesture == "SCROLL_UP":
        if now - state["last_scroll_time"] > SCROLL_COOLDOWN:
            windows_scroll(SCROLL_NOTCHES)
            state["last_scroll_time"] = now
            if DEBUG:
                print("EXECUTING SCROLL: UP")

    elif gesture == "SCROLL_DOWN":
        if now - state["last_scroll_time"] > SCROLL_COOLDOWN:
            windows_scroll(-SCROLL_NOTCHES)
            state["last_scroll_time"] = now
            if DEBUG:
                print("EXECUTING SCROLL: DOWN")

    elif gesture == "VOLUME_UP":
        if now - state["last_volume_time"] > VOLUME_COOLDOWN:
            pyautogui.press("volumeup")
            state["last_volume_time"] = now
            if DEBUG:
                print("EXECUTING: VOLUME UP")

    elif gesture == "VOLUME_DOWN":
        if now - state["last_volume_time"] > VOLUME_COOLDOWN:
            pyautogui.press("volumedown")
            state["last_volume_time"] = now
            if DEBUG:
                print("EXECUTING: VOLUME DOWN")

    elif gesture == "ZOOM_IN":
        if now - state["last_zoom_time"] > ZOOM_COOLDOWN:
            pyautogui.hotkey("ctrl", "+")
            state["last_zoom_time"] = now
            print("ZOOM IN")

    elif gesture == "ZOOM_OUT":
        if now - state["last_zoom_time"] > ZOOM_COOLDOWN:
            pyautogui.hotkey("ctrl", "-")
            state["last_zoom_time"] = now
            print("ZOOM OUT")

    elif gesture == "PLAY_PAUSE":
        if entered:
            pyautogui.press("playpause")
            print("PLAY/PAUSE TOGGLED")

    elif gesture == "SCREENSHOT":
        if entered and now - state["last_screenshot_time"] > SCREENSHOT_COOLDOWN:
            success, filepath = takeScreenshot()
            state["last_screenshot_time"] = now
            if success:
                state["screenshot_flash_until"] = now + 1.2
                showScreenshotNotification(filepath)

    # ============================================================
    # COPY / CUT / PASTE
    # ============================================================
    elif gesture == "COPY":
        # Edge-triggered: only execute when first detected, not while held
        if not state["copy_latched"] and state["copy_confirm_count"] >= COPY_CONFIRM_FRAMES:
            if now - state["last_copy_time"] > COPY_COOLDOWN:
                pyautogui.hotkey("ctrl", "c")
                state["last_copy_time"] = now
                state["copy_latched"] = True
                print("COPY EXECUTED")
        else:
            if DEBUG and state["copy_latched"]:
                print("COPY LOCKED - Holding pose")

    elif gesture == "CUT":
        if not state["cut_latched"] and state["cut_confirm_count"] >= CUT_CONFIRM_FRAMES:
            if now - state["last_cut_time"] > CUT_COOLDOWN:
                pyautogui.hotkey("ctrl", "x")
                state["last_cut_time"] = now
                state["cut_latched"] = True
                print("CUT EXECUTED")
        else:
            if DEBUG and state["cut_latched"]:
                print("CUT LOCKED - Holding pose")

    elif gesture == "PASTE":
        if not state["paste_latched"] and state["paste_confirm_count"] >= PASTE_CONFIRM_FRAMES:
            if now - state["last_paste_time"] > PASTE_COOLDOWN:
                pyautogui.hotkey("ctrl", "v")
                state["last_paste_time"] = now
                state["paste_latched"] = True
                print("PASTE EXECUTED")
        else:
            if DEBUG and state["paste_latched"]:
                print("PASTE LOCKED - Holding pose")

    elif gesture == "WINDOW_SWITCH":
        if entered:
            pyautogui.hotkey("alt", "tab")
            print("ALT+TAB")

    state["previous_gesture"] = gesture


# ========================
# ON-SCREEN STATUS OVERLAY - CLEAN UI
# ========================
def drawStatus(img, gesture, fps, fingers, dists, state):
    """
    Clean overlay showing ONLY:
    1. FPS
    2. Gesture
    3. Fingers
    
    All other status text has been removed for a clean interface.
    """
    # FPS - Top left
    cv2.putText(
        img, f"FPS: {int(fps)}", (20, 40),
        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
    )

    # Current Gesture - Below FPS
    label = gesture if gesture != "NONE" else "IDLE"
    cv2.putText(
        img, f"Gesture: {label}", (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2
    )

    # Detected Fingers - Below Gesture
    cv2.putText(
        img, f"Fingers: {fingers}", (20, 130),
        cv2.FONT_HERSHEY_PLAIN, 1.4, (200, 200, 200), 2
    )


# ========================
# MAIN LOOP
# ========================
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera failed to open")
        return

    detector = HandDetector()
    pTime = 0

    # All persistent, per-run state lives in this single dict so detectGesture(),
    # handleGesture(), and moveCursor() can read/update it without globals.
    state = {
        "prevX": 0, "prevY": 0,
        "is_dragging": False,
        "previous_gesture": "NONE",
        "last_click_time": 0,
        "last_right_click_time": 0,
        "last_double_click_time": 0,
        "last_scroll_time": 0,
        "last_volume_time": 0,
        "last_zoom_time": 0,
        "last_screenshot_time": 0,
        "last_mute_time": 0,
        "zoom_history": deque(),
        "swipe_history": deque(),
        "swipe_fired": False,
        "screenshot_frame_count": 0,
        "screenshot_flash_until": 0,
        "last_thumb_direction": "NEUTRAL",
        "volume_candidate": None,
        "volume_pending": None,
        "volume_pending_count": 0,
        "volume_lock": None,
        "volume_release_count": 0,
        "scroll_allowed": True,
        "right_click_frame_count": 0,
        "double_click_frame_count": 0,
        "fist_confirm_count": 0,
        "mute_latched": False,
        "playpause_frame_count": 0,
        "last_pinch": None,
        "scroll_lock_active": False,
        # COPY / CUT / PASTE state
        "last_copy_time": 0,
        "last_cut_time": 0,
        "last_paste_time": 0,
        "copy_confirm_count": 0,
        "cut_confirm_count": 0,
        "paste_confirm_count": 0,
        "copy_latched": False,
        "cut_latched": False,
        "paste_latched": False,
        "copy_release_count": 0,
        "cut_release_count": 0,
        "paste_release_count": 0,
        # Pinch Zoom state
        "last_pinch_zoom_time": 0,
        "pinch_zoom_initialized": False,
        "pinch_zoom_prev_ratio": 0.0,
        # Cursor smoothing state
        "pos_history": deque(maxlen=SMOOTHING_HISTORY),
    }

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        gesture = "NONE"
        fingers = [0, 0, 0, 0, 0]
        dists = (0, 0, 0, 0, 0)

        if lmList:
            fingers = fingersUp(lmList)
            gesture, dists = detectGesture(lmList, fingers, state)
            handleGesture(gesture, lmList, state)
        else:
            # Safety: if the hand disappears mid-drag, release the mouse button
            # so it can never get stuck held down.
            if state["is_dragging"]:
                pyautogui.mouseUp()
                state["is_dragging"] = False
                print("DRAG ENDED (hand lost)")
            state["previous_gesture"] = "NONE"
            state["zoom_history"].clear()
            state["swipe_history"].clear()
            state["swipe_fired"] = False
            state["screenshot_frame_count"] = 0
            state["volume_candidate"] = None
            state["volume_pending"] = None
            state["volume_pending_count"] = 0
            state["volume_lock"] = None
            state["volume_release_count"] = 0
            state["right_click_frame_count"] = 0
            state["double_click_frame_count"] = 0
            state["fist_confirm_count"] = 0
            # SIMPLIFIED MUTE: Re-arm on hand lost
            if state["mute_latched"]:
                state["mute_latched"] = False
                if DEBUG:
                    print("MUTE RE-ARMED (hand lost)")
            state["playpause_frame_count"] = 0
            state["last_pinch"] = None
            # Reset COPY/CUT/PASTE confirm counts when hand lost
            state["copy_confirm_count"] = 0
            state["cut_confirm_count"] = 0
            state["paste_confirm_count"] = 0
            # Reset pinch zoom state
            state["pinch_zoom_initialized"] = False
            state["pinch_zoom_prev_ratio"] = 0.0
            # Reset cursor history on hand loss
            state["pos_history"].clear()

        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime else 0
        pTime = cTime

        drawStatus(img, gesture, fps, fingers, dists, state)

        cv2.imshow("SilentFlow – Touchless Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if state["is_dragging"]:
                pyautogui.mouseUp()
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import cv2
import mediapipe as mp
import time
import os
import math
import ctypes
from datetime import datetime
from collections import deque

import numpy as np
import pyautogui

# winotify is optional — if it's not installed, screenshots still save fine,
# they just won't show a toast notification. Nothing else in the program
# depends on this.
try:
    from winotify import Notification
    WINOTIFY_AVAILABLE = True
except ImportError:
    WINOTIFY_AVAILABLE = False

# ========================
# CONFIG
# ========================
SCREEN_W, SCREEN_H = pyautogui.size()
pyautogui.FAILSAFE = False

DEBUG = False   # Set to False to keep UI clean - only shows FPS, Gesture, Fingers

CAM_W, CAM_H = 640, 480

# ========================
# IMPROVED CURSOR SMOOTHING
# ========================
# Multiple smoothing techniques combined for buttery-smooth cursor movement
SMOOTHENING = 5              # Higher = smoother but more lag (3-8 recommended)
SMOOTHING_HISTORY = 8        # Number of recent positions to average
SMOOTHING_WEIGHT = 0.6       # Weight for exponential moving average (0.3-0.7)
JITTER_THRESHOLD = 3.0       # Minimum movement to register (pixels)
PREDICTION_WEIGHT = 0.15     # Slight movement prediction (0.05-0.25)

# --- pinch-family thresholds (all in camera pixels, thumb-tip to fingertip) ---
CLICK_DIST = 40          # thumb + index  -> left click
RIGHT_CLICK_DIST = 40    # thumb + middle -> right click
DOUBLE_CLICK_DIST = 40   # thumb + ring   -> double click
DRAG_START_DIST = 40     # thumb + pinky  -> start drag
DRAG_RELEASE_DIST = 55   # bigger than DRAG_START_DIST on purpose: this hysteresis gap
                          # stops the drag flickering on/off right at the boundary
PINCH_SEPARATION = 12    # the winning fingertip must be at least this much closer to the
                          # thumb than every other fingertip, or the pinch is rejected as
                          # ambiguous (returns None) instead of guessing which one you meant

# --- thumb-direction geometry (Volume Up/Down) ---
THUMB_EXTENSION_MIN = 35     # thumb tip must be at least this far from its MCP joint
                              # to count as "extended" rather than curled into the fist
THUMB_VERTICAL_RATIO = 1.15  # vertical component of the thumb vector must exceed the
                              # horizontal component by this factor to call it "up"/"down"
                              # rather than "sideways" (ambiguous -> NEUTRAL)

# --- volume pending/lock state machine ---
VOLUME_CONFIRM_FRAMES = 3    # consecutive frames a candidate must hold before it LOCKS in
VOLUME_RELEASE_FRAMES = 5    # consecutive non-matching frames needed to release the lock

# --- thumb-folded geometry (Screenshot vs. Move) ---
THUMB_FOLDED_MAX = 40        # thumb tip within this distance of its MCP joint counts as
                              # "tucked across the palm" rather than openly extended

# --- zoom (middle+ring fingertip distance, tracked over time) ---
ZOOM_HISTORY_WINDOW = 0.35   # seconds of recent samples considered
ZOOM_MIN_DELTA = 18          # px change within the window needed to count as intentional
ZOOM_COOLDOWN = 0.35         # seconds between successive zoom hotkey fires

# --- NEW: One-hand pinch zoom using thumb-index distance ---
PINCH_ZOOM_DEADZONE = 0.03       # Minimum normalized distance change to trigger zoom
PINCH_ZOOM_MIN_INTERVAL = 0.08   # Seconds between zoom updates (throttle)
PINCH_ZOOM_STEP_SCALE = 8.0      # Steps per unit of normalized distance change
PINCH_ZOOM_MIN_RATIO = 0.05      # Minimum thumb-index distance ratio
PINCH_ZOOM_MAX_RATIO = 0.80      # Maximum thumb-index distance ratio

# --- Alt+Tab (three-finger horizontal swipe) ---
SWIPE_HISTORY_WINDOW = 0.5   # seconds — a deliberate swipe should complete within this
SWIPE_MIN_DISTANCE = 90      # px of horizontal index-tip movement needed to confirm a swipe

# --- confirmation frames (require the pose to hold steady before firing) ---
RIGHT_CLICK_CONFIRM_FRAMES = 2
DOUBLE_CLICK_CONFIRM_FRAMES = 2
SCREENSHOT_STABLE_FRAMES = 3
MUTE_CONFIRM_FRAMES = 2      # SIMPLIFIED: Only need confirmation frames
PLAYPAUSE_CONFIRM_FRAMES = 2

# --- COPY / CUT / PASTE confirmation frames ---
COPY_CONFIRM_FRAMES = 3     # require 3 stable frames before executing
CUT_CONFIRM_FRAMES = 3
PASTE_CONFIRM_FRAMES = 1    # OPTIMIZED: Require only 1 frame for Paste (more responsive)

# --- cooldowns (seconds) — throttle continuous/repeatable actions, guard one-shot ones ---
CLICK_COOLDOWN = 0.3
RIGHT_CLICK_COOLDOWN = 0.4
DOUBLE_CLICK_COOLDOWN = 0.5
SCROLL_COOLDOWN = 0.12
SCROLL_NOTCHES = 1
VOLUME_COOLDOWN = 0.3
SCREENSHOT_COOLDOWN = 1.5
MUTE_COOLDOWN = 0.15   # Kept as a safety net, but mute_latched now handles the edge trigger
COPY_COOLDOWN = 0.5    # cooldown for copy/cut/paste to prevent accidental repeats
CUT_COOLDOWN = 0.5
PASTE_COOLDOWN = 0.3   # OPTIMIZED: Slightly reduced cooldown for Paste


# ========================
# WINDOWS WHEEL EVENT (real hardware-level scroll)
# ========================
_MOUSEEVENTF_WHEEL = 0x0800
_WHEEL_DELTA = 120

def windows_scroll(notches):
    """notches: positive = scroll up, negative = scroll down."""
    ctypes.windll.user32.mouse_event(_MOUSEEVENTF_WHEEL, 0, 0, int(notches * _WHEEL_DELTA), 0)


# ========================
# HAND DETECTOR
# ======================== Virtual Mouse itself — use your hands! Turn
#the pages the way you'd control a ges Virtual Mouse itself — use your hands! Turn
#the pages the way you'd control a ges Virtual Mouse itself — use your hands! Turn
#the pages the way you'd control a ges
class HandDetector:
    def __init__(self, detectionCon=0.7, trackCon=0.7):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    img, handLms, self.mpHands.HAND_CONNECTIONS
                )
        return img

    def findPosition(self, img):
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            h, w, _ = img.shape
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList


# ========================
# LOW-LEVEL HELPERS
# ========================
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def fingersUp(lmList):
    """
    Returns [thumb, index, middle, ring, pinky] as 1 (extended) / 0 (curled).
    NOTE: fingers[0] (thumb) is x-coordinate based and flips with hand rotation —
    kept only for debug display. No gesture uses it as a gate; thumb-based
    gestures use detectThumbDirection()/isThumbFolded() (geometry) instead.
    """
    fingers = []
    fingers.append(lmList[4][1] > lmList[3][1])  # thumb (debug display only)

    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(lmList[tip][2] < lmList[tip - 2][2])

    return [int(f) for f in fingers]


def detectThumbDirection(lmList):
    """
    Returns "UP", "DOWN", or "NEUTRAL" using the full thumb landmark chain
    (wrist -> MCP -> IP -> TIP), not just a single joint pair — this rejects a
    thumb that's merely bent/rotated in frame rather than genuinely pointing
    up or down, and stays correct regardless of hand rotation.
    Landmarks: 0 = wrist, 2 = thumb MCP, 3 = thumb IP, 4 = thumb tip.
    """
    wrist = lmList[0]
    mcp = lmList[2]
    ip = lmList[3]
    tip = lmList[4]

    vx = tip[1] - mcp[1]
    vy = tip[2] - mcp[2]
    extension = math.hypot(vx, vy)

    if extension < THUMB_EXTENSION_MIN:
        return "NEUTRAL"  # tucked in, not extended enough to judge direction

    if abs(vy) < abs(vx) * THUMB_VERTICAL_RATIO:
        return "NEUTRAL"  # pointing too sideways to call it clearly up or down

    # Image y grows downward. Require the WHOLE chain to agree on direction
    # (tip clearly past ip, past mcp, past wrist) rather than just one joint
    # pair, so a bent-but-not-really-pointing thumb doesn't slip through.
    if vy > 0 and tip[2] > ip[2] and tip[2] > mcp[2] and tip[2] > wrist[2]:
        return "DOWN"
    if vy < 0 and tip[2] < ip[2] and tip[2] < mcp[2] and tip[2] < wrist[2]:
        return "UP"
    return "NEUTRAL"


def isThumbFolded(lmList):
    """
    True when the thumb tip sits close to its own MCP joint — i.e. tucked
    across the palm rather than extended outward. Used to tell MOVE (open
    hand, thumb out) apart from SCREENSHOT (open hand, thumb tucked in).
    """
    mcp = lmList[2]
    tip = lmList[4]
    extension = math.hypot(tip[1] - mcp[1], tip[2] - mcp[2])
    return extension < THUMB_FOLDED_MAX


def classifyPinch(d_index, d_middle, d_ring, d_pinky):
    """
    Returns "index" / "middle" / "ring" / "pinky" / None.

    Picks whichever fingertip is closest to the thumb, but only commits to it
    if (a) that distance is under ITS OWN gesture threshold, AND (b) it's
    clearly closer than every other fingertip by at least PINCH_SEPARATION.
    If two fingertips are nearly tied, this returns None rather than guessing —
    which is exactly the case that previously let "thumb near middle" also
    register as a left click just because the index happened to be nearby too.
    """
    candidates = {
        "index": (d_index, CLICK_DIST),
        "middle": (d_middle, RIGHT_CLICK_DIST),
        "ring": (d_ring, DOUBLE_CLICK_DIST),
        "pinky": (d_pinky, DRAG_START_DIST),
    }
    raw = {name: val[0] for name, val in candidates.items()}

    closest_finger = min(raw, key=raw.get)
    closest_distance = raw[closest_finger]
    threshold = candidates[closest_finger][1]

    if closest_distance >= threshold:
        return None

    for finger, dist in raw.items():
        if finger != closest_finger and dist < closest_distance + PINCH_SEPARATION:
            return None  # another fingertip is too close a competitor — ambiguous

    return closest_finger


def takeScreenshot():
    """
    Captures the full screen and saves it to an absolute screenshots/ folder.
    Returns (success: bool, filepath_or_None) so the caller can tell a real
    save apart from a failure and react accordingly.
    """
    try:
        screenshot_dir = os.path.abspath("screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join(screenshot_dir, f"screenshot_{timestamp}.png")

        screenshot = pyautogui.screenshot()
        screenshot.save(filepath)

        print(f"SCREENSHOT SAVED: {filepath}")
        return True, filepath

    except Exception as e:
        print(f"SCREENSHOT ERROR: {e}")
        return False, None


def showScreenshotNotification(filepath):
    """
    Shows a Windows toast notification after a screenshot is successfully
    saved. Safe to call even if winotify isn't installed or the toast fails
    for any reason — it just prints a message instead of crashing SilentFlow.
    """
    if not WINOTIFY_AVAILABLE:
        print("Notification skipped: winotify not installed (pip install winotify)")
        return

    try:
        toast = Notification(
            app_id="SilentFlow",
            title="Screenshot Captured",
            msg=f"Screenshot saved successfully!\n{filepath}",
            duration="short"
        )
        toast.show()
    except Exception as e:
        print(f"Notification error: {e}")


# ========================
# GESTURE CLASSIFICATION
# ========================
def detectGesture(lmList, fingers, state):
    """
    Returns (gesture_label, distances) where distances =
    (d_index, d_middle, d_ring, d_pinky, d_mid_ring) — for debug display.

    Exactly ONE gesture label is returned per call, in this explicit priority
    order (first match wins):
      1. Active DRAG (must resolve before anything else)
      2. Volume LOCK already engaged -> keep firing Volume Up/Down
         Volume PENDING (forming) -> suppress everything else this frame
      3. Mute
      4. Scroll
      5. Confirmed pinch classification (LEFT/RIGHT/DOUBLE click, DRAG start)
      6. Screenshot
      7. COPY / CUT / PASTE
      8. Play/Pause
      9. Zoom (middle+ring spread)
      10. Window switch (Alt+Tab)
      11. Move
      12. NONE
    """
    now = time.time()

    thumb_tip = (lmList[4][1], lmList[4][2])
    index_tip = (lmList[8][1], lmList[8][2])
    middle_tip = (lmList[12][1], lmList[12][2])
    ring_tip = (lmList[16][1], lmList[16][2])
    pinky_tip = (lmList[20][1], lmList[20][2])

    d_index = distance(thumb_tip, index_tip)
    d_middle = distance(thumb_tip, middle_tip)
    d_ring = distance(thumb_tip, ring_tip)
    d_pinky = distance(thumb_tip, pinky_tip)
    d_mid_ring = distance(middle_tip, ring_tip)
    dists = (d_index, d_middle, d_ring, d_pinky, d_mid_ring)

    f = fingers[1:]  # [index, middle, ring, pinky] — thumb excluded from shape checks

    # ---- shared geometry, computed once ----
    thumb_direction = detectThumbDirection(lmList)
    thumb_folded = isThumbFolded(lmList)
    state["last_thumb_direction"] = thumb_direction

    # ---- DETECT COPY/CUT/PASTE SHAPES ----
    # COPY: Only Pinky up -> [thumb, index, middle, ring, pinky] = [0, 0, 0, 0, 1]
    # CUT: Ring + Pinky up -> [thumb, index, middle, ring, pinky] = [0, 0, 0, 1, 1]
    # PASTE: Index + Middle + Ring + Pinky up -> [thumb, index, middle, ring, pinky] = [0, 1, 1, 1, 1]
    
    copy_shape = (fingers == [0, 0, 0, 0, 1])  # Only pinky up
    cut_shape = (fingers == [0, 0, 0, 1, 1])   # Ring + Pinky up
    paste_shape = (fingers == [0, 1, 1, 1, 1]) # Index + Middle + Ring + Pinky up
    
    # Check if any COPY/CUT/PASTE gesture is active
    is_copy_paste_gesture = copy_shape or cut_shape or paste_shape

    # A "protected" pose: while the hand clearly matches one of these shapes,
    # pinch classification is skipped entirely. This is the core fix for both
    # scroll and fist — previously classifyPinch() ran unconditionally every
    # frame, so a closed fist (or ☝️/✌️) could still have its thumb "win" the
    # pinch classification and get reported as a click/drag instead.
    scroll_shape = f in ([1, 0, 0, 0], [1, 1, 0, 0])
    fist_shape = (f == [0, 0, 0, 0])

    if scroll_shape or fist_shape or is_copy_paste_gesture:
        pinch = None
        if scroll_shape:
            if DEBUG and not state["scroll_lock_active"]:
                print("SCROLL POSE LOCKED - PINCH DETECTION DISABLED")
            state["scroll_lock_active"] = True
        else:
            state["scroll_lock_active"] = False
    else:
        pinch = classifyPinch(d_index, d_middle, d_ring, d_pinky)
        state["scroll_lock_active"] = False

    state["last_pinch"] = pinch                      # debug: "Closest pinch: ..."

    # ---- bookkeeping: confirmation counters (always updated, regardless of
    #      what ultimately gets returned this frame) ----

    # ---- VOLUME pending/lock state machine ----
    # Use exact finger matching for volume detection
    if is_copy_paste_gesture:
        # Volume detection is completely disabled for COPY/CUT/PASTE
        volume_candidate = None
        state["volume_candidate"] = None
        # Don't modify volume lock state during COPY/CUT/PASTE
    else:
        # Check for exact volume pose: all four non-thumb fingers must be folded
        four_fingers_folded = (f == [0, 0, 0, 0])
        
        volume_candidate = None
        if four_fingers_folded and thumb_direction == "UP":
            volume_candidate = "UP"
        elif four_fingers_folded and thumb_direction == "DOWN":
            volume_candidate = "DOWN"
        
        state["volume_candidate"] = volume_candidate  # debug: "Volume candidate: ..."

        # Release bookkeeping — only relevant while a lock is already engaged.
        if state["volume_lock"] is not None:
            if volume_candidate == state["volume_lock"]:
                state["volume_release_count"] = 0
            else:
                state["volume_release_count"] += 1
                if state["volume_release_count"] >= VOLUME_RELEASE_FRAMES:
                    if DEBUG:
                        print(f"VOLUME {state['volume_lock']} RELEASED")
                    state["volume_lock"] = None
                    state["volume_release_count"] = 0

        # Pending/confirm bookkeeping — only while nothing is currently locked.
        if state["volume_lock"] is None:
            if volume_candidate is not None:
                if state["volume_pending"] == volume_candidate:
                    state["volume_pending_count"] += 1
                else:
                    state["volume_pending"] = volume_candidate
                    state["volume_pending_count"] = 1
                    if DEBUG:
                        print(f"VOLUME {volume_candidate} PENDING")

                if state["volume_pending_count"] >= VOLUME_CONFIRM_FRAMES:
                    state["volume_lock"] = volume_candidate
                    state["volume_pending"] = None
                    state["volume_pending_count"] = 0
                    state["volume_release_count"] = 0
                    if DEBUG:
                        print(f"VOLUME {volume_candidate} CONFIRMED")
            else:
                state["volume_pending"] = None
                state["volume_pending_count"] = 0

    # Scroll is allowed only when the thumb is genuinely neutral AND no volume
    # gesture is forming or locked AND not in COPY/CUT/PASTE.
    scroll_allowed = (
        thumb_direction == "NEUTRAL"
        and state["volume_lock"] is None
        and state["volume_pending"] is None
        and not is_copy_paste_gesture  # Disable scroll during COPY/CUT/PASTE
    )
    state["scroll_allowed"] = scroll_allowed  # debug: "Scroll allowed: ..."

    # Right click / double click — only counts up while THAT specific pinch
    # is the confirmed classification (not just "close enough").
    if pinch == "middle":
        state["right_click_frame_count"] += 1
    else:
        state["right_click_frame_count"] = 0

    if pinch == "ring":
        state["double_click_frame_count"] += 1
    else:
        state["double_click_frame_count"] = 0

    # ---- SIMPLIFIED MUTE: FIST / MUTE ----
    # Now works like a button: press on fist close, release on fist open
    # No more complex release frame counting
    
    if fist_shape:
        state["fist_confirm_count"] += 1
    else:
        state["fist_confirm_count"] = 0
        # SIMPLIFIED: Immediately re-arm when fist is released
        if state["mute_latched"]:
            state["mute_latched"] = False
            if DEBUG:
                print("MUTE RE-ARMED (fist released)")

    confirmed_fist = state["fist_confirm_count"] >= MUTE_CONFIRM_FRAMES

    # ---- COPY / CUT / PASTE detection ----
    # Update COPY confirmation counter
    if copy_shape:
        state["copy_confirm_count"] += 1
        if DEBUG and state["copy_confirm_count"] == COPY_CONFIRM_FRAMES:
            print("COPY GESTURE DETECTED")
    else:
        state["copy_confirm_count"] = 0

    # Update CUT confirmation counter
    if cut_shape:
        state["cut_confirm_count"] += 1
        if DEBUG and state["cut_confirm_count"] == CUT_CONFIRM_FRAMES:
            print("CUT GESTURE DETECTED")
    else:
        state["cut_confirm_count"] = 0

    # Update PASTE confirmation counter
    if paste_shape:
        state["paste_confirm_count"] += 1
        if DEBUG and state["paste_confirm_count"] >= PASTE_CONFIRM_FRAMES:
            print("PASTE GESTURE DETECTED")
    else:
        state["paste_confirm_count"] = 0

    # COPY / CUT / PASTE release tracking
    # Reset latches when pose is released
    if not copy_shape and state["copy_latched"]:
        state["copy_release_count"] += 1
        if state["copy_release_count"] >= 3:  # 3 frames to re-arm
            state["copy_latched"] = False
            state["copy_release_count"] = 0
            if DEBUG:
                print("COPY RE-ARMED")
    elif copy_shape:
        state["copy_release_count"] = 0

    if not cut_shape and state["cut_latched"]:
        state["cut_release_count"] += 1
        if state["cut_release_count"] >= 3:
            state["cut_latched"] = False
            state["cut_release_count"] = 0
            if DEBUG:
                print("CUT RE-ARMED")
    elif cut_shape:
        state["cut_release_count"] = 0

    if not paste_shape and state["paste_latched"]:
        state["paste_release_count"] += 1
        if state["paste_release_count"] >= 3:
            state["paste_latched"] = False
            state["paste_release_count"] = 0
            if DEBUG:
                print("PASTE RE-ARMED")
    elif paste_shape:
        state["paste_release_count"] = 0

    # ---- SCREENSHOT: four fingers open, thumb folded across the palm ----
    # **FIX: Skip screenshot detection during COPY/CUT/PASTE**
    if not is_copy_paste_gesture:
        screenshot_shape = (f == [1, 1, 1, 1]) and thumb_folded
        if screenshot_shape:
            state["screenshot_frame_count"] += 1
            if DEBUG and state["screenshot_frame_count"] == SCREENSHOT_STABLE_FRAMES:
                print("SCREENSHOT GESTURE DETECTED")
        else:
            state["screenshot_frame_count"] = 0
    else:
        # Reset screenshot frame count during COPY/CUT/PASTE to prevent accidental triggers
        state["screenshot_frame_count"] = 0

    # Play/Pause
    playpause_shape = (f == [1, 0, 0, 1])
    if playpause_shape:
        state["playpause_frame_count"] += 1
    else:
        state["playpause_frame_count"] = 0

    # Zoom (middle+ring spread, tracked over a short time window)
    zoom_candidate = (f == [0, 1, 1, 0])
    if zoom_candidate:
        state["zoom_history"].append((now, d_mid_ring))
        while state["zoom_history"] and now - state["zoom_history"][0][0] > ZOOM_HISTORY_WINDOW:
            state["zoom_history"].popleft()
    else:
        state["zoom_history"].clear()

    # ---- NEW: One-hand pinch zoom using thumb and index ----
    # This is a SEPARATE feature that doesn't interfere with any existing gestures
    # It uses the thumb-index distance normalized by palm width
    # Only activates when thumb and index are the ONLY fingers extended
    
    # Calculate palm width for normalization (landmark 5 = index MCP, landmark 17 = pinky MCP)
    palm_width = distance(
        (lmList[5][1], lmList[5][2]),
        (lmList[17][1], lmList[17][2])
    )
    
    # Normalize thumb-index distance
    if palm_width > 0:
        pinch_zoom_ratio = d_index / palm_width
    else:
        pinch_zoom_ratio = 0.0
    
    # Clamp ratio
    pinch_zoom_ratio = max(PINCH_ZOOM_MIN_RATIO, min(PINCH_ZOOM_MAX_RATIO, pinch_zoom_ratio))
    
    # Detect pinch-zoom gesture: ONLY thumb and index extended (all others folded)
    # Pattern: [thumb, index, middle, ring, pinky] = [1, 1, 0, 0, 0]
    is_pinch_zoom_gesture = fingers == [1, 1, 0, 0, 0]
    
    # Initialize or update pinch zoom state
    if is_pinch_zoom_gesture:
        if not state["pinch_zoom_initialized"]:
            state["pinch_zoom_prev_ratio"] = pinch_zoom_ratio
            state["pinch_zoom_initialized"] = True
        
        # Calculate change in normalized distance
        ratio_change = pinch_zoom_ratio - state["pinch_zoom_prev_ratio"]
        
        # Only trigger if change exceeds deadzone
        if abs(ratio_change) > PINCH_ZOOM_DEADZONE:
            if now - state["last_pinch_zoom_time"] > PINCH_ZOOM_MIN_INTERVAL:
                # Calculate zoom steps based on ratio change
                zoom_steps = int(abs(ratio_change) * PINCH_ZOOM_STEP_SCALE)
                zoom_steps = max(1, min(zoom_steps, 8))  # Limit to 1-8 steps per trigger
                
                if ratio_change > 0:
                    # Fingers spreading apart -> Zoom In
                    for _ in range(zoom_steps):
                        pyautogui.hotkey("ctrl", "+")
                    if DEBUG:
                        print(f"PINCH ZOOM IN (steps: {zoom_steps})")
                else:
                    # Fingers coming together -> Zoom Out
                    for _ in range(zoom_steps):
                        pyautogui.hotkey("ctrl", "-")
                    if DEBUG:
                        print(f"PINCH ZOOM OUT (steps: {zoom_steps})")
                
                state["last_pinch_zoom_time"] = now
                state["pinch_zoom_prev_ratio"] = pinch_zoom_ratio
        
        # Smoothly track the ratio
        state["pinch_zoom_prev_ratio"] = state["pinch_zoom_prev_ratio"] * 0.6 + pinch_zoom_ratio * 0.4
        
    else:
        # Reset zoom tracking when gesture ends
        if state["pinch_zoom_initialized"]:
            state["pinch_zoom_initialized"] = False
            state["pinch_zoom_prev_ratio"] = 0.0

    # Window switch (three-finger horizontal swipe, tracked over a short time window)
    swipe_candidate = (f == [1, 1, 1, 0])
    if swipe_candidate:
        state["swipe_history"].append((now, index_tip[0]))
        while state["swipe_history"] and now - state["swipe_history"][0][0] > SWIPE_HISTORY_WINDOW:
            state["swipe_history"].popleft()
    else:
        state["swipe_history"].clear()
        state["swipe_fired"] = False

    # ============================================================
    # 1. Active DRAG must resolve before anything else is considered
    # ============================================================
    if state["is_dragging"]:
        if d_pinky > DRAG_RELEASE_DIST:
            return "DRAG_END", dists
        return "DRAGGING", dists

    # ============================================================
    # 2. Volume — a LOCKED gesture keeps firing; a PENDING one suppresses
    #    absolutely everything else (scroll, mute, clicks included) until
    #    it either confirms into a lock or the candidate disappears.
    # ============================================================
    if not is_copy_paste_gesture:
        if state["volume_lock"] is not None:
            return ("VOLUME_UP" if state["volume_lock"] == "UP" else "VOLUME_DOWN"), dists

        if state["volume_pending"] is not None:
            return "NONE", dists

    # ============================================================
    # 3. SIMPLIFIED MUTE — Fist
    # ============================================================
    if fist_shape:
        if confirmed_fist:
            return "MUTE", dists
        return "NONE", dists

    # ============================================================
    # 4. Scroll — requires scroll_allowed (neutral thumb, no volume
    #    candidate forming or locked, AND not in COPY/CUT/PASTE)
    # ============================================================
    if f == [1, 0, 0, 0] and scroll_allowed:
        return "SCROLL_UP", dists

    if f == [1, 1, 0, 0] and scroll_allowed:
        return "SCROLL_DOWN", dists

    # ============================================================
    # 5. Confirmed pinch classification
    # ============================================================
    if pinch == "index":
        # Left click stays immediately responsive (no confirmation-frame
        # delay) — it already requires a clean, unambiguous pinch via
        # classifyPinch()'s separation check, and handleGesture()'s edge
        # trigger requires a full release before it can fire again.
        return "LEFT_CLICK", dists

    if pinch == "middle":
        if state["right_click_frame_count"] >= RIGHT_CLICK_CONFIRM_FRAMES:
            return "RIGHT_CLICK", dists
        return "NONE", dists  # still confirming — do not fall through to anything else

    if pinch == "ring":
        if state["double_click_frame_count"] >= DOUBLE_CLICK_CONFIRM_FRAMES:
            return "DOUBLE_CLICK", dists
        return "NONE", dists

    if pinch == "pinky":
        return "DRAG_START", dists  # responsive on purpose — see LEFT_CLICK note above

    # ============================================================
    # 6. Screenshot (open hand, thumb folded — no pinch involved)
    # ============================================================
    if not is_copy_paste_gesture:
        if screenshot_shape and state["screenshot_frame_count"] >= SCREENSHOT_STABLE_FRAMES:
            return "SCREENSHOT", dists
        # If screenshot not confirmed yet, return NONE to allow other gestures
        if screenshot_shape:
            return "NONE", dists

    # ============================================================
    # 7. COPY / CUT / PASTE (with proper confirmation)
    # ============================================================
    # COPY: Only Pinky up (unchanged, requires 3 frames)
    if copy_shape and state["copy_confirm_count"] >= COPY_CONFIRM_FRAMES:
        return "COPY", dists
    
    # CUT: Ring + Pinky up (unchanged, requires 3 frames)
    if cut_shape and state["cut_confirm_count"] >= CUT_CONFIRM_FRAMES:
        return "CUT", dists
    
    # PASTE: Index + Middle + Ring + Pinky up (requires only 1 frame now)
    if paste_shape and state["paste_confirm_count"] >= PASTE_CONFIRM_FRAMES:
        return "PASTE", dists

    # ============================================================
    # 8. Play/Pause
    # ============================================================
    if playpause_shape:
        if state["playpause_frame_count"] >= PLAYPAUSE_CONFIRM_FRAMES:
            return "PLAY_PAUSE", dists
        return "NONE", dists

    # ============================================================
    # 9. Zoom (middle+ring spread)
    # ============================================================
    if zoom_candidate:
        if len(state["zoom_history"]) >= 2:
            delta = state["zoom_history"][-1][1] - state["zoom_history"][0][1]
            if delta > ZOOM_MIN_DELTA:
                return "ZOOM_IN", dists
            if delta < -ZOOM_MIN_DELTA:
                return "ZOOM_OUT", dists
        return "NONE", dists

    # ============================================================
    # 10. Window switch (Alt+Tab)
    # ============================================================
    if swipe_candidate:
        if len(state["swipe_history"]) >= 2 and not state["swipe_fired"]:
            displacement = state["swipe_history"][-1][1] - state["swipe_history"][0][1]
            if abs(displacement) > SWIPE_MIN_DISTANCE:
                state["swipe_fired"] = True
                return "WINDOW_SWITCH", dists
        return "NONE", dists

    # ============================================================
    # 11. Move — open hand with the thumb NOT folded (distinguishes it from Screenshot)
    # ============================================================
    if f == [1, 1, 1, 1] and not thumb_folded:
        return "MOVE", dists

    # ============================================================
    # 12. Nothing matched
    # ============================================================
    return "NONE", dists


# ========================
# IMPROVED CURSOR MOVEMENT - Smooth & Glitch-Free
# ========================
def moveCursor(lmList, state):
    """
    Advanced cursor smoothing combining multiple techniques:
    1. Exponential moving average (EMA)
    2. History averaging
    3. Jitter threshold
    4. Movement prediction
    5. Velocity-based adaptive smoothing
    """
    x1, y1 = lmList[8][1], lmList[8][2]  # index fingertip drives the cursor

    # X range reversed (SCREEN_W -> 0) so cursor direction matches hand direction
    screenX = np.interp(x1, (0, CAM_W), (SCREEN_W, 0))
    screenY = np.interp(y1, (0, CAM_H), (0, SCREEN_H))

    # Calculate movement since last frame
    if state["prevX"] != 0 or state["prevY"] != 0:
        dx = screenX - state["prevX"]
        dy = screenY - state["prevY"]
        movement = math.hypot(dx, dy)
        
        # Jitter threshold - ignore tiny movements
        if movement < JITTER_THRESHOLD:
            # Don't update position - keep cursor stable
            return
        
        # Adaptive smoothing based on movement speed
        # Fast movements = less smoothing (more responsive)
        # Slow movements = more smoothing (less jitter)
        speed_factor = min(movement / 50.0, 1.0)
        adaptive_smooth = SMOOTHENING * (1.0 - speed_factor * 0.4)
        adaptive_smooth = max(3, adaptive_smooth)  # Clamp to reasonable range
    else:
        adaptive_smooth = SMOOTHENING

    # Store in history for averaging
    state["pos_history"].append((screenX, screenY))
    if len(state["pos_history"]) > SMOOTHING_HISTORY:
        state["pos_history"].popleft()

    # Calculate moving average from history
    if len(state["pos_history"]) >= 3:
        avg_x = sum(p[0] for p in state["pos_history"]) / len(state["pos_history"])
        avg_y = sum(p[1] for p in state["pos_history"]) / len(state["pos_history"])
    else:
        avg_x = screenX
        avg_y = screenY

    # Exponential moving average (EMA) - smooths current position toward average
    if state["prevX"] != 0 or state["prevY"] != 0:
        # Blend: 60% current average, 40% previous position
        smoothX = avg_x * SMOOTHING_WEIGHT + state["prevX"] * (1 - SMOOTHING_WEIGHT)
        smoothY = avg_y * SMOOTHING_WEIGHT + state["prevY"] * (1 - SMOOTHING_WEIGHT)
    else:
        smoothX = avg_x
        smoothY = avg_y

    # Apply slight prediction for smoother motion during consistent movement
    if len(state["pos_history"]) >= 3 and movement > 10:
        # Predict next position based on recent movement trend
        pred_x = smoothX + (smoothX - state["prevX"]) * PREDICTION_WEIGHT
        pred_y = smoothY + (smoothY - state["prevY"]) * PREDICTION_WEIGHT
        smoothX = smoothX * (1 - PREDICTION_WEIGHT) + pred_x * PREDICTION_WEIGHT
        smoothY = smoothY * (1 - PREDICTION_WEIGHT) + pred_y * PREDICTION_WEIGHT

    # Clamp to screen bounds
    currX = float(np.clip(smoothX, 0, SCREEN_W - 1))
    currY = float(np.clip(smoothY, 0, SCREEN_H - 1))

    # Only move if the change is significant (reduces micro-movements)
    if math.hypot(currX - state["prevX"], currY - state["prevY"]) > JITTER_THRESHOLD:
        pyautogui.moveTo(currX, currY)
        state["prevX"], state["prevY"] = currX, currY


# ========================
# GESTURE -> ACTION
# ========================
def handleGesture(gesture, lmList, state):
    now = time.time()
    entered = (gesture != state["previous_gesture"])  # true on the first frame of this gesture

    if gesture == "MOVE":
        moveCursor(lmList, state)

    elif gesture == "DRAG_START":
        pyautogui.mouseDown()
        state["is_dragging"] = True
        moveCursor(lmList, state)
        print("DRAG STARTED")

    elif gesture == "DRAGGING":
        moveCursor(lmList, state)

    elif gesture == "DRAG_END":
        pyautogui.mouseUp()
        state["is_dragging"] = False
        print("DRAG ENDED")

    elif gesture == "LEFT_CLICK":
        if entered and now - state["last_click_time"] > CLICK_COOLDOWN:
            pyautogui.click()
            state["last_click_time"] = now
            print("LEFT CLICK")

    elif gesture == "RIGHT_CLICK":
        if entered and now - state["last_right_click_time"] > RIGHT_CLICK_COOLDOWN:
            pyautogui.rightClick()
            state["last_right_click_time"] = now
            print("RIGHT CLICK")

    elif gesture == "DOUBLE_CLICK":
        if entered and now - state["last_double_click_time"] > DOUBLE_CLICK_COOLDOWN:
            pyautogui.doubleClick()
            state["last_double_click_time"] = now
            print("DOUBLE CLICK")

    # ============================================================
    # SIMPLIFIED MUTE — Works like a button
    # ============================================================
    elif gesture == "MUTE":
        # Simple edge trigger: execute once when fist becomes confirmed
        if not state["mute_latched"]:
            pyautogui.press("volumemute")
            state["mute_latched"] = True
            state["last_mute_time"] = now
            print("MUTE / UNMUTE TOGGLED")
        else:
            if DEBUG:
                print("MUTE LOCKED - Holding fist")

    elif gesture == "SCROLL_UP":
        if now - state["last_scroll_time"] > SCROLL_COOLDOWN:
            windows_scroll(SCROLL_NOTCHES)
            state["last_scroll_time"] = now
            if DEBUG:
                print("EXECUTING SCROLL: UP")

    elif gesture == "SCROLL_DOWN":
        if now - state["last_scroll_time"] > SCROLL_COOLDOWN:
            windows_scroll(-SCROLL_NOTCHES)
            state["last_scroll_time"] = now
            if DEBUG:
                print("EXECUTING SCROLL: DOWN")

    elif gesture == "VOLUME_UP":
        if now - state["last_volume_time"] > VOLUME_COOLDOWN:
            pyautogui.press("volumeup")
            state["last_volume_time"] = now
            if DEBUG:
                print("EXECUTING: VOLUME UP")

    elif gesture == "VOLUME_DOWN":
        if now - state["last_volume_time"] > VOLUME_COOLDOWN:
            pyautogui.press("volumedown")
            state["last_volume_time"] = now
            if DEBUG:
                print("EXECUTING: VOLUME DOWN")

    elif gesture == "ZOOM_IN":
        if now - state["last_zoom_time"] > ZOOM_COOLDOWN:
            pyautogui.hotkey("ctrl", "+")
            state["last_zoom_time"] = now
            print("ZOOM IN")

    elif gesture == "ZOOM_OUT":
        if now - state["last_zoom_time"] > ZOOM_COOLDOWN:
            pyautogui.hotkey("ctrl", "-")
            state["last_zoom_time"] = now
            print("ZOOM OUT")

    elif gesture == "PLAY_PAUSE":
        if entered:
            pyautogui.press("playpause")
            print("PLAY/PAUSE TOGGLED")

    elif gesture == "SCREENSHOT":
        if entered and now - state["last_screenshot_time"] > SCREENSHOT_COOLDOWN:
            success, filepath = takeScreenshot()
            state["last_screenshot_time"] = now
            if success:
                state["screenshot_flash_until"] = now + 1.2
                showScreenshotNotification(filepath)

    # ============================================================
    # COPY / CUT / PASTE
    # ============================================================
    elif gesture == "COPY":
        # Edge-triggered: only execute when first detected, not while held
        if not state["copy_latched"] and state["copy_confirm_count"] >= COPY_CONFIRM_FRAMES:
            if now - state["last_copy_time"] > COPY_COOLDOWN:
                pyautogui.hotkey("ctrl", "c")
                state["last_copy_time"] = now
                state["copy_latched"] = True
                print("COPY EXECUTED")
        else:
            if DEBUG and state["copy_latched"]:
                print("COPY LOCKED - Holding pose")

    elif gesture == "CUT":
        if not state["cut_latched"] and state["cut_confirm_count"] >= CUT_CONFIRM_FRAMES:
            if now - state["last_cut_time"] > CUT_COOLDOWN:
                pyautogui.hotkey("ctrl", "x")
                state["last_cut_time"] = now
                state["cut_latched"] = True
                print("CUT EXECUTED")
        else:
            if DEBUG and state["cut_latched"]:
                print("CUT LOCKED - Holding pose")

    elif gesture == "PASTE":
        if not state["paste_latched"] and state["paste_confirm_count"] >= PASTE_CONFIRM_FRAMES:
            if now - state["last_paste_time"] > PASTE_COOLDOWN:
                pyautogui.hotkey("ctrl", "v")
                state["last_paste_time"] = now
                state["paste_latched"] = True
                print("PASTE EXECUTED")
        else:
            if DEBUG and state["paste_latched"]:
                print("PASTE LOCKED - Holding pose")

    elif gesture == "WINDOW_SWITCH":
        if entered:
            pyautogui.hotkey("alt", "tab")
            print("ALT+TAB")

    state["previous_gesture"] = gesture


# ========================
# ON-SCREEN STATUS OVERLAY - CLEAN UI
# ========================
def drawStatus(img, gesture, fps, fingers, dists, state):
    """
    Clean overlay showing ONLY:
    1. FPS
    2. Gesture
    3. Fingers
    
    All other status text has been removed for a clean interface.
    """
    # FPS - Top left
    cv2.putText(
        img, f"FPS: {int(fps)}", (20, 40),
        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
    )

    # Current Gesture - Below FPS
    label = gesture if gesture != "NONE" else "IDLE"
    cv2.putText(
        img, f"Gesture: {label}", (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2
    )

    # Detected Fingers - Below Gesture
    cv2.putText(
        img, f"Fingers: {fingers}", (20, 130),
        cv2.FONT_HERSHEY_PLAIN, 1.4, (200, 200, 200), 2
    )


# ========================
# MAIN LOOP
# ========================
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera failed to open")
        return

    detector = HandDetector()
    pTime = 0

    # All persistent, per-run state lives in this single dict so detectGesture(),
    # handleGesture(), and moveCursor() can read/update it without globals.
    state = {
        "prevX": 0, "prevY": 0,
        "is_dragging": False,
        "previous_gesture": "NONE",
        "last_click_time": 0,
        "last_right_click_time": 0,
        "last_double_click_time": 0,
        "last_scroll_time": 0,
        "last_volume_time": 0,
        "last_zoom_time": 0,
        "last_screenshot_time": 0,
        "last_mute_time": 0,
        "zoom_history": deque(),
        "swipe_history": deque(),
        "swipe_fired": False,
        "screenshot_frame_count": 0,
        "screenshot_flash_until": 0,
        "last_thumb_direction": "NEUTRAL",
        "volume_candidate": None,
        "volume_pending": None,
        "volume_pending_count": 0,
        "volume_lock": None,
        "volume_release_count": 0,
        "scroll_allowed": True,
        "right_click_frame_count": 0,
        "double_click_frame_count": 0,
        "fist_confirm_count": 0,
        "mute_latched": False,
        "playpause_frame_count": 0,
        "last_pinch": None,
        "scroll_lock_active": False,
        # COPY / CUT / PASTE state
        "last_copy_time": 0,
        "last_cut_time": 0,
        "last_paste_time": 0,
        "copy_confirm_count": 0,
        "cut_confirm_count": 0,
        "paste_confirm_count": 0,
        "copy_latched": False,
        "cut_latched": False,
        "paste_latched": False,
        "copy_release_count": 0,
        "cut_release_count": 0,
        "paste_release_count": 0,
        # Pinch Zoom state
        "last_pinch_zoom_time": 0,
        "pinch_zoom_initialized": False,
        "pinch_zoom_prev_ratio": 0.0,
        # Cursor smoothing state
        "pos_history": deque(maxlen=SMOOTHING_HISTORY),
    }

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        gesture = "NONE"
        fingers = [0, 0, 0, 0, 0]
        dists = (0, 0, 0, 0, 0)

        if lmList:
            fingers = fingersUp(lmList)
            gesture, dists = detectGesture(lmList, fingers, state)
            handleGesture(gesture, lmList, state)
        else:
            # Safety: if the hand disappears mid-drag, release the mouse button
            # so it can never get stuck held down.
            if state["is_dragging"]:
                pyautogui.mouseUp()
                state["is_dragging"] = False
                print("DRAG ENDED (hand lost)")
            state["previous_gesture"] = "NONE"
            state["zoom_history"].clear()
            state["swipe_history"].clear()
            state["swipe_fired"] = False
            state["screenshot_frame_count"] = 0
            state["volume_candidate"] = None
            state["volume_pending"] = None
            state["volume_pending_count"] = 0
            state["volume_lock"] = None
            state["volume_release_count"] = 0
            state["right_click_frame_count"] = 0
            state["double_click_frame_count"] = 0
            state["fist_confirm_count"] = 0
            # SIMPLIFIED MUTE: Re-arm on hand lost
            if state["mute_latched"]:
                state["mute_latched"] = False
                if DEBUG:
                    print("MUTE RE-ARMED (hand lost)")
            state["playpause_frame_count"] = 0
            state["last_pinch"] = None
            # Reset COPY/CUT/PASTE confirm counts when hand lost
            state["copy_confirm_count"] = 0
            state["cut_confirm_count"] = 0
            state["paste_confirm_count"] = 0
            # Reset pinch zoom state
            state["pinch_zoom_initialized"] = False
            state["pinch_zoom_prev_ratio"] = 0.0
            # Reset cursor history on hand loss
            state["pos_history"].clear()

        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime else 0
        pTime = cTime

        drawStatus(img, gesture, fps, fingers, dists, state)

        cv2.imshow("SilentFlow – Touchless Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if state["is_dragging"]:
                pyautogui.mouseUp()
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()