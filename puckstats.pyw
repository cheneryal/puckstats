import cv2
import numpy as np
import mss
# import easyocr # Moved to main() to handle ModuleNotFoundError and packaging issues
import time
import re
import os
import sys # Added for icon path
import pygetwindow as gw
# import torch # Moved to main() to handle ModuleNotFoundError and packaging issues
import traceback
import json
import winreg

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

# --- Configuration ---

CONFIG_FILE = "config.json"
GAME_WINDOW_TITLE = "Puck"

# Region of Interest (ROI) Coordinates (RELATIVE to game window)
# This will be updated by the manual selection tool.
ROI_TIME_REL = {'top': 15, 'left': 1160, 'width': 273, 'height': 87}
ROI_PERIOD_REL = {'top': 120, 'left': 1118, 'width': 330, 'height': 78}

# --- Game Text File Configuration ---
GAME_TEXT_FILES_DIR = "" # Will be auto-detected or loaded from config
GOAL_TEAM_FILE = "goal_team.txt"
GOAL_SCORER_FILE = "goal_scorer.txt"
GOAL_ASSISTER_FILE = "goal_assister.txt"
GOAL_ASSISTER2_FILE = "goal_assister2.txt"
SCORE_RED_FILE = "scorered.txt"
SCORE_BLUE_FILE = "scoreblue.txt"
CLOCK_FILE = "realclock.txt"
PERIOD_NAME_FILE = "period_name.txt"
# --- End Game Text File Configuration ---

# General Settings
POLL_INTERVAL_SECONDS = 0.2 # Adjusted for GUI responsiveness
RECHECK_WINDOW_INTERVAL_SECONDS = 15
DEBUG_ENABLED = False
DEBUG_CANVAS_WIDTH = 800
USE_GPU_IF_AVAILABLE = False # Set to False to force CPU usage for EasyOCR. Reduces EXE size with CPU-only PyTorch.

ROI_ADJUSTMENT_STEP = 5 # Pixels to adjust ROI by for each button press

# PyInstaller: Tell EasyOCR to use a local model directory for packaging.
EASYOCR_MODEL_DIR = "easyocr_models"
# --- End Configuration ---

# --- Globals ---
last_known_good_period = 0
last_known_good_period_name = ""
last_period_roi_state = "UNKNOWN"
last_logged_stat_for_debug = ""
last_time_seconds = -1
easyocr_reader = None
game_window_roi = None
last_window_check_time = 0

logged_stats = []
stats_text_widget = None
gui_root = None
roi_display_label = None
debug_image_label = None

# --- Helper Functions ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def debug_print(*args, **kwargs):
    if DEBUG_ENABLED:
        print("[Debug]", *args, **kwargs)

def get_game_window_roi(title):
    target_window = None
    try:
        windows = gw.getWindowsWithTitle(title)
        if not windows: return None
        possible_game_windows = []
        for i, w in enumerate(windows):
              if w.title == title and not w.isMinimized and w.width > 100 and w.height > 100:
                  possible_game_windows.append(w)
        if not possible_game_windows: return None
        target_window = possible_game_windows[0]
        roi = {"top": target_window.top, "left": target_window.left, "width": target_window.width, "height": target_window.height}
        if roi["width"] <= 0 or roi["height"] <= 0:
            print(f"[ERROR] Selected window '{target_window.title}' has invalid dimensions: {roi}")
            return None
        return roi
    except Exception as e:
        print(f"[ERROR] Failed to get game window geometry: {e}")
        return None

def preprocess_image_for_ocr(img_bgr):
    """Preprocesses BGR image. Returns processed image (usually binary)."""
    if img_bgr is None or img_bgr.size == 0: return None
    if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    elif len(img_bgr.shape) == 2:
        gray = img_bgr
    else:
        print(f"Warn: Unexpected shape in preprocess: {img_bgr.shape}")
        return None

    try:
        thresh_val = 200
        _, processed = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        debug_print("Applied Standard Threshold")
    except cv2.error as e:
        print(f"Error during OpenCV processing: {e}")
        return None

    return processed


def ocr_region(sct, screen_roi, roi_content_type="general_text"):
    global easyocr_reader
    img_np, processed_img, text = None, None, ""
    img_bgr = None

    if easyocr_reader is None:
        print("[ERROR] EasyOCR Reader not initialized!")
        return "", None, None
    if not screen_roi or screen_roi['width'] <= 0 or screen_roi['height'] <= 0:
          debug_print(f"Invalid screen_roi for OCR: {screen_roi}")
          return "", None, None

    try:
        img_raw = sct.grab(screen_roi)
        if img_raw is None: raise ValueError("sct.grab returned None")
        img_np = np.array(img_raw)
        if img_np is None or img_np.size == 0 : raise ValueError("Failed to convert grab to numpy array")

        if len(img_np.shape) == 3 and img_np.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
              img_bgr = img_np
        else:
              print(f"Warning: Unexpected image format from mss: {img_np.shape}")
              raise ValueError(f"Unexpected raw image format: {img_np.shape}")

        if img_bgr is None: raise ValueError("cvtColor failed")

        processed_img_for_debug = preprocess_image_for_ocr(img_bgr)

        ocr_params = {'detail': 0, 'paragraph': True}
        if roi_content_type == "time_digits":
            ocr_params['allowlist'] = '0123456789:'
            debug_print("Using EasyOCR with allowlist for time digits.")
        elif roi_content_type == "period_or_goal":
              ocr_params['allowlist'] = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
              debug_print("Using EasyOCR with allowlist for period/goal text.")

        ocr_start_time = time.time()
        results = easyocr_reader.readtext(img_bgr, **ocr_params)
        ocr_duration = time.time() - ocr_start_time
        debug_print(f"EasyOCR took {ocr_duration:.3f}s")

        text = ""
        if results:
            text = " ".join(results).strip()
        debug_print(f"EasyOCR Raw Results: {results}")

    except mss.ScreenShotError as sse:
        print(f"Error taking screenshot for ROI {screen_roi}: {sse}")
        return "", None, None
    except Exception as e:
        print(f"Error during EasyOCR processing for ROI {screen_roi} (Type: {roi_content_type}): {e}")
        raw_shape = img_np.shape if img_np is not None else "N/A"
        bgr_shape = img_bgr.shape if img_bgr is not None else "N/A"
        proc_shape = processed_img_for_debug.shape if processed_img_for_debug is not None else "N/A"
        debug_print(f"ocr_region (EasyOCR) FAILED. State: raw={raw_shape}, bgr={bgr_shape}, proc={proc_shape}", flush=True)
        return "", None, None

    debug_print(f"ocr_region (EasyOCR) returning OK: text='{text}', raw_shape={img_np.shape if img_np is not None else 'N/A'}, proc_shape={processed_img_for_debug.shape if processed_img_for_debug is not None else 'N/A'}", flush=True)
    return text, img_np, processed_img_for_debug


# --- Read Game Text File Function ---
def read_game_text_file(filename):
    """Reads a text file from the game's textfiles directory."""
    if not filename:
        return ""
    try:
        filepath = os.path.join(GAME_TEXT_FILES_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                debug_print(f"Read from '{filename}': '{content}'")
                return content
        else:
            debug_print(f"File not found: {filepath}")
            return ""
    except Exception as e:
        print(f"[ERROR] Failed to read game text file '{filename}': {e}")
        return ""

# --- Config Save/Load Functions ---
def save_config():
    """Saves the current ROI_PERIOD_REL and game path to the config file."""
    global ROI_PERIOD_REL, GAME_TEXT_FILES_DIR
    config_data = {
        "ROI_PERIOD_REL": ROI_PERIOD_REL,
        "GAME_TEXT_FILES_DIR": GAME_TEXT_FILES_DIR
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)
        debug_print(f"Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to save configuration: {e}")

def load_config():
    """Loads ROI_PERIOD_REL and game path from the config file if it exists."""
    global ROI_PERIOD_REL, GAME_TEXT_FILES_DIR
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config_data = json.load(f)

            loaded_roi = config_data.get("ROI_PERIOD_REL")
            if loaded_roi and all(k in loaded_roi for k in ('top', 'left', 'width', 'height')):
                ROI_PERIOD_REL = loaded_roi
                print(f"Loaded ROI configuration from {CONFIG_FILE}")
            else:
                print(f"[WARN] ROI configuration in {CONFIG_FILE} is invalid. Using defaults.")

            loaded_path = config_data.get("GAME_TEXT_FILES_DIR")
            if loaded_path and os.path.isdir(loaded_path):
                GAME_TEXT_FILES_DIR = loaded_path
                print(f"Loaded Game Text Files directory from {CONFIG_FILE}: {GAME_TEXT_FILES_DIR}")

        except Exception as e:
            print(f"[ERROR] Failed to load or parse configuration from {CONFIG_FILE}: {e}. Using defaults.")
    else:
        debug_print("No config file found. Using default settings.")


def find_and_set_game_path():
    """
    Finds the game's textfiles directory.
    1. Checks if a valid path was loaded from config.
    2. Tries to auto-detect via Windows Registry (Steam path).
    3. Prompts the user to select the folder if auto-detection fails.
    Returns True if a valid path is set, False otherwise.
    """
    global GAME_TEXT_FILES_DIR, gui_root
    if GAME_TEXT_FILES_DIR and os.path.isdir(GAME_TEXT_FILES_DIR):
        print(f"Using game text files path from config: {GAME_TEXT_FILES_DIR}")
        return True

    print("Attempting to auto-detect Puck game folder...")
    try:
        steam_path = ""
        for key_path in (r"SOFTWARE\WOW6432Node\Valve\Steam", r"SOFTWARE\Valve\Steam"):
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                    steam_path_val, _ = winreg.QueryValueEx(key, "InstallPath")
                    if steam_path_val:
                        steam_path = steam_path_val
                        break
            except FileNotFoundError:
                continue

        if steam_path:
            possible_textfiles_path = os.path.join(steam_path, "steamapps", "common", "Puck", "textfiles")
            if os.path.isdir(possible_textfiles_path):
                print(f"Auto-detected Puck textfiles directory: {possible_textfiles_path}")
                GAME_TEXT_FILES_DIR = possible_textfiles_path
                save_config()
                return True
    except Exception as e:
        print(f"[WARN] Error during registry scan for Steam path: {e}")

    print("Auto-detection failed.")

    messagebox.showinfo(
        "Puck Folder Not Found",
        "Could not automatically find the 'Puck' game folder.\n\nPlease select your main 'Puck' game directory in the next window (e.g., C:\\...\\steamapps\\common\\Puck)."
    )

    user_selected_path = filedialog.askdirectory(title="Select your 'Puck' game folder")

    if user_selected_path:
        textfiles_path = os.path.join(user_selected_path, "textfiles")
        if os.path.isdir(textfiles_path):
            print(f"User selected path. Textfiles directory is: {textfiles_path}")
            GAME_TEXT_FILES_DIR = textfiles_path
            save_config()
            return True
        else:
            messagebox.showerror("Invalid Folder", f"The selected folder does not contain a 'textfiles' sub-directory.\n\nPath checked: {textfiles_path}\n\nThe application cannot continue.")
            return False
    else:
        messagebox.showerror("Path Not Provided", "No game folder was selected. The application cannot continue.")
        return False


# --- calculate_absolute_roi, parse_time, get_period_ordinal, parse_period_from_name ---
def calculate_absolute_roi(game_win_roi, relative_roi):
    if not game_win_roi: return None
    if not all(k in relative_roi for k in ('top', 'left', 'width', 'height')):
        print(f"[ERROR] Invalid relative ROI structure: {relative_roi}")
        return None
    return {
        "top": game_win_roi["top"] + relative_roi["top"],
        "left": game_win_roi["left"] + relative_roi["left"],
        "width": relative_roi["width"],
        "height": relative_roi["height"]
    }

def parse_time(time_str):
    """Parse MM:SS string into total seconds. Tolerates extra spaces/chars."""
    if not time_str: return -1
    time_str_cleaned = re.sub(r'[^\d:]', '', time_str)
    match = re.search(r'(\d{1,2}):(\d{2})', time_str_cleaned)
    if match:
        try:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            if 0 <= minutes <= 59 and 0 <= seconds <= 59:
                return minutes * 60 + seconds
            else:
                debug_print(f"Invalid time values after regex: {minutes}:{seconds}")
                return -1
        except ValueError:
            debug_print(f"Time ValueError during conversion: {match.groups()}")
            return -1
    debug_print(f"Time regex mismatch: Input='{time_str}', Cleaned='{time_str_cleaned}'")
    return -1

def get_period_ordinal(period_num):
    """Converts period number (int) to ordinal string (1st, 2nd, etc.)."""
    if period_num <= 0:
        return "?"
    if period_num == 4:
        return "overtime"
    if period_num == 5:
        return "shootout"
    if 11 <= (period_num % 100) <= 13:
        return f"{period_num}th"
    suffixes = {1: 'st', 2: 'nd', 3: 'rd'}
    suffix = suffixes.get(period_num % 10, 'th')
    return f"{period_num}{suffix}"

def parse_period_from_name(period_name_str):
    """Parses period name string (e.g., "1st Period", "Overtime") into a number."""
    if not period_name_str: return -1
    name_upper = period_name_str.upper().replace(" ", "")

    if "1ST" in name_upper or "FIRST" in name_upper: return 1
    if "2ND" in name_upper or "SECOND" in name_upper: return 2
    if "3RD" in name_upper or "THIRD" in name_upper: return 3
    if "OVERTIME" in name_upper or "OT" in name_upper: return 4
    if "SHOOTOUT" in name_upper or "SO" in name_upper: return 5

    match_num = re.search(r'(?:PERIOD|PER)(\d+)', name_upper)
    if match_num:
        try:
            num = int(match_num.group(1))
            if 1 <= num <= 9: return num
        except ValueError: pass

    debug_print(f"Period name parse failed for: '{period_name_str}'")
    return -1

# --- GUI Functions ---
def update_stats_display():
    global stats_text_widget, logged_stats
    if stats_text_widget:
        for item in stats_text_widget.get_children():
            stats_text_widget.delete(item)
        for i, stat_tuple in enumerate(logged_stats):
            if len(stat_tuple) == len(stats_text_widget["columns"]):
                stats_text_widget.insert("", tk.END, iid=i, values=stat_tuple)
            else:
                print(f"[WARN] Mismatch in stat_tuple length: {stat_tuple}")
        stats_text_widget.yview_moveto(1)

def copy_stats_to_clipboard():
    global stats_text_widget, gui_root
    if stats_text_widget and gui_root:
        header = "\t".join([stats_text_widget.heading(col)["text"] for col in stats_text_widget["columns"]])
        content_lines = [header]
        for item_id in stats_text_widget.get_children():
            values = stats_text_widget.item(item_id, "values")
            content_lines.append("\t".join(map(str, values)))

        full_content = "\n".join(content_lines)
        gui_root.clipboard_clear()
        gui_root.clipboard_append(full_content)
        messagebox.showinfo("Copied", "Stats (tab-separated) copied to clipboard!")
    else:
        messagebox.showerror("Error", "Stats display not available.")

def save_stats_to_file():
    global stats_text_widget
    if stats_text_widget:
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Stats As"
        )
        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    header = ",".join([f'"{stats_text_widget.heading(col)["text"]}"' for col in stats_text_widget["columns"]])
                    f.write(header + "\n")
                    for item_id in stats_text_widget.get_children():
                        values = stats_text_widget.item(item_id, "values")
                        formatted_values = [f'"{str(v)}"' for v in values]
                        f.write(",".join(formatted_values) + "\n")
                messagebox.showinfo("Saved", f"Stats saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save stats: {e}")
    else:
        messagebox.showerror("Error", "Stats display not available.")

def update_roi_display_label():
    global roi_display_label, ROI_PERIOD_REL
    if roi_display_label:
        text = (f"Period ROI (Rel to Window): Top={ROI_PERIOD_REL['top']}, Left={ROI_PERIOD_REL['left']}, "
                f"W={ROI_PERIOD_REL['width']}, H={ROI_PERIOD_REL['height']}")
        roi_display_label.config(text=text)

def adjust_period_roi(param, delta):
    global ROI_PERIOD_REL
    if param in ROI_PERIOD_REL:
        ROI_PERIOD_REL[param] += delta
        if param == 'width' and ROI_PERIOD_REL[param] < 10:
            ROI_PERIOD_REL[param] = 10
        if param == 'height' and ROI_PERIOD_REL[param] < 10:
            ROI_PERIOD_REL[param] = 10

        print(f"Adjusted ROI: {param} by {delta}. New ROI_PERIOD_REL: {ROI_PERIOD_REL}")
        update_roi_display_label()
        save_config()

def prompt_for_roi_with_snapshot():
    """
    Takes a snapshot of the game window and opens a new window
    for the user to draw the ROI manually. The snapshot is resized if too large.
    """
    global ROI_PERIOD_REL, gui_root

    MAX_SNAPSHOT_WIDTH = 1280
    MAX_SNAPSHOT_HEIGHT = 720

    print("Attempting to capture game window for manual ROI selection...")
    current_game_roi = get_game_window_roi(GAME_WINDOW_TITLE)
    if not current_game_roi:
        messagebox.showerror("Error", f"Game window '{GAME_WINDOW_TITLE}' not found. Cannot set ROI.")
        return

    with mss.mss() as sct:
        try:
            screenshot_raw = sct.grab(current_game_roi)
            screenshot_img = Image.frombytes("RGB", screenshot_raw.size, screenshot_raw.bgra, "raw", "BGRX")
        except mss.ScreenShotError as e:
            messagebox.showerror("Screenshot Failed", f"Could not capture the game window. Is it obstructed or minimized?\n\nError: {e}")
            return

    original_width, original_height = screenshot_img.size
    scale = 1.0
    if original_width > MAX_SNAPSHOT_WIDTH or original_height > MAX_SNAPSHOT_HEIGHT:
        scale = min(MAX_SNAPSHOT_WIDTH / original_width, MAX_SNAPSHOT_HEIGHT / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        screenshot_img = screenshot_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"Snapshot resized from {original_width}x{original_height} to {new_width}x{new_height} (scale: {scale:.2f})")

    selector_window = tk.Toplevel(gui_root)
    selector_window.title("Draw Period/Goal ROI")
    selector_window.transient(gui_root)
    selector_window.grab_set()

    instruction_label = ttk.Label(selector_window, text="Click and drag to draw a rectangle over the Period/Goal text. Then click 'Confirm'.")
    instruction_label.pack(pady=5)

    canvas = tk.Canvas(selector_window, width=screenshot_img.width, height=screenshot_img.height, cursor="cross")
    canvas.pack()
    tk_photo = ImageTk.PhotoImage(screenshot_img)
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_photo)
    canvas.image = tk_photo

    rect_coords = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
    rect_id = None

    def on_press(event):
        nonlocal rect_id
        rect_coords['x1'] = event.x
        rect_coords['y1'] = event.y
        if rect_id:
            canvas.delete(rect_id)

    def on_motion(event):
        nonlocal rect_id
        rect_coords['x2'] = event.x
        rect_coords['y2'] = event.y
        if rect_id:
            canvas.delete(rect_id)
        rect_id = canvas.create_rectangle(rect_coords['x1'], rect_coords['y1'], rect_coords['x2'], rect_coords['y2'], outline='red', width=2)

    def on_confirm():
        nonlocal rect_coords
        scaled_left = min(rect_coords['x1'], rect_coords['x2'])
        scaled_top = min(rect_coords['y1'], rect_coords['y2'])
        scaled_width = abs(rect_coords['x2'] - rect_coords['x1'])
        scaled_height = abs(rect_coords['y2'] - rect_coords['y1'])

        if scaled_width > 5 and scaled_height > 5:
            original_left = int(scaled_left / scale)
            original_top = int(scaled_top / scale)
            original_width_val = int(scaled_width / scale)
            original_height_val = int(scaled_height / scale)

            ROI_PERIOD_REL['left'] = original_left
            ROI_PERIOD_REL['top'] = original_top
            ROI_PERIOD_REL['width'] = original_width_val
            ROI_PERIOD_REL['height'] = original_height_val

            print(f"Manually set new ROI: {ROI_PERIOD_REL}")
            update_roi_display_label()
            save_config()
            selector_window.destroy()
        else:
            messagebox.showwarning("Invalid ROI", "The selected area is too small. Please try drawing a larger rectangle.", parent=selector_window)

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_motion)

    button_frame = ttk.Frame(selector_window)
    button_frame.pack(fill=tk.X, pady=5, padx=5)

    confirm_btn = ttk.Button(button_frame, text="Confirm ROI", command=on_confirm)
    confirm_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))

    cancel_btn = ttk.Button(button_frame, text="Cancel", command=selector_window.destroy)
    cancel_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(2,0))

    selector_window.update_idletasks()
    main_x = gui_root.winfo_x()
    main_y = gui_root.winfo_y()
    main_w = gui_root.winfo_width()
    main_h = gui_root.winfo_height()
    sel_w = selector_window.winfo_width()
    sel_h = selector_window.winfo_height()
    x = main_x + (main_w // 2) - (sel_w // 2)
    y = main_y + (main_h // 2) - (sel_h // 2)
    selector_window.geometry(f'+{x}+{y}')
    selector_window.wait_window()

def setup_gui():
    global gui_root, stats_text_widget, roi_display_label, debug_image_label
    gui_root = tk.Tk()
    gui_root.title("Puck Stat Logger")
    
    # --- Set window icon ---
    try:
        icon_path = resource_path("icon.ico")
        gui_root.iconbitmap(icon_path)
    except Exception as e:
        print(f"Could not set window icon: {e}")
    # --- End icon setting ---
    
    gui_root.geometry("750x800")

    style = ttk.Style(gui_root)
    main_frame = ttk.Frame(gui_root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    columns = ("period", "time", "team", "scorer", "assists")
    stats_table_frame = ttk.Frame(main_frame)
    stats_table_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP, pady=(0,10))

    stats_text_widget = ttk.Treeview(stats_table_frame, columns=columns, show="headings", selectmode="extended")

    stats_text_widget.heading("period", text="Period")
    stats_text_widget.heading("time", text="Time")
    stats_text_widget.heading("team", text="Team")
    stats_text_widget.heading("scorer", text="Scorer")
    stats_text_widget.heading("assists", text="Assists")

    stats_text_widget.column("period", width=80, anchor=tk.W)
    stats_text_widget.column("time", width=60, anchor=tk.CENTER)
    stats_text_widget.column("team", width=80, anchor=tk.W)
    stats_text_widget.column("scorer", width=150, anchor=tk.W)
    stats_text_widget.column("assists", width=200, anchor=tk.W)

    scrollbar = ttk.Scrollbar(stats_table_frame, orient=tk.VERTICAL, command=stats_text_widget.yview)
    stats_text_widget.configure(yscrollcommand=scrollbar.set)

    stats_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    roi_display_label = ttk.Label(main_frame, text="Period ROI: Initializing...", justify=tk.LEFT)
    roi_display_label.pack(side=tk.TOP, fill=tk.X, pady=(5,5))
    update_roi_display_label()

    roi_controls_frame = ttk.Frame(main_frame)
    roi_controls_frame.pack(fill=tk.X, side=tk.TOP, pady=(0,10))
    for i in range(6):
        roi_controls_frame.columnconfigure(i, weight=1, uniform="roi_group")

    ttk.Label(roi_controls_frame, text="Top:").grid(row=0, column=0, sticky=tk.W, pady=2)
    ttk.Button(roi_controls_frame, text="-", width=3, command=lambda: adjust_period_roi('top', -ROI_ADJUSTMENT_STEP)).grid(row=0, column=1, sticky=tk.E, padx=2, pady=2)
    ttk.Button(roi_controls_frame, text="+", width=3, command=lambda: adjust_period_roi('top', ROI_ADJUSTMENT_STEP)).grid(row=0, column=2, sticky=tk.W, padx=2, pady=2)

    ttk.Label(roi_controls_frame, text="Left:").grid(row=0, column=3, sticky=tk.W, padx=(10,0), pady=2)
    ttk.Button(roi_controls_frame, text="-", width=3, command=lambda: adjust_period_roi('left', -ROI_ADJUSTMENT_STEP)).grid(row=0, column=4, sticky=tk.E, padx=2, pady=2)
    ttk.Button(roi_controls_frame, text="+", width=3, command=lambda: adjust_period_roi('left', ROI_ADJUSTMENT_STEP)).grid(row=0, column=5, sticky=tk.W, padx=2, pady=2)

    ttk.Label(roi_controls_frame, text="Width:").grid(row=1, column=0, sticky=tk.W, pady=2)
    ttk.Button(roi_controls_frame, text="-", width=3, command=lambda: adjust_period_roi('width', -ROI_ADJUSTMENT_STEP)).grid(row=1, column=1, sticky=tk.E, padx=2, pady=2)
    ttk.Button(roi_controls_frame, text="+", width=3, command=lambda: adjust_period_roi('width', ROI_ADJUSTMENT_STEP)).grid(row=1, column=2, sticky=tk.W, padx=2, pady=2)

    ttk.Label(roi_controls_frame, text="Height:").grid(row=1, column=3, sticky=tk.W, padx=(10,0), pady=2)
    ttk.Button(roi_controls_frame, text="-", width=3, command=lambda: adjust_period_roi('height', -ROI_ADJUSTMENT_STEP)).grid(row=1, column=4, sticky=tk.E, padx=2, pady=2)
    ttk.Button(roi_controls_frame, text="+", width=3, command=lambda: adjust_period_roi('height', ROI_ADJUSTMENT_STEP)).grid(row=1, column=5, sticky=tk.W, padx=2, pady=2)

    debug_image_label = ttk.Label(main_frame)
    debug_image_label.pack(side=tk.TOP, pady=10)

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10,0))

    copy_button = ttk.Button(button_frame, text="Copy to Clipboard", command=copy_stats_to_clipboard)
    copy_button.pack(side=tk.LEFT, padx=(0, 5), expand=True, fill=tk.X)

    save_button = ttk.Button(button_frame, text="Save to File", command=save_stats_to_file)
    save_button.pack(side=tk.LEFT, padx=(5, 0), expand=True, fill=tk.X)

    set_roi_button = ttk.Button(button_frame, text="Set Period ROI Manually", command=prompt_for_roi_with_snapshot)
    set_roi_button.pack(side=tk.LEFT, padx=(5,0), expand=True, fill=tk.X)

    gui_root.protocol("WM_DELETE_WINDOW", on_closing_gui)

def on_closing_gui():
    global gui_root
    if messagebox.askokcancel("Quit", "Do you want to quit the Stat Logger?"):
        if gui_root:
            gui_root.quit()
            gui_root.destroy()

        print("GUI closed by user. Exiting script.")
        os._exit(0)

# --- create_debug_canvas function ---
def create_debug_canvas(images_dict, canvas_width=800):
    global ROI_PERIOD_REL

    period_data = images_dict.get('Period', (None, None, "Period ROI Not Captured"))
    raw_img, processed_img, ocr_text = period_data

    display_img = None
    img_h, img_w = 0, 0

    if raw_img is not None and raw_img.size > 0:
        try:
            if len(raw_img.shape) == 3 and raw_img.shape[2] == 4:
                display_img = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2BGR)
            elif len(raw_img.shape) == 3 and raw_img.shape[2] == 3:
                display_img = raw_img
            elif len(raw_img.shape) == 2:
                display_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)

            if display_img is not None:
                img_h, img_w = display_img.shape[:2]
        except Exception as e:
            print(f"Error preparing period image for debug: {e}")
            display_img = None

    max_debug_img_w = 300
    max_debug_img_h = 150
    scaled_display_img = None

    if display_img is not None and img_w > 0 and img_h > 0:
        scale = min(max_debug_img_w / img_w, max_debug_img_h / img_h)
        if scale > 0:
            scaled_display_img = cv2.resize(display_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            img_h, img_w = scaled_display_img.shape[:2]
        else:
            scaled_display_img = display_img

    if scaled_display_img is None:
        img_h, img_w = max_debug_img_h, max_debug_img_w

    text_area_h = 80
    canvas_h = img_h + text_area_h
    canvas_w = img_w if img_w > 0 else max_debug_img_w

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    line_type = cv2.LINE_AA

    if scaled_display_img is not None:
        canvas[0:img_h, 0:img_w] = scaled_display_img
    else:
        cv2.putText(canvas, "No Period Image", (10, img_h // 2), font, font_scale, color, thickness, line_type)

    y_text_offset = img_h + 20

    roi_info_lines = [
        f"Period ROI (Rel):",
        f" T={ROI_PERIOD_REL['top']}, L={ROI_PERIOD_REL['left']}",
        f" W={ROI_PERIOD_REL['width']}, H={ROI_PERIOD_REL['height']}",
        f"OCR: {ocr_text if ocr_text else 'N/A'}"
    ]

    for line in roi_info_lines:
        if y_text_offset + 15 < canvas_h:
            cv2.putText(canvas, line, (10, y_text_offset), font, font_scale, color, thickness, line_type)
            y_text_offset += 18
        else:
            break

    return canvas

# --- Main Loop ---
def main():
    global last_known_good_period, last_time_seconds, last_known_good_period_name
    global easyocr_reader, game_window_roi
    global last_window_check_time, last_period_roi_state, last_logged_stat_for_debug
    global gui_root, logged_stats

    setup_gui()

    try:
        import easyocr
        import torch
    except ModuleNotFoundError as e:
        error_msg = (
            f"A required module was not found: {e.name}\n\n"
            "This can happen when running the compiled .exe if the module was not included by PyInstaller.\n\n"
            "Please ensure you have installed the required packages, especially 'easyocr' and its dependencies ('torch', etc.), in your environment before compiling.\n\n"
            "The application will now close."
        )
        print(f"[FATAL] {error_msg}")
        messagebox.showerror("Fatal Error: Missing Module", error_msg)
        if gui_root:
            gui_root.destroy()
        return

    print("Initializing Stat Logger (using EasyOCR)...")
    load_config()
    update_roi_display_label()

    if not find_and_set_game_path():
        print("Could not determine game path. Exiting.")
        if gui_root:
            try:
                gui_root.destroy()
            except tk.TclError:
                pass
        return

    try:
        if torch.cuda.is_available():
            if not USE_GPU_IF_AVAILABLE:
                messagebox.showwarning(
                    "Performance & Size Warning",
                    "A GPU-enabled version of PyTorch was detected, but GPU usage is disabled in the script (USE_GPU_IF_AVAILABLE = False).\n\n"
                    "The compiled .exe will be very large. For a smaller file size, please uninstall PyTorch and reinstall the CPU-only version before compiling:\n\n"
                    "pip uninstall torch torchvision torchaudio\n"
                    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
                )
            else:
                messagebox.showinfo(
                    "GPU Detected",
                    "A GPU-enabled version of PyTorch was detected and will be used.\n\n"
                    "Note: This will result in a very large compiled .exe file."
                )

        print("Loading EasyOCR model...")
        use_gpu = USE_GPU_IF_AVAILABLE and torch.cuda.is_available()
        easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu, model_storage_directory=EASYOCR_MODEL_DIR)
        print(f"EasyOCR model loaded. Using GPU: {use_gpu}")
    except Exception as e:
        print(f"Fatal Error loading EasyOCR: {e}")
        traceback.print_exc()
        if gui_root:
            gui_root.destroy()
        return

    print(f"Monitoring '{GAME_WINDOW_TITLE}'. Press Ctrl+C or 'q' in debug window to stop.")
    print("---", flush=True)

    loop_count = 0
    debug_print("[Debug] Preparing to enter MSS context manager...", flush=True)

    last_known_good_period = 0
    last_known_good_period_name = ""

    with mss.mss() as sct:
        while True:
            if gui_root is None or not tk._default_root:
                print("GUI window closed or not available. Exiting main loop.")
                break

            debug_print(f"\n--- Loop Start {loop_count} ---", flush=True)
            debug_images = {}

            try:
                current_time_ms = time.time()

                if game_window_roi is None or (current_time_ms - last_window_check_time > RECHECK_WINDOW_INTERVAL_SECONDS):
                    found_roi = get_game_window_roi(GAME_WINDOW_TITLE)
                    last_window_check_time = current_time_ms
                    if found_roi:
                        if game_window_roi != found_roi:
                            debug_print(f"Game window '{GAME_WINDOW_TITLE}' found/updated at: {found_roi}", flush=True)
                            game_window_roi = found_roi
                        elif game_window_roi is None:
                            game_window_roi = found_roi
                    else:
                        if game_window_roi is not None:
                            print(f"Game window '{GAME_WINDOW_TITLE}' lost. Searching...", flush=True)
                            game_window_roi = None
                        debug_print("[Debug] Window not found this check.", flush=True)
                        time.sleep(1.0)
                        continue

                if game_window_roi is None:
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                time_abs_roi = calculate_absolute_roi(game_window_roi, ROI_TIME_REL)
                period_abs_roi = calculate_absolute_roi(game_window_roi, ROI_PERIOD_REL)

                if not time_abs_roi or not period_abs_roi:
                      print("[WARN] One or more absolute ROIs could not be calculated. Skipping OCR this cycle.")
                      time.sleep(POLL_INTERVAL_SECONDS)
                      continue

                time_str, time_img_raw, time_img_ocr = ocr_region(sct, time_abs_roi, "time_digits")
                period_roi_text_raw, period_img_raw, period_img_ocr = ocr_region(sct, period_abs_roi, "period_or_goal")

                current_period_roi_state = "OTHER"
                if "GOAL" in period_roi_text_raw.upper().replace(" ", ""):
                      current_period_roi_state = "GOAL"
                debug_print(f"Period ROI OCR Text: '{period_roi_text_raw}', Detected State: {current_period_roi_state}")

                current_period_name_from_file = read_game_text_file(PERIOD_NAME_FILE)
                parsed_period_num_from_file = parse_period_from_name(current_period_name_from_file)

                period_debug_text_ocr = period_roi_text_raw
                debug_images['Period'] = (period_img_raw, period_img_ocr, period_debug_text_ocr)

                current_time_seconds = parse_time(time_str)
                debug_print(f"Parsed Time: {current_time_seconds}s (Raw: '{time_str}')")

                period_changed_from_file = False
                if current_period_name_from_file and current_period_name_from_file != last_known_good_period_name:
                    if parsed_period_num_from_file != -1:
                        print(f"--- Period Name Changed (File): '{last_known_good_period_name}' -> '{current_period_name_from_file}' (Parsed as: {parsed_period_num_from_file}) ---")
                        if last_known_good_period > 0 and parsed_period_num_from_file < last_known_good_period and parsed_period_num_from_file != -1 :
                            debug_print(f"Period number decreased ({last_known_good_period} -> {parsed_period_num_from_file}). Assuming game reset or new match.")
                        last_known_good_period = parsed_period_num_from_file
                        period_changed_from_file = True
                    else:
                        debug_print(f"Period name from file changed to '{current_period_name_from_file}', but could not parse to a known period number. Keeping period number {last_known_good_period}.")
                    last_known_good_period_name = current_period_name_from_file

                current_display_period = last_known_good_period

                goal_just_appeared = (current_period_roi_state == "GOAL" and last_period_roi_state != "GOAL")

                if goal_just_appeared:
                    print("Goal trigger detected ('GOAL' appeared). Logging stat...")

                    goal_team = read_game_text_file(GOAL_TEAM_FILE)
                    scorer_name = read_game_text_file(GOAL_SCORER_FILE)
                    assister1_name = read_game_text_file(GOAL_ASSISTER_FILE)
                    assister2_name = read_game_text_file(GOAL_ASSISTER2_FILE)
                    goal_time_clock_raw = read_game_text_file(CLOCK_FILE)

                    goal_time_clock = goal_time_clock_raw
                    if goal_time_clock_raw and goal_time_clock_raw.startswith('0') and len(goal_time_clock_raw) > 1 and goal_time_clock_raw[1].isdigit():
                        goal_time_clock = goal_time_clock_raw[1:]

                    debug_print(f"Goal Info from Files: Team='{goal_team}', Scorer='{scorer_name}', Assist1='{assister1_name}', Assist2='{assister2_name}', Clock='{goal_time_clock}' (Raw: '{goal_time_clock_raw}'), PeriodFile='{current_period_name_from_file}'")

                    if scorer_name:
                        period_for_log = current_period_name_from_file if current_period_name_from_file else get_period_ordinal(last_known_good_period)
                        time_for_log = goal_time_clock if goal_time_clock else "Unknown Time"
                        team_for_log = goal_team.capitalize() if goal_team else "N/A"

                        assisters_list = []
                        if assister1_name: assisters_list.append(assister1_name)
                        if assister2_name and assister2_name != assister1_name: assisters_list.append(assister2_name)
                        assist_str_log = ", ".join(assisters_list) if assisters_list else "Unassisted"

                        logged_stat_tuple = (period_for_log, time_for_log, team_for_log, scorer_name, assist_str_log)

                        if not logged_stats or logged_stats[-1] != logged_stat_tuple:
                            logged_stats.append(logged_stat_tuple)
                            last_logged_stat_for_debug = f"Logged: {scorer_name}"
                            print(f"Logged Stat: {logged_stat_tuple}")
                            update_stats_display()
                        else:
                            debug_print(f"Duplicate goal event detected. Ignoring: {logged_stat_tuple}")
                            last_logged_stat_for_debug = "Duplicate goal ignored"
                    else:
                        debug_print("Goal triggered, but no scorer name found in text files. Not logging.")
                        last_logged_stat_for_debug = "Goal detected (No scorer in file)"

                last_period_roi_state = current_period_roi_state

                if current_period_roi_state != "GOAL" and not goal_just_appeared:
                      if loop_count % 50 == 0:
                          last_logged_stat_for_debug = ""

                if current_time_seconds != -1:
                    last_time_seconds = current_time_seconds

                if debug_image_label:
                    try:
                        debug_canvas = create_debug_canvas(debug_images)
                        if debug_canvas is not None and debug_canvas.size > 0:
                            img_rgb = cv2.cvtColor(debug_canvas, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(img_rgb)
                            photo_img = ImageTk.PhotoImage(image=pil_img)
                            debug_image_label.config(image=photo_img)
                            debug_image_label.image = photo_img
                    except Exception as debug_e:
                        print(f"Error updating debug image in GUI: {debug_e}")

                if gui_root:
                    try:
                        gui_root.update_idletasks()
                        gui_root.update()
                    except tk.TclError as e:
                        print(f"Tkinter error during update: {e}. GUI might have been closed.")
                        break

                loop_end_time = time.time()
                elapsed = loop_end_time - current_time_ms
                sleep_time = POLL_INTERVAL_SECONDS - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    debug_print(f"Warning: Loop took longer ({elapsed:.3f}s) than poll interval ({POLL_INTERVAL_SECONDS}s)")

                debug_print(f"--- Loop End {loop_count} ---", flush=True)
                loop_count += 1

            except KeyboardInterrupt:
                print("\nStopping (Ctrl+C).")
                break
            except Exception as e:
                if "application has been destroyed" in str(e).lower():
                    print("GUI was closed. Exiting.")
                    break
                print(f"\n--- Error in Main Loop ---")
                traceback.print_exc()
                print("-------------------------")
                print("Pausing for 5 seconds before retrying...")
                time.sleep(5)

    print("Cleaning up...")

    if gui_root:
        try:
            gui_root.destroy()
        except tk.TclError:
            pass
    print("Stat Logger stopped.")


if __name__ == "__main__":
    main()
