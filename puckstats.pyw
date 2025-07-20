import os
import sys
import json
import winreg
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# --- Configuration ---
CONFIG_FILE = "config.json"

# --- Game Text File Configuration ---
# This will be auto-detected or loaded from config. It should point to the 'textfiles' directory.
GAME_TEXT_FILES_DIR = "" 
GOAL_TEAM_FILE = "goal_team.txt"
GOAL_SCORER_FILE = "goal_scorer.txt"
GOAL_ASSISTER_FILE = "goal_assister.txt"
GOAL_ASSISTER2_FILE = "goal_assister2.txt"
CLOCK_FILE = "realclock.txt"
PERIOD_NAME_FILE = "period_name.txt"
SCORE_RED_FILE = "scorered.txt"
SCORE_BLUE_FILE = "scoreblue.txt"
# --- End Game Text File Configuration ---

# General Settings
POLL_INTERVAL_SECONDS = 0.25 # How often to check the files
DEBUG_ENABLED = False

# --- Globals ---
logged_stats = []
stats_text_widget = None
gui_root = None
directory_display_label = None
# These will store the last known scores to detect an increase.
last_known_score_red = -1
last_known_score_blue = -1

# --- Helper Functions ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def debug_print(*args, **kwargs):
    if DEBUG_ENABLED:
        print("[Debug]", *args, **kwargs)

# --- File Reading Function ---
def read_game_text_file(filename):
    """Reads a text file from the game's textfiles directory."""
    if not GAME_TEXT_FILES_DIR or not filename:
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
    """Saves the game path to the config file."""
    global GAME_TEXT_FILES_DIR
    config_data = {
        "GAME_TEXT_FILES_DIR": GAME_TEXT_FILES_DIR
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)
        debug_print(f"Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to save configuration: {e}")

def load_config():
    """Loads the game path from the config file if it exists."""
    global GAME_TEXT_FILES_DIR
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config_data = json.load(f)
            
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
    global GAME_TEXT_FILES_DIR
    if GAME_TEXT_FILES_DIR and os.path.isdir(GAME_TEXT_FILES_DIR):
        print(f"Using game text files path from config: {GAME_TEXT_FILES_DIR}")
        return True

    print("Attempting to auto-detect Puck game folder...")
    try:
        steam_path = ""
        # Look in both possible registry locations for the Steam path
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
            # Construct the expected path to the 'textfiles' directory
            possible_textfiles_path = os.path.join(steam_path, "steamapps", "common", "Puck", "textfiles")
            if os.path.isdir(possible_textfiles_path):
                print(f"Auto-detected Puck textfiles directory: {possible_textfiles_path}")
                GAME_TEXT_FILES_DIR = possible_textfiles_path
                save_config()
                return True
    except Exception as e:
        print(f"[WARN] Error during registry scan for Steam path: {e}")

    print("Auto-detection failed.")
    
    # Prompt user to manually select the folder
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

# --- GUI Functions ---
def update_stats_display():
    """Refreshes the stats table in the GUI with the latest logged data."""
    global stats_text_widget, logged_stats
    if stats_text_widget:
        # Clear existing entries
        for item in stats_text_widget.get_children():
            stats_text_widget.delete(item)
        # Insert new entries
        for i, stat_tuple in enumerate(logged_stats):
            if len(stat_tuple) == len(stats_text_widget["columns"]):
                stats_text_widget.insert("", tk.END, iid=i, values=stat_tuple)
            else:
                print(f"[WARN] Mismatch in stat_tuple length: {stat_tuple}")
        # Auto-scroll to the bottom
        stats_text_widget.yview_moveto(1)

def copy_stats_to_clipboard():
    """Copies the content of the stats table to the clipboard in a tab-separated format."""
    global stats_text_widget, gui_root
    if not stats_text_widget or not gui_root:
        messagebox.showerror("Error", "Stats display not available.")
        return

    header = "\t".join([stats_text_widget.heading(col)["text"] for col in stats_text_widget["columns"]])
    content_lines = [header]
    for item_id in stats_text_widget.get_children():
        values = stats_text_widget.item(item_id, "values")
        content_lines.append("\t".join(map(str, values)))
    
    full_content = "\n".join(content_lines)
    gui_root.clipboard_clear()
    gui_root.clipboard_append(full_content)
    messagebox.showinfo("Copied", "Stats (tab-separated) copied to clipboard!")

def save_stats_to_file():
    """Opens a dialog to save the stats table to a CSV file."""
    global stats_text_widget
    if not stats_text_widget:
        messagebox.showerror("Error", "Stats display not available.")
        return
        
    filepath = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")],
        title="Save Stats As"
    )
    if filepath:
        try:
            with open(filepath, "w", encoding="utf-8", newline='') as f:
                header = ",".join([f'"{stats_text_widget.heading(col)["text"]}"' for col in stats_text_widget["columns"]])
                f.write(header + "\n")
                for item_id in stats_text_widget.get_children():
                    values = stats_text_widget.item(item_id, "values")
                    # Enclose each value in quotes to handle commas within fields
                    formatted_values = [f'"{str(v)}"' for v in values]
                    f.write(",".join(formatted_values) + "\n")
            messagebox.showinfo("Saved", f"Stats saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save stats: {e}")

def update_directory_display():
    """Updates the GUI label that shows the current game textfiles directory."""
    global directory_display_label, GAME_TEXT_FILES_DIR
    if directory_display_label:
        path_text = GAME_TEXT_FILES_DIR if GAME_TEXT_FILES_DIR else "Not Set"
        directory_display_label.config(text=f"Current Path: {path_text}")

def prompt_to_change_directory():
    """Prompts the user to select a new game directory."""
    global GAME_TEXT_FILES_DIR
    messagebox.showinfo(
        "Select Game Folder",
        "Please select your main 'Puck' game directory in the next window (e.g., C:\\...\\steamapps\\common\\Puck)."
    )
    
    user_selected_path = filedialog.askdirectory(title="Select your 'Puck' game folder")

    if user_selected_path:
        textfiles_path = os.path.join(user_selected_path, "textfiles")
        if os.path.isdir(textfiles_path):
            print(f"User changed path. New textfiles directory is: {textfiles_path}")
            GAME_TEXT_FILES_DIR = textfiles_path
            save_config()
            update_directory_display() # Update the label
            messagebox.showinfo("Success", f"Game directory updated to:\n{textfiles_path}")
        else:
            messagebox.showerror("Invalid Folder", f"The selected folder does not contain a 'textfiles' sub-directory.\n\nPath checked: {textfiles_path}", parent=gui_root)
    else:
        print("User cancelled directory change.")

def setup_gui():
    """Initializes and configures the main application window and its widgets."""
    global gui_root, stats_text_widget, directory_display_label
    gui_root = tk.Tk()
    gui_root.title("Puck Stat Logger (File Monitor)")
    
    # Set window icon if available
    try:
        icon_path = resource_path("icon.ico")
        if os.path.exists(icon_path):
            gui_root.iconbitmap(icon_path)
    except Exception as e:
        print(f"Could not set window icon: {e}")
        
    gui_root.geometry("750x650") # Increased height slightly for new widget

    main_frame = ttk.Frame(gui_root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # --- Stats Table ---
    stats_table_frame = ttk.Frame(main_frame)
    stats_table_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP, pady=(0,10))

    stats_text_widget = ttk.Treeview(stats_table_frame, columns=("period", "time", "team", "scorer", "assists"), show="headings", selectmode="extended")

    stats_text_widget.heading("period", text="Period")
    stats_text_widget.heading("time", text="Time")
    stats_text_widget.heading("team", text="Team")
    stats_text_widget.heading("scorer", text="Scorer")
    stats_text_widget.heading("assists", text="Assists")

    stats_text_widget.column("period", width=100, anchor=tk.W)
    stats_text_widget.column("time", width=80, anchor=tk.CENTER)
    stats_text_widget.column("team", width=100, anchor=tk.W)
    stats_text_widget.column("scorer", width=150, anchor=tk.W)
    stats_text_widget.column("assists", width=250, anchor=tk.W)

    scrollbar = ttk.Scrollbar(stats_table_frame, orient=tk.VERTICAL, command=stats_text_widget.yview)
    stats_text_widget.configure(yscrollcommand=scrollbar.set)

    stats_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # --- Directory Display and Control Frame ---
    dir_frame = ttk.Labelframe(main_frame, text="Game Text Files Directory", padding="5")
    dir_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 5))

    directory_display_label = ttk.Label(dir_frame, text="Current Path: Initializing...", wraplength=550, justify=tk.LEFT)
    directory_display_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

    change_dir_button = ttk.Button(dir_frame, text="Change...", command=prompt_to_change_directory)
    change_dir_button.pack(side=tk.RIGHT, padx=5, pady=5)

    # --- Control Buttons ---
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))

    copy_button = ttk.Button(button_frame, text="Copy to Clipboard", command=copy_stats_to_clipboard)
    copy_button.pack(side=tk.LEFT, padx=(0, 5), expand=True, fill=tk.X)

    save_button = ttk.Button(button_frame, text="Save to File", command=save_stats_to_file)
    save_button.pack(side=tk.LEFT, padx=(5, 0), expand=True, fill=tk.X)

    gui_root.protocol("WM_DELETE_WINDOW", on_closing_gui)

def on_closing_gui():
    """Handles the event when the user closes the GUI window."""
    global gui_root
    if messagebox.askokcancel("Quit", "Do you want to quit the Stat Logger?"):
        if gui_root:
            gui_root.quit()
            gui_root.destroy()
        print("GUI closed by user. Exiting script.")
        # Use os._exit to ensure the script terminates even if the main loop is sleeping
        os._exit(0)

# --- Main Loop ---
def main():
    global gui_root, logged_stats
    global last_known_score_red, last_known_score_blue

    setup_gui()
    
    print("Initializing Score-Based Stat Logger...")
    load_config()

    if not find_and_set_game_path():
        print("Could not determine game path. Exiting.")
        if gui_root:
            try:
                gui_root.destroy()
            except tk.TclError:
                pass
        return

    update_directory_display()

    print(f"Monitoring score files in: {GAME_TEXT_FILES_DIR}")
    print("---", flush=True)

    while True:
        try:
            # Check if the GUI window has been closed
            if gui_root is None or not tk._default_root:
                print("GUI window not available. Exiting main loop.")
                break

            # Read the current scores from the text files.
            score_red_str = read_game_text_file(SCORE_RED_FILE)
            score_blue_str = read_game_text_file(SCORE_BLUE_FILE)

            # Convert score strings to integers, defaulting to 0 if empty or invalid.
            try:
                current_score_red = int(score_red_str) if score_red_str else 0
            except (ValueError, TypeError):
                current_score_red = 0

            try:
                current_score_blue = int(score_blue_str) if score_blue_str else 0
            except (ValueError, TypeError):
                current_score_blue = 0
            
            # Initialize the scores on the first run.
            if last_known_score_red == -1:
                last_known_score_red = current_score_red
                debug_print(f"Initial Red score set to: {last_known_score_red}")
            if last_known_score_blue == -1:
                last_known_score_blue = current_score_blue
                debug_print(f"Initial Blue score set to: {last_known_score_blue}")

            # Check if either score has increased. This is the new goal trigger.
            if current_score_red > last_known_score_red or current_score_blue > last_known_score_blue:
                print("Score increase detected! Logging goal event...")
                
                # Add a small delay to give the game time to write all the goal info files.
                time.sleep(0.1)

                # A goal was scored, so we gather all related information.
                goal_team = read_game_text_file(GOAL_TEAM_FILE)
                scorer = read_game_text_file(GOAL_SCORER_FILE)
                assister1 = read_game_text_file(GOAL_ASSISTER_FILE)
                assister2 = read_game_text_file(GOAL_ASSISTER2_FILE)
                goal_time = read_game_text_file(CLOCK_FILE)
                period_name = read_game_text_file(PERIOD_NAME_FILE)

                # Format the data for logging
                period_for_log = period_name if period_name else "Unknown Period"
                time_for_log = goal_time if goal_time else "Unknown Time"
                team_for_log = goal_team.capitalize() if goal_team else "N/A"
                
                assisters_list = []
                if assister1: assisters_list.append(assister1)
                # Ensure the second assister is not the same as the first
                if assister2 and assister2 != assister1: assisters_list.append(assister2)
                assist_str_log = ", ".join(assisters_list) if assisters_list else "Unassisted"
                
                # Create the final tuple for the log
                logged_stat_tuple = (period_for_log, time_for_log, team_for_log, scorer, assist_str_log)
                
                print(f"Logged Stat: {logged_stat_tuple}")
                logged_stats.append(logged_stat_tuple)
                
                # Update the display in the GUI
                update_stats_display()

                # IMPORTANT: Update the last known scores to the current ones.
                last_known_score_red = current_score_red
                last_known_score_blue = current_score_blue
            
            # NEW: Detect a game reset (score has decreased)
            elif current_score_red < last_known_score_red or current_score_blue < last_known_score_blue:
                print("Game reset detected (score decreased). Updating score tracking.")
                last_known_score_red = current_score_red
                last_known_score_blue = current_score_blue


            # Keep the GUI responsive
            if gui_root:
                gui_root.update_idletasks()
                gui_root.update()

            # Wait before checking the files again
            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("\nStopping (Ctrl+C).")
            break
        except Exception as e:
            # Handle cases where the GUI is closed unexpectedly
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
