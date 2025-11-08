# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an automated video editing pipeline for YouTube content creation, specifically designed for music album videos. The system processes audio files, creates visual effects in Adobe After Effects, edits in Adobe Premiere Pro, and automatically uploads to YouTube. The entire workflow is automated using PyAutoGUI for GUI automation and various Python libraries for media processing.

## Environment Setup

### Dependencies Installation

Run the PowerShell setup script:
```powershell
.\pip_install.ps1
```

This creates a virtual environment and installs required packages:
- **GUI Automation**: `pyautogui`, `pynput`, `keyboard`
- **Media Processing**: `pydub`, `pillow`, `scipy`, `eyed3`, `mutagen`
- **Web Automation**: `playwright` (recently migrated from Selenium), `beautifulsoup4`, `requests`
- **Utilities**: `numpy`, `unidecode`, `colormath`, `psutil`

### External Dependencies

- **FFmpeg**: Must be installed at `C:\Program Files\FFmpeg\bin\`
- **Adobe After Effects**: Required for script execution (5. auto_effects.py)
- **Adobe Premiere Pro**: Required for video editing (6. auto_premier.py)
- **Adobe Media Encoder**: Must be running before executing 6. auto_premier.py

## Project Architecture

### Pipeline Overview

The system follows a numbered sequential pipeline (0-11), where each script performs a specific task. Scripts are designed to be run in order, with each step preparing data for the next:

```
Input: Raw audio files + cover images
  ↓
0. limpieza_de_impurezas.py (Cleanup)
  ↓
1. juntar_audios.py (Merge audio + metadata extraction + web scraping)
  ↓
2. cantidad_de_archivos.py (Verification)
  ↓
3. cambiar_nombre_imagen.py (Image processing)
  ↓
4. verificacion_humana.py (Manual verification)
  ↓
5. auto_effects.py (After Effects automation)
  ↓
6. auto_premier.py (Premiere Pro automation)
  ↓
7. mover_videos_terminados.py (File organization)
  ↓
8. subir_video_coordenadas.py (YouTube upload with coordinates)
  ↓
9-11. Additional upload/appeal scripts
  ↓
Output: Published YouTube videos
```

### Key Design Patterns

**1. Coordinate-Based GUI Automation**

All Adobe software automation uses hardcoded PyAutoGUI coordinates. This is extremely fragile:
- Screen resolution must be exactly as specified in scripts
- Window positions must match exactly
- Any UI movement breaks the automation

Example from 5. auto_effects.py:712:
```python
pyautogui.click(2000, 193)  # New Project button
pyautogui.click(3020, 397)  # New Composition with Footage
```

**CRITICAL**: When modifying automation scripts, even a 1-pixel movement in After Effects can break everything (see README warning at line 40-42).

**2. ExtendScript (.jsx) Integration**

After Effects automation executes ExtendScript files for visual effects:
- `audio_to_keyframes.jsx`: Generates keyframes from audio waveform
- `imagen_movimiento.jsx`: Adds wiggle rotation and audio-reactive scaling to images (Affter Effects/imagen_movimiento.jsx:1-24)
- `espectro_de_audio.jsx`: Creates audio spectrum visualization with complementary colors (Affter Effects/espectro_de_audio.jsx:1-21)

These are triggered via PyAutoGUI clicking File > Scripts > Run Script File.

**3. Color Analysis Pipeline**

The system analyzes album cover colors and applies complementary colors to audio spectrum:
1. Extract average RGB from album cover (5. auto_effects.py:94-104)
2. Convert RGB → HSV → rotate hue 180° → convert back to RGB (5. auto_effects.py:26-70)
3. Write complementary color to After Effects spectrum effect (5. auto_effects.py:210-220)

**4. Metadata Extraction with Web Scraping**

Script 1. juntar_audios.py combines audio processing with automated web scraping using Playwright:
- Extracts ID3 tags (artist, album, year, genre) from audio files
- Scrapes DuckDuckGo for music platform links (Bandcamp, Spotify, Apple Music, Deezer, Amazon, YouTube Music)
- Scrapes social media links (Facebook, Instagram, YouTube, TikTok, Twitter)
- Scrapes metal database links (Metal Archives, Spirit of Metal)
- Generates video description with all links (1. juntar_audios.py:287-348)

Note: Recently migrated from Selenium to Playwright (see git commit da21d3c).

**5. Process Monitoring**

6. auto_premier.py includes process monitoring to ensure sequential execution:
```python
def is_premier_running():
    """Verifies if Adobe Premier Pro is running"""
    # Checks psutil for Adobe Premiere Pro.exe
```

This prevents pipeline conflicts when rendering multiple videos.

### Directory Structure Conventions

The scripts expect specific Windows directory paths:
- **Input**: `E:\01_edicion_automatizada\audio_scripts` (raw audio folders)
- **After Effects Projects**: `C:\Users\banar\Desktop\save_after_effects`
- **Premiere Pro Projects**: `C:\Users\banar\Desktop\save_premier_pro`
- **Render Output**: `E:\01_edicion_automatizada\audio_scripts` (same as input)
- **Upload Queue**: `E:\01_edicion_automatizada\upload_video`
- **JSX Scripts**: `C:\Users\banar\Desktop\click-auto-editor\Affter Effects\`

### File Tracking

The system maintains persistent state:
- `bandas-subidas-al-canal.txt`: Tracks uploaded video titles to prevent duplicates (8. subir_video_coordenadas.py:16-19)

## Running Scripts

### Individual Script Execution

Each script is standalone:
```bash
python "0. limpieza_de_impurezas.py"
python "1. juntar_audios.py"
python "2. cantidad_de_archivos.py"
# etc.
```

### Critical Execution Notes

1. **5. auto_effects.py**: Processes up to 150 random folders (5. auto_effects.py:24). Adjust if needed.

2. **6. auto_premier.py**: Adobe Media Encoder MUST be running before execution (README line 48-50).

3. **8. subir_video_coordenadas.py**: Interactive - prompts for number of videos to upload (8. subir_video_coordenadas.py:26).

4. **Timing Delays**: All scripts use `time.sleep()` extensively. These are calibrated for system performance and should not be reduced without testing.

## Modifying Coordinate-Based Automation

When working with 5. auto_effects.py, 6. auto_premier.py, or 8. subir_video_coordenadas.py:

1. Use `coordenadas.py` to capture new screen coordinates:
```python
# Prints mouse position to console
import pyautogui
import time
while True:
    print(pyautogui.position())
    time.sleep(1)
```

2. Document coordinate changes with comments explaining what UI element is being clicked

3. Test thoroughly - coordinate changes can cascade failures

## Video Specifications

- **Resolution**: Originally 4K, now 1080p (see git commit 99c52bb)
- **Audio**: 320kbps MP3
- **Intro Duration**: 8 seconds (hardcoded in tracklist timing at 1. juntar_audios.py:340)
- **Transition Effect**: VHS transition between intro and main content (6. auto_premier.py:138-150)
- **Premiere Pro Export Settings**: Configured for 1080p render to Adobe Media Encoder

## Recent Changes

- Migrated from Selenium to Playwright for web scraping (git commit da21d3c)
- Reduced spectrum visualization to 50% size (git commit 783389a)
- Resolution downgrade from 4K to 1080p (git commit 99c52bb)
- Increased Premiere Pro processing time (git commit 2ab23db)

## Common Issues

1. **"PermissionError: The folder is currently in use"**: Another process (often Windows Explorer or antivirus) is accessing the folder. Scripts handle this with try/catch but may skip folders.

2. **Coordinate Misalignment**: If Adobe apps update UI or window positions change, all coordinates must be recaptured.

3. **Audio Spectrum Not Matching Colors**: Ensure PNG image exists before JSX script execution. Script expects .png format specifically (5. auto_effects.py:90).

4. **Upload Verification Failures**: Check `bandas-subidas-al-canal.txt` format - expects exact title match without the last 13 characters (8. subir_video_coordenadas.py:73).
