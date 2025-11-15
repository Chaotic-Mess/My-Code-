# Storage Scanner

A beautiful, interactive storage analysis tool that helps you identify the largest folders on your computer. Built with pure Python (no third-party dependencies) and a modern web interface.

## Features

- **Zero Dependencies**: Pure Python implementation using only standard library modules
- **Interactive Web Interface**: Modern, dark-themed UI with real-time folder navigation
- **Smart Path Optimization**: Automatically eliminates redundant scans when parent/child folders are selected
- **Configurable Depth**: Control how deep the scanner searches (1-10 levels)
- **Real-time Progress**: Live terminal display showing scan progress
- **Detailed Reports**: Beautiful HTML reports highlighting the top 10 heaviest folders
- **Microsoft Detection**: Automatically identifies Microsoft-related system folders
- **Safety First**: Read-only scanner - never modifies or deletes files

## Demo

### Selection Interface
The scanner presents an interactive tree view of your drives and folders. Expand any directory to see its contents, select multiple folders, and configure scan depth.

![Folder Selection](https://via.placeholder.com/800x450/0a0a0f/34d399?text=Folder+Selection+View)

### Live Scanning
Watch real-time progress as the scanner analyzes your storage. The terminal display shows which folders are being processed.

![Scanning Progress](https://via.placeholder.com/800x450/0a0a0f/10b981?text=Live+Scanning+Progress)

### Analysis Report
View your top 10 heaviest folders with direct links, size information, and safety notes about each directory.

![Analysis Report](https://via.placeholder.com/800x450/0a0a0f/059669?text=Storage+Report)

## Installation

### Requirements

- Python 3.7 or higher
- Windows, macOS, or Linux
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Setup

1. Clone or download this repository:

```bash
git clone https://github.com/yourusername/ComputerStorageScanner.git
cd ComputerStorageScanner
```

2. No additional dependencies to install - the project uses only Python standard library.

## Usage

### Starting the Scanner

Run the scanner with Python:

```bash
python storage_scanner.py
```

The application will automatically:
- Start a local web server on port 8000
- Open your default browser to `http://localhost:8000`
- Display the folder selection interface

### Selecting Folders to Scan

1. **Browse Drives**: The interface loads all available drives automatically
2. **Expand Folders**: Click the arrow icon to expand and view subdirectories
3. **Select Folders**: Check the boxes next to folders you want to analyze
4. **Use Quick Selection**: Click "Select/Deselect All" to quickly toggle all subdirectories
5. **Adjust Depth**: Set the maximum folder depth (default: 3 levels)
   - Lower depth = faster load times
   - Higher depth = more granular folder selection

### Running the Scan

1. Select one or more folders using the checkboxes
2. Click the "Scan Selected Folders" button
3. Watch the live terminal output as folders are analyzed
4. Wait for the scan to complete (may take several minutes for large drives)
5. The report will automatically open in a new browser tab

### Understanding the Report

The report displays the top 10 heaviest folders with:

- **Rank**: Position in the top 10
- **Folder Path**: Clickable link to open the folder in your file explorer
- **Size**: Total size in appropriate units (B, KB, MB, GB, TB)
- **Notes**: Information about the folder:
  - Microsoft-related folders are tagged with a blue banner
  - System folders show warnings about deletion risks
  - General folders show standard advisory text

## Configuration

### Changing the Port

Edit `storage_scanner.py` and modify the `PORT` variable:

```python
PORT = 8000  # Change to your desired port
```

### Adjusting Scan Depth

The default maximum depth is 3 levels. You can change this in the web interface or modify the default in `scan_interface.html`:

```html
<input type="number" id="max-depth" value="3" min="1" max="10">
```

## How It Works

### Smart Path Optimization

The scanner automatically optimizes your folder selection:

- If you select a parent folder and all its children, only the parent is scanned
- If you select some children but not all, only the selected children are scanned individually
- This prevents duplicate scanning and significantly reduces scan time

### Size Calculation

The scanner uses `os.walk()` to recursively traverse directories and sum file sizes. It:

- Skips symbolic links and junctions to prevent infinite loops
- Handles permission errors gracefully
- Excludes files it cannot access

### Microsoft Folder Detection

The tool identifies Microsoft-related folders by checking:

- Installation directories (`Program Files`, `Program Files (x86)`)
- System directories (`Windows`)
- AppData locations
- Path components containing "Microsoft"

## Technical Details

### Architecture

- **Backend**: Python HTTP server using `http.server` and `socketserver`
- **Frontend**: Vanilla HTML/CSS/JavaScript (no frameworks)
- **Communication**: RESTful JSON API
- **Progress Updates**: Server-sent polling every 500ms

### API Endpoints

- `GET /` - Serves the main interface
- `GET /get_drives` - Returns list of available drives
- `GET /get_subdirs?path=<path>` - Returns subdirectories of a path
- `GET /progress` - Returns current scan progress
- `POST /scan` - Initiates a scan with selected paths

### File Structure

```
ComputerStorageScanner/
├── storage_scanner.py      # Main Python application and web server
├── scan_interface.html     # Interactive folder selection interface
├── report_template.html    # HTML template for scan reports
├── storage_report.html     # Generated report (created after scan)
└── README.md              # This file
```

## Troubleshooting

### Port Already in Use

If port 8000 is already in use:

1. Change the `PORT` variable in `storage_scanner.py`
2. Or stop the application using that port

### Permission Errors

Some folders may be inaccessible due to permissions:

- The scanner will skip these folders and continue
- Check the terminal for "access denied" messages
- Run as administrator (Windows) or with sudo (Linux/macOS) if needed

### Scan Takes Too Long

To speed up scans:

- Reduce the maximum depth setting
- Select specific folders instead of entire drives
- Exclude network drives and external storage

### Browser Doesn't Open Automatically

Manually navigate to `http://localhost:8000` in your web browser.

## Safety Notes

This tool is designed as a **read-only scanner**:

- It never modifies files or folders
- It never deletes anything
- It only reads file metadata (size, name, path)
- It's safe to use on system folders

However, always exercise caution:

- Review the report before manually deleting any folders
- System folders are marked with warnings
- Some folders may be required by applications or the OS

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Interface design inspired by modern dark-themed web applications
- Built with Python's robust standard library
- No external dependencies for maximum portability

