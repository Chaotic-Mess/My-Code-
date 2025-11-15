import os
import webbrowser
from datetime import datetime
import http.server
import socketserver
import json
from urllib.parse import urlparse, parse_qs
import threading

PORT = 8000
scan_progress = {'current': 0, 'total': 0, 'folder': ''}
progress_lock = threading.Lock()

def get_folder_size(folder_path):
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            # Exclude reparse points (like junctions and symbolic links) from recursion
            dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    try:
                        total_size += os.path.getsize(fp)
                    except OSError:
                        pass # Ignore files that can't be accessed
    except OSError:
        return 0
    return total_size

def format_size(size):
    if size < 1024:
        return f"{size} B"
    elif size < 1024**2:
        return f"{size/1024:.2f} KB"
    elif size < 1024**3:
        return f"{size/1024**2:.2f} MB"
    elif size < 1024**4:
        return f"{size/1024**3:.2f} GB"
    else:
        return f"{size/1024**4:.2f} TB"

def is_microsoft_folder(folder_path):
    ms_paths = [
        os.environ.get("ProgramFiles", "C:\\Program Files"),
        os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
        os.environ.get("WinDir", "C:\\Windows"),
        os.environ.get("AppData", ""),
        os.environ.get("LocalAppData", ""),
    ]
    # Check for "Microsoft" as a component in the path, which is more robust
    normalized_path = os.path.normpath(folder_path).lower()
    if "microsoft" in normalized_path:
        return True
    for ms_path in ms_paths:
        if ms_path and normalized_path.startswith(os.path.normpath(ms_path).lower()):
            return True
    return False

def get_folder_notes(folder_path):
    notes = []
    if is_microsoft_folder(folder_path):
        notes.append('<span class="banner">Microsoft Related</span>')

    system_paths = [
        os.environ.get("WinDir", "C:\\Windows"),
        os.environ.get("ProgramFiles", "C:\\Program Files"),
        os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
    ]

    normalized_path = os.path.normpath(folder_path)
    is_system = False
    for sys_path in system_paths:
        if sys_path and normalized_path.startswith(os.path.normpath(sys_path)):
            notes.append("<strong>Warning:</strong> System folder. Modification can cause instability.")
            is_system = True
            break
    
    if not is_system:
        notes.append("General user folder. Review contents before deleting.")

    return " ".join(notes)


def generate_html_report(folder_sizes):
    template_path = 'report_template.html'
    report_path = 'storage_report.html'

    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    rows = ""
    for i, (size, path) in enumerate(folder_sizes, 1):
        rows += "<tr>"
        rows += f"<td>{i}</td>"
        rows += f'<td><a href="file:///{os.path.abspath(path)}">{path}</a></td>'
        rows += f"<td>{format_size(size)}</td>"
        rows += f"<td>{get_folder_notes(path)}</td>"
        rows += "</tr>\n"

    report_content = template.replace("{{report_rows}}", rows)
    report_content = report_content.replace("{{generation_date}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    webbrowser.open('file://' + os.path.realpath(report_path))

class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = 'scan_interface.html'
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        
        if self.path.startswith('/get_drives'):
            drives = [f"{d}:\\" for d in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{d}:")]
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(drives).encode('utf-8'))
            return

        if self.path.startswith('/get_subdirs'):
            query_components = parse_qs(urlparse(self.path).query)
            path = query_components.get('path', [None])[0]
            subdirs = []
            if path and os.path.isdir(path):
                try:
                    for entry in os.scandir(path):
                        if entry.is_dir():
                            subdirs.append(entry.path)
                except OSError:
                    pass # Ignore folders we can't access
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(subdirs).encode('utf-8'))
            return

        if self.path.startswith('/progress'):
            with progress_lock:
                progress_data = scan_progress.copy()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(progress_data).encode('utf-8'))
            return

        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path == '/scan':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            scan_paths = json.loads(post_data.decode('utf-8'))

            print(f"Received scan request for: {scan_paths}")

            all_folders = []
            for path in scan_paths:
                print(f"Scanning subdirectories of {path}...")
                try:
                    for entry in os.scandir(path):
                        if entry.is_dir():
                            all_folders.append(entry.path)
                except OSError as e:
                    print(f"Could not scan {path}: {e}")

            folder_sizes = []
            # Simple progress for terminal
            total_folders = len(all_folders)
            with progress_lock:
                scan_progress['total'] = total_folders
                scan_progress['current'] = 0
            
            for i, folder in enumerate(all_folders):
                with progress_lock:
                    scan_progress['current'] = i + 1
                    scan_progress['folder'] = folder
                
                print(f"Calculating size for: {folder} ({i+1}/{total_folders})")
                size = get_folder_size(folder)
                if size > 0:
                    folder_sizes.append((size, folder))

            print("\nScan complete. Sorting results...")
            folder_sizes.sort(key=lambda x: x[0], reverse=True)
            top_10 = folder_sizes[:10]

            print("Generating HTML report...")
            generate_html_report(top_10)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'success', 'report': 'storage_report.html'}).encode('utf-8'))
            print(f"Report 'storage_report.html' has been generated and opened.")
            return

def main():
    Handler = MyHttpRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("Serving at port", PORT)
        print("Open http://localhost:8000 in your browser.")
        webbrowser.open(f'http://localhost:{PORT}')
        httpd.serve_forever()

if __name__ == "__main__":
    main()
