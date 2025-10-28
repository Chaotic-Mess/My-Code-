import os
import requests
import sys

def parse_github_link(link):
    """
    Parse a GitHub directory link to extract owner, repo, branch, and path.
    """
    # Example: https://github.com/Chaotic-Mess/My-Code-/tree/main/Helpful%20Utilities
    parts = link.strip().split('/')
    try:
        owner = parts[3]
        repo = parts[4]
        branch = parts[6]
        path = '/'.join(parts[7:])
        return owner, repo, branch, path
    except Exception:
        raise ValueError("Invalid GitHub directory link format.")

def fetch_github_directory_contents(owner, repo, directory_path, branch="main", token=None):
    """
    Recursively fetch all file metadata (path, size, download_url) in a GitHub directory.
    Returns a list of file dicts.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{directory_path}?ref={branch}"
    headers = {"Authorization": f"token {token}"} if token else {}
    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch {api_url}: {response.status_code}")
        sys.exit(1)
    items = response.json()
    files = []
    if isinstance(items, dict) and items.get('type') == 'file':
        items = [items]
    for item in items:
        if item['type'] == 'file':
            files.append({
                'path': item['path'],
                'size': item['size'],
                'download_url': item['download_url'],
            })
        elif item['type'] == 'dir':
            files.extend(fetch_github_directory_contents(owner, repo, item['path'], branch, token))
    return files

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, "Y", suffix)

def download_files(files, local_base_dir, token=None):
    headers = {"Authorization": f"token {token}"} if token else {}
    for file in files:
        local_path = os.path.join(local_base_dir, file['path'])
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {file['path']}...")
        file_response = requests.get(file['download_url'], headers=headers)
        with open(local_path, 'wb') as f:
            f.write(file_response.content)
    print("Download completed.")

def main():
    print("Enter a GitHub directory link (e.g. https://github.com/owner/repo/tree/main/path):")
    link = input().strip()
    try:
        owner, repo, branch, path = parse_github_link(link)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    print("Optionally enter a GitHub token (leave empty for public repos):")
    token = input().strip() or None
    print("Fetching file list...")
    files = fetch_github_directory_contents(owner, repo, path, branch, token)
    total_size = sum(f['size'] for f in files)
    print(f"There are {len(files)} files total. The total file size would be {sizeof_fmt(total_size)}. Are you sure you want to download? (y/n)")
    confirm = input().strip().lower()
    if confirm != 'y':
        print("Aborted.")
        sys.exit(0)
    local_dir = os.path.basename(path) or repo
    download_files(files, local_dir, token)

if __name__ == "__main__":
    main()
