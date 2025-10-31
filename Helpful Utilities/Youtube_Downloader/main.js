// PythonAnywhere API endpoint - Replace 'yourusername' with your actual PythonAnywhere username
const API_ENDPOINT = 'https://yourusername.pythonanywhere.com/api/download';

function logToConsole(msg, type = 'info') {
  const consoleArea = document.getElementById('console-area');
  const line = document.createElement('p');
  line.textContent = msg;
  if (type === 'error') line.style.color = '#ff6b6b';
  if (type === 'success') line.style.color = '#51cf66';
  consoleArea.appendChild(line);
  consoleArea.scrollTop = consoleArea.scrollHeight;
}

function clearConsole() {
  document.getElementById('console-area').innerHTML = '';
}

function isValidYouTubeUrl(url) {
  const patterns = [
    /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+$/,
    /^(https?:\/\/)?(www\.)?youtube\.com\/watch\?v=[\w-]+/,
    /^(https?:\/\/)?(www\.)?youtu\.be\/[\w-]+/,
    /^(https?:\/\/)?(www\.)?youtube\.com\/shorts\/[\w-]+/
  ];
  return patterns.some(pattern => pattern.test(url));
}

async function fetchDownloadLink(url, options) {
  const requestBody = {
    url: url,
    quality: options.quality,
    audio_only: options.audioOnly,
    audio_format: options.audioFormat
  };

  logToConsole('🔍 Fetching download link from server...');
  
  try {
    const response = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    logToConsole(`❌ Error: ${error.message}`, 'error');
    throw error;
  }
}

function displayResult(data) {
  const resultArea = document.getElementById('result-area');
  resultArea.innerHTML = '';

  if (data.error) {
    resultArea.innerHTML = `
      <div class="error-box">
        <h3>❌ Error</h3>
        <p>${data.error}</p>
      </div>
    `;
    logToConsole(`❌ ${data.error}`, 'error');
    return;
  }

  if (data.success && data.download_url) {
    const videoInfo = data.info || {};
    resultArea.innerHTML = `
      <div class="success-box">
        <h3>✅ Ready to Download!</h3>
        ${videoInfo.title ? `<p><strong>Title:</strong> ${videoInfo.title}</p>` : ''}
        ${videoInfo.duration ? `<p><strong>Duration:</strong> ${videoInfo.duration}</p>` : ''}
        ${videoInfo.format ? `<p><strong>Format:</strong> ${videoInfo.format}</p>` : ''}
        <p>Click the button below to download your ${data.audio_only ? 'audio' : 'video'} file.</p>
        <a href="${data.download_url}" class="btn-download" target="_blank" download>⬇️ Download Now</a>
      </div>
    `;
    logToConsole('✅ Download link ready!', 'success');
  } else {
    resultArea.innerHTML = `
      <div class="error-box">
        <h3>⚠️ Unexpected Response</h3>
        <p>Received an unexpected response from the server.</p>
      </div>
    `;
    logToConsole('⚠️ Unexpected server response', 'error');
  }

  resultArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

document.getElementById('fetch-video').onclick = async function() {
  const url = document.getElementById('youtube-url').value.trim();
  const audioOnly = document.getElementById('audio-only').checked;
  const quality = document.getElementById('quality').value;
  const audioFormat = document.getElementById('audio-format').value;
  
  const resultArea = document.getElementById('result-area');
  resultArea.innerHTML = '';
  clearConsole();

  if (!url) {
    resultArea.innerHTML = '<p class="error-box">Please enter a YouTube URL.</p>';
    return;
  }

  if (!isValidYouTubeUrl(url)) {
    resultArea.innerHTML = '<p class="error-box">Invalid YouTube URL. Please check and try again.</p>';
    return;
  }

  logToConsole(`🎬 Processing: ${url}`);
  logToConsole(`Settings: ${audioOnly ? 'Audio Only' : 'Video'} | Quality: ${quality} | Audio Format: ${audioFormat}`);

  try {
    const data = await fetchDownloadLink(url, {
      audioOnly: audioOnly,
      quality: quality,
      audioFormat: audioFormat
    });
    
    displayResult(data);
  } catch (error) {
    resultArea.innerHTML = `
      <div class="error-box">
        <h3>❌ Error</h3>
        <p>Failed to fetch download link. This could be due to:</p>
        <ul>
          <li>Network connectivity issues</li>
          <li>The video being private or unavailable</li>
          <li>Age-restricted content</li>
          <li>Geographic restrictions</li>
          <li>Temporary API issues</li>
        </ul>
        <p><strong>Error details:</strong> ${error.message}</p>
      </div>
    `;
  }
};

// Allow Enter key to trigger download
document.getElementById('youtube-url').addEventListener('keypress', function(e) {
  if (e.key === 'Enter') {
    document.getElementById('fetch-video').click();
  }
});

// Toggle audio format visibility based on audio-only checkbox
document.getElementById('audio-only').addEventListener('change', function() {
  const audioFormatSelect = document.getElementById('audio-format');
  const qualitySelect = document.getElementById('quality');
  
  if (this.checked) {
    audioFormatSelect.style.display = 'inline-block';
    audioFormatSelect.previousElementSibling.style.display = 'inline-block';
    qualitySelect.style.display = 'none';
    qualitySelect.previousElementSibling.style.display = 'none';
  } else {
    audioFormatSelect.style.display = 'none';
    audioFormatSelect.previousElementSibling.style.display = 'none';
    qualitySelect.style.display = 'inline-block';
    qualitySelect.previousElementSibling.style.display = 'inline-block';
  }
});

// Initialize visibility
document.getElementById('audio-format').style.display = 'none';
document.getElementById('audio-format').previousElementSibling.style.display = 'none';
