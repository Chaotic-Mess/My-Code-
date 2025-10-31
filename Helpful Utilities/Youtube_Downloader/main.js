// cobalt.tools API endpoint
const COBALT_API = 'https://api.cobalt.tools/api/json';

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
    vCodec: "h264",
    vQuality: options.quality,
    aFormat: options.audioFormat,
    isAudioOnly: options.audioOnly,
    filenamePattern: "pretty",
    downloadMode: "auto"
  };

  logToConsole('Fetching download link from cobalt.tools...');
  
  try {
    const response = await fetch(COBALT_API, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    logToConsole(`Error: ${error.message}`, 'error');
    throw error;
  }
}

function displayResult(data) {
  const resultArea = document.getElementById('result-area');
  resultArea.innerHTML = '';

  if (data.status === 'error' || data.status === 'rate-limit') {
    resultArea.innerHTML = `
      <div class="error-box">
        <h3>Error</h3>
        <p>${data.text || 'Failed to fetch download link. Please try again later.'}</p>
      </div>
    `;
    logToConsole(`${data.text}`, 'error');
    return;
  }

  if (data.status === 'picker') {
    // Multiple download options (e.g., different qualities)
    resultArea.innerHTML = `
      <div class="success-box">
        <h3>Multiple Options Available</h3>
        <p>Choose from the available download options:</p>
        <div id="picker-options"></div>
      </div>
    `;
    
    const pickerDiv = document.getElementById('picker-options');
    data.picker.forEach((item, index) => {
      const btn = document.createElement('a');
      btn.href = item.url;
      btn.className = 'btn-download';
      btn.textContent = `Download Option ${index + 1}`;
      btn.target = '_blank';
      btn.download = '';
      pickerDiv.appendChild(btn);
    });
    
    logToConsole('✅ Multiple download options available!', 'success');
  } else if (data.status === 'redirect' || data.status === 'stream') {
    // Single download link
    resultArea.innerHTML = `
      <div class="success-box">
        <h3>Ready to Download!</h3>
        <p>Your download link is ready. Click the button below to download your ${data.status === 'redirect' ? 'video' : 'file'}.</p>
        <a href="${data.url}" class="btn-download" target="_blank" download>⬇Download Now</a>
      </div>
    `;
    logToConsole('✅ Download link ready!', 'success');
  } else if (data.status === 'tunnel') {
    // Cobalt tunnel (handles the download through their service)
    resultArea.innerHTML = `
      <div class="success-box">
        <h3>Processing Video</h3>
        <p>Your video is being processed. Click below to start the download.</p>
        <a href="${data.url}" class="btn-download" target="_blank" download>⬇Download Now</a>
      </div>
    `;
    logToConsole('Video processing complete!', 'success');
  } else {
    resultArea.innerHTML = `
      <div class="error-box">
        <h3>Unexpected Response</h3>
        <p>Received an unexpected response from the API. Status: ${data.status}</p>
      </div>
    `;
    logToConsole(`Unexpected status: ${data.status}`, 'error');
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
