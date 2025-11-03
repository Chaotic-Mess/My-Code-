const fileInput = document.getElementById("videoInput");
const videoSelect = document.getElementById("videoSelect");
const video = document.getElementById("video");
const asciiOutput = document.getElementById("asciiOutput");
const startBtn = document.getElementById("startBtn");
const scaleRange = document.getElementById("scaleRange");
const scaleValue = document.getElementById("scaleValue");
const charsetSelect = document.getElementById("charsetSelect");

// controls
const pauseBtn = document.getElementById("pauseBtn");
const seekBar = document.getElementById("seekBar");
const timeLabel = document.getElementById("timeLabel");

// Character set presets
const CHARSETS = {
  basic: "@%#*+=-:. ",
  extended: "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^'. ",
  retro: "█▓▒░ ."
};

// Initialize from the current select value
let ASCII_CHARS = CHARSETS[charsetSelect.value];

charsetSelect.addEventListener("change", () => {
  ASCII_CHARS = CHARSETS[charsetSelect.value];
});

let ctx, canvas;
let scaleFactor = 1;

//    keep target size + loop id + seeking state
let targetWidth = 120;
let targetHeight = 60;
let rafId = null;
let isUserSeeking = false;

// --- Scale control ---
scaleRange.addEventListener("input", () => {
  scaleFactor = parseFloat(scaleRange.value);
  scaleValue.textContent = `${scaleFactor.toFixed(2)}x`;
});

// --- File upload handler ---
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    const url = URL.createObjectURL(file);
    video.src = url;
    videoSelect.value = "";
  }
});

// --- Preloaded video handler ---
videoSelect.addEventListener("change", (e) => {
  const selected = e.target.value;
  if (selected) {
    video.src = selected;
    fileInput.value = "";
  }
});

// when metadata is ready, enable seek bar with duration
video.addEventListener("loadedmetadata", () => {
  seekBar.max = video.duration.toString();
  seekBar.value = "0";
  updateTimeLabel();
});

// keep seek bar in sync while playing (unless user is dragging)
video.addEventListener("timeupdate", () => {
  if (!isUserSeeking) {
    seekBar.value = video.currentTime.toString();
    updateTimeLabel();
  }
});

// user starts dragging the slider
seekBar.addEventListener("input", () => {
  isUserSeeking = true;
  // show the previewed time immediately
  timeLabel.textContent = `${formatTime(parseFloat(seekBar.value))} / ${formatTime(video.duration)}`;
});

// user releases the slider thumb -> jump video to that time and render the frame
seekBar.addEventListener("change", () => {
  const t = parseFloat(seekBar.value);
  video.pause();
  pauseBtn.textContent = "Resume";
  video.currentTime = Number.isFinite(t) ? t : 0;
});

// after seeking finishes, draw the frame at that exact timestamp
video.addEventListener("seeked", () => {
  isUserSeeking = false;
  drawFrameOnce(); // renders the paused frame in ASCII
});

// Pause/Resume toggle
pauseBtn.addEventListener("click", () => {
  if (!video.src) return;
  if (video.paused) {
    video.play();
    pauseBtn.textContent = "Pause";
    startRenderLoop(); // resume ASCII loop
  } else {
    video.pause();
    pauseBtn.textContent = "Resume";
    // loop will naturally stop because render checks video.paused
  }
});

startBtn.addEventListener("click", async () => {
  if (!video.src) {
    alert("Please upload or select a video first!");
    return;
  }

  await video.play();
  video.pause();

  // Compute scaled target size while preserving aspect ratio
  const baseWidth = 120;
  targetWidth = Math.floor(baseWidth * scaleFactor);
  const aspectRatio = video.videoHeight / video.videoWidth;
  const charAspect = 0.55;
  targetHeight = Math.floor(targetWidth * aspectRatio * charAspect);

  canvas = document.createElement("canvas");
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  ctx = canvas.getContext("2d");

  // enable controls
  pauseBtn.disabled = false;
  seekBar.disabled = false;

  video.currentTime = 0;
  seekBar.value = "0";
  updateTimeLabel();

  video.play();
  pauseBtn.textContent = "Pause";
  startRenderLoop();
});

function startRenderLoop() {
  cancelAnimationFrame(rafId);
  const tick = () => {
    if (!video.paused && !video.ended) {
      ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
      const frame = ctx.getImageData(0, 0, targetWidth, targetHeight);
      asciiOutput.textContent = convertToAscii(frame);
      rafId = requestAnimationFrame(tick);
    }
  };
  rafId = requestAnimationFrame(tick);
}

// draw a single frame (used after seeking while paused)
function drawFrameOnce() {
  ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
  const frame = ctx.getImageData(0, 0, targetWidth, targetHeight);
  asciiOutput.textContent = convertToAscii(frame);
  // sync the UI
  seekBar.value = video.currentTime.toString();
  updateTimeLabel();
}

function convertToAscii(imageData) {
  const data = imageData.data;
  let ascii = "";
  for (let y = 0; y < imageData.height; y++) {
    for (let x = 0; x < imageData.width; x++) {
      const offset = (y * imageData.width + x) * 4;
      const r = data[offset];
      const g = data[offset + 1];
      const b = data[offset + 2];
      const brightness = (r + g + b) / 3;
      const idx = Math.floor((brightness / 255) * (ASCII_CHARS.length - 1));
      ascii += ASCII_CHARS[idx];
    }
    ascii += "\n";
  }
  return ascii;
}

// small helpers
function updateTimeLabel() {
  timeLabel.textContent = `${formatTime(video.currentTime)} / ${formatTime(video.duration)}`;
}
function formatTime(s) {
  if (!isFinite(s)) return "00:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}
