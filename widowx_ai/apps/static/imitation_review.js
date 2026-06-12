let imitationReviewRecordings = [];
let imitationReviewRecording = null;
let imitationReviewFrames = [];
let imitationReviewIndex = 0;
let imitationReviewPlaying = false;
let imitationReviewTimer = null;
let lastCompletedImitationSession = '';

function formatImitationReviewTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
  const minutes = Math.floor(seconds / 60);
  const wholeSeconds = Math.floor(seconds % 60);
  return `${minutes}:${String(wholeSeconds).padStart(2, '0')}`;
}

function formatImitationReviewFps(actual, nominal) {
  const actualNumber = Number(actual);
  const nominalNumber = Number(nominal);
  if (Number.isFinite(actualNumber) && actualNumber > 0) {
    const actualText = actualNumber >= 10 ? actualNumber.toFixed(1) : actualNumber.toFixed(2);
    return Number.isFinite(nominalNumber) && nominalNumber > 0
      ? `${actualText}/${nominalNumber.toFixed(0)} FPS`
      : `${actualText} FPS`;
  }
  return Number.isFinite(nominalNumber) && nominalNumber > 0 ? `${nominalNumber.toFixed(0)} FPS target` : 'FPS n/a';
}

function imitationFrameTime(index) {
  if (!imitationReviewFrames.length) return 0;
  const first = Number(imitationReviewFrames[0].timestamp);
  const frame = imitationReviewFrames[Math.max(0, Math.min(index, imitationReviewFrames.length - 1))];
  const current = Number(frame.timestamp);
  if (!Number.isFinite(first) || !Number.isFinite(current)) return index / 30;
  return Math.max(0, current - first);
}

function imitationRecordingDuration() {
  return imitationReviewFrames.length < 2 ? 0 : imitationFrameTime(imitationReviewFrames.length - 1);
}

function imitationRecordingLabel(recording) {
  const task = recording.task_name ? ` · ${recording.task_name}` : '';
  const fps = formatImitationReviewFps(recording.actual_camera_fps, recording.nominal_camera_fps);
  return `${recording.name}${task} · ${fps} · ${recording.samples} images`;
}

function imitationFrameImage(frame) {
  const images = frame.images || {};
  return images.hamster_view || frame.image || Object.values(images)[0] || null;
}

function updateImitationReviewState() {
  const current = formatImitationReviewTime(imitationFrameTime(imitationReviewIndex));
  const total = formatImitationReviewTime(imitationRecordingDuration());
  const frame = imitationReviewFrames.length ? imitationReviewIndex + 1 : 0;
  document.getElementById('imitationReviewState').textContent =
    `${current} / ${total} · frame ${frame}/${imitationReviewFrames.length}`;
}

function renderEmptyImitationReview(message) {
  stopImitationReviewPlayback(false);
  const image = document.getElementById('imitationReviewImage');
  const placeholder = document.getElementById('imitationReviewPlaceholder');
  image.style.display = 'none';
  image.removeAttribute('src');
  placeholder.style.display = 'block';
  placeholder.textContent = message;
  document.getElementById('imitationReviewMeta').textContent = '';
  updateImitationReviewState();
}

async function refreshImitationRecordings(preferredPath = null) {
  try {
    const data = await api('/api/recordings');
    imitationReviewRecordings = (data.recordings || []).filter(
      recording => recording.capture_type === 'hamster_imitation'
    );
    const select = document.getElementById('imitationReviewSelect');
    const previous = preferredPath || select.value;
    select.innerHTML = '';
    imitationReviewRecordings.forEach((recording) => {
      const option = document.createElement('option');
      option.value = recording.path;
      option.textContent = imitationRecordingLabel(recording);
      select.appendChild(option);
    });
    if (!imitationReviewRecordings.length) {
      imitationReviewRecording = null;
      imitationReviewFrames = [];
      renderEmptyImitationReview('Aucun enregistrement imitation disponible');
      return;
    }
    select.value = imitationReviewRecordings.some(recording => recording.path === previous)
      ? previous
      : imitationReviewRecordings[0].path;
    await loadImitationReview();
  } catch (err) {
    log(`Erreur verification enregistrement: ${err.message}`);
  }
}

async function loadImitationReview() {
  const path = document.getElementById('imitationReviewSelect').value;
  if (!path) return renderEmptyImitationReview('Aucun enregistrement imitation selectionne');
  try {
    stopImitationReviewPlayback(false);
    imitationReviewRecording = await api('/api/recording/load', {path});
    imitationReviewFrames = imitationReviewRecording.frames || [];
    imitationReviewIndex = 0;
    const slider = document.getElementById('imitationReviewSlider');
    slider.max = Math.max(0, imitationReviewFrames.length - 1);
    slider.value = 0;
    showImitationReviewFrame(0);
  } catch (err) {
    log(`Erreur chargement enregistrement: ${err.message}`);
  }
}

function showImitationReviewFrame(index) {
  if (!imitationReviewRecording || !imitationReviewFrames.length) {
    renderEmptyImitationReview('Cet enregistrement ne contient aucune image');
    return;
  }
  imitationReviewIndex = Math.max(0, Math.min(index, imitationReviewFrames.length - 1));
  const frame = imitationReviewFrames[imitationReviewIndex];
  const imagePath = imitationFrameImage(frame);
  const image = document.getElementById('imitationReviewImage');
  const placeholder = document.getElementById('imitationReviewPlaceholder');
  if (imagePath) {
    image.src = `/api/recording/image?path=${encodeURIComponent(imitationReviewRecording.path)}&image=${encodeURIComponent(imagePath)}&t=${Date.now()}`;
    image.style.display = 'block';
    placeholder.style.display = 'none';
  } else {
    image.style.display = 'none';
    placeholder.style.display = 'block';
    placeholder.textContent = 'Image absente pour cette frame';
  }
  document.getElementById('imitationReviewSlider').value = imitationReviewIndex;
  const metadata = imitationReviewRecording.metadata || {};
  const cameraFps = formatImitationReviewFps(
    imitationReviewRecording.actual_camera_fps,
    imitationReviewRecording.nominal_camera_fps
  );
  const motorFps = formatImitationReviewFps(
    imitationReviewRecording.actual_motor_fps,
    imitationReviewRecording.nominal_motor_fps
  );
  const qpos = Array.isArray(frame.qpos) ? frame.qpos.map(value => Number(value).toFixed(3)).join(', ') : 'n/a';
  const gripper = frame.gripper_position == null ? 'n/a' : `${(Number(frame.gripper_position) * 1000).toFixed(1)} mm`;
  document.getElementById('imitationReviewMeta').textContent =
    `${imitationReviewRecording.name} · task ${metadata.task_name || 'n/a'} · camera ${cameraFps} · motor ${motorFps} · gripper ${gripper} · qpos ${qpos}`;
  updateImitationReviewState();
}

function toggleImitationReviewPlayback() {
  if (imitationReviewPlaying) stopImitationReviewPlayback(false);
  else startImitationReviewPlayback();
}

function startImitationReviewPlayback() {
  if (!imitationReviewFrames.length) return log('Aucun enregistrement imitation a lire');
  if (imitationReviewIndex >= imitationReviewFrames.length - 1) showImitationReviewFrame(0);
  imitationReviewPlaying = true;
  document.getElementById('imitationReviewPlayButton').textContent = 'Pause';
  scheduleNextImitationReviewFrame();
}

function stopImitationReviewPlayback(reset) {
  if (imitationReviewTimer) clearTimeout(imitationReviewTimer);
  imitationReviewTimer = null;
  imitationReviewPlaying = false;
  const button = document.getElementById('imitationReviewPlayButton');
  if (button) button.textContent = 'Play';
  if (reset && imitationReviewFrames.length) showImitationReviewFrame(0);
  else updateImitationReviewState();
}

function scheduleNextImitationReviewFrame() {
  if (!imitationReviewPlaying) return;
  if (imitationReviewIndex >= imitationReviewFrames.length - 1) {
    stopImitationReviewPlayback(false);
    return;
  }
  const current = Number(imitationReviewFrames[imitationReviewIndex].timestamp);
  const next = Number(imitationReviewFrames[imitationReviewIndex + 1].timestamp);
  let delayMs = Number.isFinite(current) && Number.isFinite(next) ? (next - current) * 1000 : 33;
  if (!Number.isFinite(delayMs) || delayMs <= 0) delayMs = 33;
  imitationReviewTimer = setTimeout(() => {
    showImitationReviewFrame(imitationReviewIndex + 1);
    scheduleNextImitationReviewFrame();
  }, Math.max(15, Math.min(delayMs, 250)));
}

function stepImitationReviewFrame(delta) {
  stopImitationReviewPlayback(false);
  showImitationReviewFrame(imitationReviewIndex + delta);
}

async function previousImitationRecording() {
  const select = document.getElementById('imitationReviewSelect');
  if (!select.options.length) return;
  select.selectedIndex = Math.max(0, select.selectedIndex - 1);
  await loadImitationReview();
}

async function nextImitationRecording() {
  const select = document.getElementById('imitationReviewSelect');
  if (!select.options.length) return;
  select.selectedIndex = Math.min(select.options.length - 1, select.selectedIndex + 1);
  await loadImitationReview();
}

function syncImitationRecording(sessionPath, running) {
  if (running || !sessionPath || sessionPath === lastCompletedImitationSession) return;
  lastCompletedImitationSession = sessionPath;
  refreshImitationRecordings(sessionPath);
}

function initImitationReview() {
  refreshImitationRecordings();
}
