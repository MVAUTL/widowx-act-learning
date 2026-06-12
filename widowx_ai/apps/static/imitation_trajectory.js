let imitationTrajectoryPoints = [];

function setImitationTrajectory(points) {
  imitationTrajectoryPoints = Array.isArray(points) ? points : [];
  renderLiveOverlay();
}

function trajectoryMarkerColor(action) {
  if (action === 'close') return '#ef2929';
  if (action === 'open') return '#2777ff';
  return '#ffd51f';
}

function trajectorySvgDataUri(points) {
  if (!Array.isArray(points) || points.length < 2) return '';
  const svgPoints = points.map(point => `${Number(point.x) * 1000},${Number(point.y) * 1000}`).join(' ');
  const markers = points.map((point, index) => {
    const x = Number(point.x) * 1000;
    const y = Number(point.y) * 1000;
    const color = trajectoryMarkerColor(point.gripper_action);
    return `<circle cx="${x}" cy="${y}" r="14" fill="${color}" stroke="white" stroke-width="5"/>` +
      `<text x="${x + 20}" y="${y - 18}" fill="white" font-size="28" font-family="sans-serif" font-weight="700">${index + 1}</text>`;
  }).join('');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000" preserveAspectRatio="none">` +
    `<polyline points="${svgPoints}" fill="none" stroke="#39d8c1" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>` +
    markers +
    `</svg>`;
  return `data:image/svg+xml;base64,${btoa(svg)}`;
}

function renderLiveOverlay() {
  const trajectoryOverlay = document.getElementById('liveHamsterOverlay');
  const imageOverlay = document.getElementById('liveHamsterImageOverlay');
  const mode = document.getElementById('overlayMode')?.value || 'trajectory';
  const opacity = Number(document.getElementById('overlayOpacity')?.value || 0.55);
  const dataUri = trajectorySvgDataUri(imitationTrajectoryPoints);

  trajectoryOverlay.style.display = 'none';
  imageOverlay.style.display = 'none';
  if (!cameraRunning) return;

  if (mode === 'image' && hamsterOverlay) {
    if (imageOverlay.src !== hamsterOverlay) imageOverlay.src = hamsterOverlay;
    imageOverlay.style.objectFit = 'fill';
    imageOverlay.style.opacity = String(opacity);
    imageOverlay.style.mixBlendMode = 'normal';
    imageOverlay.style.display = 'block';
    return;
  }

  if (dataUri) {
    if (trajectoryOverlay.src !== dataUri) trajectoryOverlay.src = dataUri;
    trajectoryOverlay.style.objectFit = 'fill';
    trajectoryOverlay.style.opacity = '1';
    trajectoryOverlay.style.mixBlendMode = 'normal';
    trajectoryOverlay.style.display = 'block';
  }
}
