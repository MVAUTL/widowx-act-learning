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
  const overlay = document.getElementById('liveHamsterOverlay');
  const dataUri = trajectorySvgDataUri(imitationTrajectoryPoints);
  if (dataUri && cameraRunning) {
    if (overlay.src !== dataUri) overlay.src = dataUri;
    overlay.style.opacity = '1';
    overlay.style.mixBlendMode = 'normal';
    overlay.style.display = 'block';
  } else {
    overlay.style.display = 'none';
  }
}
