/**
 * Cartoonify Studio Pro — Frontend Application Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // ===== State Management =====
  const state = {
    originalImageB64: null,
    cartoonImageB64: null,
    currentStyle: 'ghibli_pro',
    styles: [],
    viewMode: 'split', // 'split' | 'side' | 'cartoon' | 'original'
    sliderPos: 50,
    isDraggingSplit: false,
    batchFiles: [],
    webcamStream: null,
  };

  // Sample portrait for instant preview
  const SAMPLE_IMAGE_URL =
    'data:image/svg+xml;charset=utf-8,' +
    encodeURIComponent(`
    <svg xmlns="http://www.w3.org/2000/svg" width="600" height="600" viewBox="0 0 600 600">
      <defs>
        <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#fd79a8"/>
          <stop offset="100%" stop-color="#6c5ce7"/>
        </linearGradient>
      </defs>
      <rect width="600" height="600" fill="url(#bg)"/>
      <circle cx="300" cy="260" r="120" fill="#ffeaa7"/>
      <circle cx="260" cy="240" r="16" fill="#2d3436"/>
      <circle cx="340" cy="240" r="16" fill="#2d3436"/>
      <path d="M 270 300 Q 300 340 330 300" stroke="#d63031" stroke-width="8" fill="none" stroke-linecap="round"/>
      <path d="M 180 500 C 180 400, 420 400, 420 500 Z" fill="#0984e3"/>
      <text x="300" y="560" font-size="24" fill="#ffffff" text-anchor="middle" font-family="sans-serif" font-weight="bold">Cartoonify Demo Portrait</text>
    </svg>`);

  // ===== DOM Elements =====
  const elements = {
    themeToggle: document.getElementById('themeToggle'),
    deviceStatus: document.getElementById('deviceStatus'),
    deviceText: document.getElementById('deviceText'),
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('fileInput'),
    browseBtn: document.getElementById('browseBtn'),
    webcamBtn: document.getElementById('webcamBtn'),
    sampleBtn: document.getElementById('sampleBtn'),
    stylesGrid: document.getElementById('stylesGrid'),
    selectedStyleBadge: document.getElementById('selectedStyleBadge'),
    strengthSlider: document.getElementById('strengthSlider'),
    strengthValue: document.getElementById('strengthValue'),
    faceAlignCheck: document.getElementById('faceAlignCheck'),
    generateBtn: document.getElementById('generateBtn'),
    canvasStage: document.getElementById('canvasStage'),
    placeholderView: document.getElementById('placeholderView'),
    comparisonViewer: document.getElementById('comparisonViewer'),
    imgOriginal: document.getElementById('imgOriginal'),
    imgCartoon: document.getElementById('imgCartoon'),
    cartoonClipWrapper: document.getElementById('cartoonClipWrapper'),
    splitHandle: document.getElementById('splitHandle'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    metricStatus: document.getElementById('metricStatus'),
    metricRes: document.getElementById('metricRes'),
    metricTime: document.getElementById('metricTime'),
    metricStyle: document.getElementById('metricStyle'),
    copyBtn: document.getElementById('copyBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    historyCount: document.getElementById('historyCount'),
    historyGrid: document.getElementById('historyGrid'),
    clearHistoryBtn: document.getElementById('clearHistoryBtn'),
    batchDropzone: document.getElementById('batchDropzone'),
    batchFileInput: document.getElementById('batchFileInput'),
    batchBrowseBtn: document.getElementById('batchBrowseBtn'),
    batchList: document.getElementById('batchList'),
    batchProcessBtn: document.getElementById('batchProcessBtn'),
    webcamModal: document.getElementById('webcamModal'),
    webcamVideo: document.getElementById('webcamVideo'),
    webcamCanvas: document.getElementById('webcamCanvas'),
    capturePhotoBtn: document.getElementById('capturePhotoBtn'),
    closeWebcamBtn: document.getElementById('closeWebcamBtn'),
  };

  // ===== Initialize App =====
  async function init() {
    setupTheme();
    setupTabs();
    setupDropzone();
    setupSplitSlider();
    setupViewModes();
    setupWebcam();
    setupCustomSliders();
    await fetchStyles();
    await fetchHistory();
    checkHealth();
  }

  // ===== Theme System =====
  function setupTheme() {
    const saved = localStorage.getItem('cartoonify_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    elements.themeToggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('cartoonify_theme', next);
      elements.themeToggle.querySelector('.theme-icon').textContent = next === 'dark' ? '🌙' : '☀️';
    });
  }

  // ===== Tab Switching =====
  function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        const target = document.getElementById(btn.dataset.tab);
        if (target) target.classList.add('active');
      });
    });
  }

  // ===== API Communications =====
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      if (data.status === 'healthy') {
        const device = data.device ? data.device.toUpperCase() : 'CPU';
        elements.deviceText.textContent = `Engine: ${device}`;
      }
    } catch {
      elements.deviceText.textContent = 'Engine Offline';
    }
  }

  async function fetchStyles() {
    try {
      const res = await fetch('/api/styles');
      state.styles = await res.json();
      renderStylesGrid();
    } catch (e) {
      console.error('Failed to load styles:', e);
    }
  }

  function renderStylesGrid() {
    elements.stylesGrid.innerHTML = '';
    state.styles.forEach(style => {
      const card = document.createElement('div');
      card.className = `style-card ${style.key === state.currentStyle ? 'active' : ''}`;
      card.dataset.key = style.key;
      card.innerHTML = `
        <span class="style-card-icon">${style.icon}</span>
        <div class="style-card-info">
          <span class="style-card-name">${style.name}</span>
          <span class="style-card-tag">${style.category}</span>
        </div>
      `;
      card.addEventListener('click', () => selectStyle(style.key));
      elements.stylesGrid.appendChild(card);
    });
  }

  function selectStyle(key) {
    state.currentStyle = key;
    const style = state.styles.find(s => s.key === key);
    if (!style) return;

    elements.selectedStyleBadge.textContent = style.name;
    elements.metricStyle.textContent = style.name;
    document.querySelectorAll('.style-card').forEach(c => {
      c.classList.toggle('active', c.dataset.key === key);
    });

    if (style.default_strength) {
      const val = Math.round(style.default_strength * 100);
      elements.strengthSlider.value = val;
      elements.strengthValue.textContent = `${val}%`;
    }

    const accordion = document.getElementById('customAccordion');
    if (key === 'custom') {
      accordion.open = true;
    }
  }

  // ===== Image Drop & Loading =====
  function setupDropzone() {
    const dz = elements.dropzone;

    ['dragenter', 'dragover'].forEach(name => {
      dz.addEventListener(name, e => {
        e.preventDefault();
        dz.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dz.addEventListener(name, e => {
        e.preventDefault();
        dz.classList.remove('dragover');
      });
    });

    dz.addEventListener('drop', e => {
      const files = e.dataTransfer.files;
      if (files.length > 0) loadFile(files[0]);
    });

    elements.browseBtn.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', e => {
      if (e.target.files.length > 0) loadFile(e.target.files[0]);
    });

    elements.sampleBtn.addEventListener('click', () => {
      loadDataUrl(SAMPLE_IMAGE_URL, 'sample_portrait.png');
    });
  }

  function loadFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file (JPEG, PNG, WebP).');
      return;
    }
    const reader = new FileReader();
    reader.onload = e => {
      loadDataUrl(e.target.result, file.name);
    };
    reader.readAsDataURL(file);
  }

  function loadDataUrl(dataUrl, filename = 'image.jpg') {
    state.originalImageB64 = dataUrl;
    state.cartoonImageB64 = null;

    elements.imgOriginal.src = dataUrl;
    elements.imgCartoon.src = dataUrl;

    const img = new Image();
    img.onload = () => {
      elements.metricRes.textContent = `${img.naturalWidth} × ${img.naturalHeight}px`;
    };
    img.src = dataUrl;

    elements.placeholderView.classList.add('hidden');
    elements.comparisonViewer.classList.remove('hidden');
    elements.generateBtn.disabled = false;
    elements.downloadBtn.disabled = true;
    elements.copyBtn.disabled = true;
    elements.metricStatus.textContent = 'Image Loaded';

    setSplitPosition(50);
  }

  // ===== Generate Action =====
  elements.generateBtn.addEventListener('click', async () => {
    if (!state.originalImageB64) return;

    elements.loadingOverlay.classList.remove('hidden');
    elements.generateBtn.disabled = true;
    elements.metricStatus.textContent = 'Processing...';

    const customParams = {
      line_thickness: parseInt(document.getElementById('paramThickness').value),
      line_opacity: parseInt(document.getElementById('paramOpacity').value) / 100.0,
      color_smoothness: parseInt(document.getElementById('paramSmooth').value),
      num_colors: parseInt(document.getElementById('paramColors').value),
      saturation: parseInt(document.getElementById('paramSaturation').value) / 10.0,
      sharpness: parseInt(document.getElementById('paramSharpness').value) / 100.0,
    };

    const payload = {
      image: state.originalImageB64,
      style: state.currentStyle,
      strength: parseInt(elements.strengthSlider.value) / 100.0,
      use_face_align: elements.faceAlignCheck.checked,
      custom_params: customParams,
    };

    try {
      const res = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!data.success) throw new Error(data.detail || 'Processing failed');

      state.cartoonImageB64 = data.image;
      elements.imgCartoon.src = data.image;

      elements.downloadBtn.disabled = false;
      elements.copyBtn.disabled = false;
      elements.metricTime.textContent = `${data.elapsed_seconds}s`;
      elements.metricStatus.textContent = 'Done ✨';

      await fetchHistory();
    } catch (err) {
      alert(`Generation failed: ${err.message}`);
      elements.metricStatus.textContent = 'Error';
    } finally {
      elements.loadingOverlay.classList.add('hidden');
      elements.generateBtn.disabled = false;
    }
  });

  // ===== Split Slider Interactivity =====
  function setupSplitSlider() {
    const container = elements.comparisonViewer;
    const handle = elements.splitHandle;

    function onPointerMove(e) {
      if (!state.isDraggingSplit) return;
      const rect = container.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const x = clientX - rect.left;
      const percent = Math.max(0, Math.min(100, (x / rect.width) * 100));
      setSplitPosition(percent);
    }

    function onPointerUp() {
      state.isDraggingSplit = false;
      window.removeEventListener('mousemove', onPointerMove);
      window.removeEventListener('mouseup', onPointerUp);
      window.removeEventListener('touchmove', onPointerMove);
      window.removeEventListener('touchend', onPointerUp);
    }

    handle.addEventListener('mousedown', () => {
      state.isDraggingSplit = true;
      window.addEventListener('mousemove', onPointerMove);
      window.addEventListener('mouseup', onPointerUp);
    });

    handle.addEventListener('touchstart', () => {
      state.isDraggingSplit = true;
      window.addEventListener('touchmove', onPointerMove);
      window.addEventListener('touchend', onPointerUp);
    });
  }

  function setSplitPosition(percent) {
    state.sliderPos = percent;
    elements.splitHandle.style.left = `${percent}%`;
    elements.cartoonClipWrapper.style.clipPath = `polygon(${percent}% 0, 100% 0, 100% 100%, ${percent}% 100%)`;
  }

  // ===== View Modes =====
  function setupViewModes() {
    const modeBtns = document.querySelectorAll('.mode-btn');
    modeBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        modeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const mode = btn.dataset.mode;
        state.viewMode = mode;

        if (mode === 'split') {
          elements.splitHandle.classList.remove('hidden');
          setSplitPosition(50);
        } else if (mode === 'original') {
          elements.splitHandle.classList.add('hidden');
          elements.cartoonClipWrapper.style.clipPath = 'polygon(100% 0, 100% 0, 100% 100%, 100% 100%)';
        } else if (mode === 'cartoon') {
          elements.splitHandle.classList.add('hidden');
          elements.cartoonClipWrapper.style.clipPath = 'polygon(0 0, 100% 0, 100% 100%, 0 100%)';
        } else if (mode === 'side') {
          elements.splitHandle.classList.remove('hidden');
          setSplitPosition(50);
        }
      });
    });
  }

  // ===== Download & Copy =====
  elements.downloadBtn.addEventListener('click', () => {
    if (!state.cartoonImageB64) return;
    const a = document.createElement('a');
    a.href = state.cartoonImageB64;
    a.download = `cartoon_${state.currentStyle}_${Date.now()}.jpg`;
    a.click();
  });

  elements.copyBtn.addEventListener('click', async () => {
    if (!state.cartoonImageB64) return;
    try {
      const res = await fetch(state.cartoonImageB64);
      const blob = await res.blob();
      await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
      const prev = elements.copyBtn.textContent;
      elements.copyBtn.textContent = 'Copied!';
      setTimeout(() => (elements.copyBtn.textContent = prev), 2000);
    } catch {
      alert('Clipboard copy is not supported in this browser context.');
    }
  });

  // ===== Slider Values Synchronizer =====
  function setupCustomSliders() {
    elements.strengthSlider.addEventListener('input', e => {
      elements.strengthValue.textContent = `${e.target.value}%`;
    });

    const bindSlider = (id, valId, suffix = '') => {
      const el = document.getElementById(id);
      const val = document.getElementById(valId);
      if (el && val) {
        el.addEventListener('input', e => {
          val.textContent = `${e.target.value}${suffix}`;
        });
      }
    };

    bindSlider('paramThickness', 'valThickness', 'px');
    bindSlider('paramOpacity', 'valOpacity', '%');
    bindSlider('paramSmooth', 'valSmooth', ' passes');
    bindSlider('paramColors', 'valColors', '');
    bindSlider('paramSaturation', 'valSaturation', 'x');
    bindSlider('paramSharpness', 'valSharpness', '%');
  }

  // ===== Webcam Camera Snap =====
  function setupWebcam() {
    elements.webcamBtn.addEventListener('click', async () => {
      elements.webcamModal.showModal();
      try {
        state.webcamStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
        });
        elements.webcamVideo.srcObject = state.webcamStream;
      } catch (err) {
        alert('Could not access camera: ' + err.message);
        elements.webcamModal.close();
      }
    });

    elements.closeWebcamBtn.addEventListener('click', () => {
      stopWebcam();
      elements.webcamModal.close();
    });

    elements.capturePhotoBtn.addEventListener('click', () => {
      const video = elements.webcamVideo;
      const canvas = elements.webcamCanvas;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const b64 = canvas.toDataURL('image/jpeg', 0.95);
      stopWebcam();
      elements.webcamModal.close();
      loadDataUrl(b64, 'webcam_capture.jpg');
    });
  }

  function stopWebcam() {
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach(t => t.stop());
      state.webcamStream = null;
    }
  }

  // ===== History =====
  async function fetchHistory() {
    try {
      const res = await fetch('/api/history');
      const records = await res.json();
      elements.historyCount.textContent = records.length;
      renderHistory(records);
    } catch {}
  }

  function renderHistory(records) {
    elements.historyGrid.innerHTML = '';
    if (!records || records.length === 0) {
      elements.historyGrid.innerHTML = '<p class="empty-state">No artwork generated yet.</p>';
      return;
    }

    records.forEach(rec => {
      const card = document.createElement('div');
      card.className = 'history-card';
      const thumb = rec.thumbnail || SAMPLE_IMAGE_URL;
      card.innerHTML = `
        <img src="${thumb}" alt="${rec.style_name}">
        <span class="history-card-title">${rec.style_name}</span>
        <span class="history-card-time">${new Date(rec.timestamp).toLocaleTimeString()}</span>
      `;
      card.addEventListener('click', () => {
        if (rec.thumbnail) loadDataUrl(rec.thumbnail, 'history_item.jpg');
      });
      elements.historyGrid.appendChild(card);
    });
  }

  elements.clearHistoryBtn.addEventListener('click', async () => {
    await fetch('/api/history', { method: 'DELETE' });
    await fetchHistory();
  });

  // Run initialization
  init();
});
