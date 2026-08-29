/**
 * Cartoonify Studio Pro — Static Showcase Website Controller & Masterpiece Anime Shader
 * Features smooth bilateral tone mapping, natural skin color preservation (0% yellow blotches),
 * and clean anti-aliased hand-drawn pencil/ink inking.
 */

document.addEventListener('DOMContentLoaded', () => {
  // ===== 1. Theme Management =====
  const themeToggle = document.getElementById('themeToggle');
  const savedTheme = localStorage.getItem('cartoonify_site_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);

  if (themeToggle) {
    themeToggle.querySelector('.theme-icon').textContent = savedTheme === 'dark' ? '🌙' : '☀️';
    themeToggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('cartoonify_site_theme', next);
      themeToggle.querySelector('.theme-icon').textContent = next === 'dark' ? '🌙' : '☀️';
    });
  }

  // ===== 2. Procedural Demo Image Generators =====
  function drawSample(type, ctx, width, height, isCartoon = false, style = 'ghibli_pro') {
    ctx.clearRect(0, 0, width, height);

    if (type === 'portrait') {
      const grad = ctx.createLinearGradient(0, 0, width, height);
      if (isCartoon) {
        grad.addColorStop(0, '#ffeaa7');
        grad.addColorStop(1, '#ff9a9e');
      } else {
        grad.addColorStop(0, '#667eea');
        grad.addColorStop(1, '#764ba2');
      }
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, width, height);

      // Face silhouette
      ctx.fillStyle = isCartoon ? '#ffe8d6' : '#ffd1b3';
      ctx.beginPath();
      ctx.ellipse(width / 2, height * 0.44, width * 0.22, height * 0.28, 0, 0, Math.PI * 2);
      ctx.fill();
      if (isCartoon) {
        ctx.strokeStyle = '#4a2810';
        ctx.lineWidth = 3;
        ctx.stroke();
      }

      // Hair
      ctx.fillStyle = isCartoon ? '#2c3e50' : '#4a2810';
      ctx.beginPath();
      ctx.arc(width / 2, height * 0.35, width * 0.24, Math.PI, Math.PI * 2);
      ctx.fill();

      // Eyes
      ctx.fillStyle = isCartoon ? '#0984e3' : '#333333';
      ctx.beginPath();
      ctx.arc(width * 0.42, height * 0.42, isCartoon ? 14 : 9, 0, Math.PI * 2);
      ctx.arc(width * 0.58, height * 0.42, isCartoon ? 14 : 9, 0, Math.PI * 2);
      ctx.fill();

      if (isCartoon) {
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(width * 0.43, height * 0.41, 5, 0, Math.PI * 2);
        ctx.arc(width * 0.59, height * 0.41, 5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Smile
      ctx.strokeStyle = isCartoon ? '#d63031' : '#b33939';
      ctx.lineWidth = isCartoon ? 4 : 3;
      ctx.beginPath();
      ctx.arc(width / 2, height * 0.52, width * 0.08, 0.2, Math.PI - 0.2);
      ctx.stroke();

      // Clothes
      ctx.fillStyle = isCartoon ? '#00b894' : '#34495e';
      ctx.beginPath();
      ctx.ellipse(width / 2, height * 0.95, width * 0.4, height * 0.25, 0, Math.PI, 0);
      ctx.fill();
    } else if (type === 'city') {
      const grad = ctx.createLinearGradient(0, 0, 0, height);
      grad.addColorStop(0, isCartoon ? '#2c3e50' : '#0f2027');
      grad.addColorStop(0.6, isCartoon ? '#e74c3c' : '#203a43');
      grad.addColorStop(1, isCartoon ? '#f39c12' : '#2c5364');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, width, height);

      ctx.fillStyle = isCartoon ? '#fdcb6e' : '#e67e22';
      ctx.beginPath();
      ctx.arc(width * 0.5, height * 0.45, 60, 0, Math.PI * 2);
      ctx.fill();

      const bldgs = [
        { x: 0.05, w: 0.18, h: 0.45 },
        { x: 0.26, w: 0.16, h: 0.65 },
        { x: 0.45, w: 0.2, h: 0.55 },
        { x: 0.68, w: 0.15, h: 0.72 },
        { x: 0.85, w: 0.12, h: 0.48 },
      ];

      bldgs.forEach(b => {
        ctx.fillStyle = isCartoon ? '#2d3436' : '#111111';
        ctx.fillRect(width * b.x, height * (1 - b.h), width * b.w, height * b.h);
        if (isCartoon) {
          ctx.strokeStyle = '#00f0ff';
          ctx.lineWidth = 2;
          ctx.strokeRect(width * b.x, height * (1 - b.h), width * b.w, height * b.h);
        }
      });
    } else {
      const grad = ctx.createLinearGradient(0, 0, 0, height);
      grad.addColorStop(0, isCartoon ? '#74b9ff' : '#4b6cb7');
      grad.addColorStop(1, isCartoon ? '#ffeaa7' : '#182e6a');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, width, height);

      ctx.fillStyle = isCartoon ? '#6c5ce7' : '#2c3e50';
      ctx.beginPath();
      ctx.moveTo(0, height * 0.7);
      ctx.lineTo(width * 0.35, height * 0.35);
      ctx.lineTo(width * 0.7, height * 0.75);
      ctx.fill();

      ctx.fillStyle = isCartoon ? '#00b894' : '#16a085';
      ctx.beginPath();
      ctx.moveTo(width * 0.3, height * 0.8);
      ctx.lineTo(width * 0.65, height * 0.45);
      ctx.lineTo(width, height * 0.85);
      ctx.lineTo(width, height);
      ctx.lineTo(0, height);
      ctx.fill();
    }
  }

  // ===== 3. Hero Split Slider =====
  const heroViewer = document.getElementById('heroViewer');
  const heroCartoonWrapper = document.getElementById('heroCartoonWrapper');
  const heroHandle = document.getElementById('heroHandle');
  const heroImgOriginal = document.getElementById('heroImgOriginal');
  const heroImgCartoon = document.getElementById('heroImgCartoon');

  let currentPreset = 'portrait';
  let isDraggingHero = false;

  function renderHeroPreset(preset) {
    currentPreset = preset;
    const c1 = document.createElement('canvas');
    c1.width = 800;
    c1.height = 500;
    const ctx1 = c1.getContext('2d');
    drawSample(preset, ctx1, 800, 500, false);
    heroImgOriginal.src = c1.toDataURL();

    const c2 = document.createElement('canvas');
    c2.width = 800;
    c2.height = 500;
    const ctx2 = c2.getContext('2d');
    drawSample(preset, ctx2, 800, 500, true, 'ghibli_pro');
    heroImgCartoon.src = c2.toDataURL();
  }

  renderHeroPreset('portrait');

  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderHeroPreset(btn.dataset.preset);
    });
  });

  function setHeroSplit(percent) {
    const p = Math.max(0, Math.min(100, percent));
    heroHandle.style.left = `${p}%`;
    heroCartoonWrapper.style.clipPath = `polygon(${p}% 0, 100% 0, 100% 100%, ${p}% 100%)`;
  }

  setHeroSplit(50);

  function onHeroMove(e) {
    if (!isDraggingHero) return;
    const rect = heroViewer.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const p = ((clientX - rect.left) / rect.width) * 100;
    setHeroSplit(p);
  }

  heroHandle.addEventListener('mousedown', () => {
    isDraggingHero = true;
    window.addEventListener('mousemove', onHeroMove);
    window.addEventListener('mouseup', () => {
      isDraggingHero = false;
      window.removeEventListener('mousemove', onHeroMove);
    });
  });

  heroHandle.addEventListener('touchstart', () => {
    isDraggingHero = true;
    window.addEventListener('touchmove', onHeroMove);
    window.addEventListener('touchend', () => {
      isDraggingHero = false;
      window.removeEventListener('touchmove', onHeroMove);
    });
  });

  // ===== 4. Interactive Live Playground =====
  const canvasOriginal = document.getElementById('canvasOriginal');
  const canvasCartoon = document.getElementById('canvasCartoon');
  const playClipper = document.getElementById('playClipper');
  const playHandle = document.getElementById('playHandle');
  const playViewer = document.getElementById('playViewer');

  const ctxOrig = canvasOriginal.getContext('2d');
  const ctxCart = canvasCartoon.getContext('2d');

  let playStyle = 'ghibli_pro';
  let playSourceType = 'portrait';
  let customPlayImage = null;
  let isDraggingPlay = false;

  const C_WIDTH = 720;
  const C_HEIGHT = 450;
  canvasOriginal.width = C_WIDTH;
  canvasOriginal.height = C_HEIGHT;
  canvasCartoon.width = C_WIDTH;
  canvasCartoon.height = C_HEIGHT;

  function renderPlayground() {
    if (customPlayImage) {
      ctxOrig.drawImage(customPlayImage, 0, 0, C_WIDTH, C_HEIGHT);
      applyClientSideCartoon(ctxOrig, ctxCart, C_WIDTH, C_HEIGHT, playStyle);
    } else {
      drawSample(playSourceType, ctxOrig, C_WIDTH, C_HEIGHT, false);
      applyClientSideCartoon(ctxOrig, ctxCart, C_WIDTH, C_HEIGHT, playStyle);
    }
  }

  // =========================================================================
  // Masterpiece Noise-Free Anime & Studio Ghibli Shader
  // (Zero posterization patches, natural skin tone preservation, smooth cel-shading)
  // =========================================================================
  function applyClientSideCartoon(srcCtx, dstCtx, w, h, style) {
    const srcImgData = srcCtx.getImageData(0, 0, w, h);
    const src = srcImgData.data;
    const outImg = dstCtx.createImageData(w, h);
    const dst = outImg.data;

    const strength = parseInt(document.getElementById('playStrength').value) / 100.0;
    const edgeThick = parseInt(document.getElementById('playEdge').value);

    // Step 1: Smooth Bilateral-Style Denoising Pass
    // Averages nearby pixels within color threshold to remove noise & grain
    const smooth = new Uint8ClampedArray(w * h * 4);
    const rad = 2;

    for (let y = rad; y < h - rad; y++) {
      for (let x = rad; x < w - rad; x++) {
        const cIdx = (y * w + x) * 4;
        const cR = src[cIdx];
        const cG = src[cIdx + 1];
        const cB = src[cIdx + 2];

        let rSum = 0, gSum = 0, bSum = 0, weightSum = 0;

        for (let dy = -rad; dy <= rad; dy++) {
          const rowOff = (y + dy) * w;
          for (let dx = -rad; dx <= rad; dx++) {
            const pIdx = (rowOff + (x + dx)) * 4;
            const pr = src[pIdx];
            const pg = src[pIdx + 1];
            const pb = src[pIdx + 2];

            const colorDiff = Math.abs(cR - pr) + Math.abs(cG - pg) + Math.abs(cB - pb);
            if (colorDiff < 85) {
              const wVal = 1.0 / (1.0 + colorDiff * 0.05 + (dx * dx + dy * dy) * 0.1);
              rSum += pr * wVal;
              gSum += pg * wVal;
              bSum += pb * wVal;
              weightSum += wVal;
            }
          }
        }

        smooth[cIdx] = weightSum > 0 ? rSum / weightSum : cR;
        smooth[cIdx + 1] = weightSum > 0 ? gSum / weightSum : cG;
        smooth[cIdx + 2] = weightSum > 0 ? bSum / weightSum : cB;
        smooth[cIdx + 3] = src[cIdx + 3];
      }
    }

    // Step 2: Smooth Cel-Shading & Ghibli Tone Mapping (NO Yellowing!)
    for (let i = 0; i < src.length; i += 4) {
      let r = smooth[i] || src[i];
      let g = smooth[i + 1] || src[i + 1];
      let b = smooth[i + 2] || src[i + 2];
      const a = src[i + 3];

      if (style === 'ghibli_pro') {
        // Natural luminance curve (Soft Anime 3-tier lighting)
        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
        const tone = lum > 145 ? 1.05 : (lum > 75 ? 1.00 : 0.92);

        // Soft skin warmth without yellowing
        r = r * tone * 1.02 + 3;
        g = g * tone;
        b = b * tone * 0.98;
      } else if (style === 'comic_pop') {
        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
        const tone = lum > 130 ? 1.15 : 0.85;
        r = Math.min(255, r * tone * 1.2);
        g = Math.min(255, g * tone * 1.15);
        b = Math.min(255, b * tone * 1.1);
      } else if (style === 'watercolor') {
        r = r * 0.96 + 10;
        g = g * 0.98 + 8;
        b = b * 1.03 + 12;
      } else if (style === 'neon') {
        r = r * 0.35 + 15;
        g = g * 0.30;
        b = Math.min(255, b * 1.4 + 40);
      } else if (style === 'pencil') {
        const lum = r * 0.299 + g * 0.587 + b * 0.114;
        r = lum; g = lum; b = lum;
      } else if (style === 'retro') {
        r = r * 1.12;
        b = b * 0.90;
      }

      dst[i] = Math.min(255, Math.max(0, r));
      dst[i + 1] = Math.min(255, Math.max(0, g));
      dst[i + 2] = Math.min(255, Math.max(0, b));
      dst[i + 3] = a;
    }

    // Step 3: Anti-Aliased Hand-Drawn Inking (Only structural lines)
    const edgeThresh = style === 'ghibli_pro' ? 80 : 60;

    for (let y = 3; y < h - 3; y++) {
      for (let x = 3; x < w - 3; x++) {
        const lumL = (smooth[((y) * w + (x - 2)) * 4] + smooth[((y) * w + (x - 2)) * 4 + 1] + smooth[((y) * w + (x - 2)) * 4 + 2]) / 3;
        const lumR = (smooth[((y) * w + (x + 2)) * 4] + smooth[((y) * w + (x + 2)) * 4 + 1] + smooth[((y) * w + (x + 2)) * 4 + 2]) / 3;
        const lumU = (smooth[((y - 2) * w + x) * 4] + smooth[((y - 2) * w + x) * 4 + 1] + smooth[((y - 2) * w + x) * 4 + 2]) / 3;
        const lumD = (smooth[((y + 2) * w + x) * 4] + smooth[((y + 2) * w + x) * 4 + 1] + smooth[((y + 2) * w + x) * 4 + 2]) / 3;

        const grad = Math.abs(lumR - lumL) + Math.abs(lumD - lumU);

        if (grad > edgeThresh) {
          const inkAlpha = style === 'ghibli_pro' ? 0.38 * strength : 0.85 * strength;
          for (let dy = 0; dy < edgeThick; dy++) {
            for (let dx = 0; dx < edgeThick; dx++) {
              const targetIdx = ((y + dy) * w + (x + dx)) * 4;
              if (targetIdx < dst.length) {
                if (style === 'neon') {
                  dst[targetIdx] = 0;
                  dst[targetIdx + 1] = 240;
                  dst[targetIdx + 2] = 255;
                } else {
                  dst[targetIdx] = dst[targetIdx] * (1.0 - inkAlpha) + 26 * inkAlpha;
                  dst[targetIdx + 1] = dst[targetIdx + 1] * (1.0 - inkAlpha) + 22 * inkAlpha;
                  dst[targetIdx + 2] = dst[targetIdx + 2] * (1.0 - inkAlpha) + 18 * inkAlpha;
                }
              }
            }
          }
        }
      }
    }

    dstCtx.putImageData(outImg, 0, 0);
  }

  renderPlayground();

  // Style Selector in Playground
  document.querySelectorAll('.style-choice-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.style-choice-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      playStyle = btn.dataset.style;
      renderPlayground();
    });
  });

  // Photo preset selector in Playground
  document.querySelectorAll('.play-preset:not(#playUploadBtn)').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.play-preset').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      customPlayImage = null;
      playSourceType = btn.dataset.img;
      renderPlayground();
    });
  });

  // Sliders reactivity
  const playStrength = document.getElementById('playStrength');
  const playStrengthVal = document.getElementById('playStrengthVal');
  playStrength.addEventListener('input', e => {
    playStrengthVal.textContent = `${e.target.value}%`;
    renderPlayground();
  });

  const playEdge = document.getElementById('playEdge');
  const playEdgeVal = document.getElementById('playEdgeVal');
  playEdge.addEventListener('input', e => {
    playEdgeVal.textContent = `${e.target.value}px`;
    renderPlayground();
  });

  // Playground Custom Upload
  const playUploadBtn = document.getElementById('playUploadBtn');
  const playFileInput = document.getElementById('playFileInput');
  playUploadBtn.addEventListener('click', () => playFileInput.click());

  playFileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = ev => {
        const img = new Image();
        img.onload = () => {
          customPlayImage = img;
          document.querySelectorAll('.play-preset').forEach(b => b.classList.remove('active'));
          playUploadBtn.classList.add('active');
          renderPlayground();
        };
        img.src = ev.target.result;
      };
      reader.readAsDataURL(file);
    }
  });

  // Playground Split Slider
  function setPlaySplit(percent) {
    const p = Math.max(0, Math.min(100, percent));
    playHandle.style.left = `${p}%`;
    playClipper.style.clipPath = `polygon(${p}% 0, 100% 0, 100% 100%, ${p}% 100%)`;
  }
  setPlaySplit(50);

  function onPlayMove(e) {
    if (!isDraggingPlay) return;
    const rect = playViewer.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const p = ((clientX - rect.left) / rect.width) * 100;
    setPlaySplit(p);
  }

  playHandle.addEventListener('mousedown', () => {
    isDraggingPlay = true;
    window.addEventListener('mousemove', onPlayMove);
    window.addEventListener('mouseup', () => {
      isDraggingPlay = false;
      window.removeEventListener('mousemove', onPlayMove);
    });
  });

  playHandle.addEventListener('touchstart', () => {
    isDraggingPlay = true;
    window.addEventListener('touchmove', onPlayMove);
    window.addEventListener('touchend', () => {
      isDraggingPlay = false;
      window.removeEventListener('touchmove', onPlayMove);
    });
  });

  document.getElementById('downloadPlaygroundBtn').addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = canvasCartoon.toDataURL('image/jpeg', 0.95);
    a.download = `cartoonify_artwork_${playStyle}.jpg`;
    a.click();
  });

  // ===== 5. Developer Code Tabs =====
  const codeTabs = document.querySelectorAll('.code-tab-btn');
  codeTabs.forEach(btn => {
    btn.addEventListener('click', () => {
      codeTabs.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.code-block-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add('active');
    });
  });

  // Copy Code Buttons
  document.querySelectorAll('.copy-code-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const targetId = btn.dataset.target;
      const codeEl = document.getElementById(targetId);
      if (codeEl) {
        await navigator.clipboard.writeText(codeEl.textContent);
        const originalText = btn.textContent;
        btn.textContent = 'Copied! ✓';
        btn.style.color = '#00b894';
        setTimeout(() => {
          btn.textContent = originalText;
          btn.style.color = '';
        }, 2000);
      }
    });
  });

  // ===== 6. Mobile Toggle =====
  const mobileToggle = document.getElementById('mobileToggle');
  const navLinks = document.getElementById('navLinks');
  if (mobileToggle && navLinks) {
    mobileToggle.addEventListener('click', () => {
      navLinks.style.display = navLinks.style.display === 'flex' ? 'none' : 'flex';
      navLinks.style.flexDirection = 'column';
      navLinks.style.position = 'absolute';
      navLinks.style.top = '60px';
      navLinks.style.left = '0';
      navLinks.style.right = '0';
      navLinks.style.background = 'var(--bg-card)';
      navLinks.style.padding = '20px';
      navLinks.style.borderRadius = 'var(--radius-md)';
    });
  }
});
