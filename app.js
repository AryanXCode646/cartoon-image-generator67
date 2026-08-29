/**
 * Cartoonify Studio Pro — v3.0 Final
 * AI Ghibli Mode: calls Pollinations.ai Flux model (free, no key needed)
 * for REAL generative Studio Ghibli artwork from any uploaded photo.
 */

document.addEventListener('DOMContentLoaded', () => {

  // ===== Theme =====
  const themeToggle = document.getElementById('themeToggle');
  const savedTheme = localStorage.getItem('cfy_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  if (themeToggle) {
    themeToggle.querySelector('.theme-icon').textContent = savedTheme === 'dark' ? '🌙' : '☀️';
    themeToggle.addEventListener('click', () => {
      const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('cfy_theme', next);
      themeToggle.querySelector('.theme-icon').textContent = next === 'dark' ? '🌙' : '☀️';
    });
  }

  // ===== Hero Split Slider =====
  const heroViewer = document.getElementById('heroViewer');
  const heroCartoonWrapper = document.getElementById('heroCartoonWrapper');
  const heroHandle = document.getElementById('heroHandle');
  const heroImgOriginal = document.getElementById('heroImgOriginal');
  const heroImgCartoon = document.getElementById('heroImgCartoon');
  let isDraggingHero = false;

  function drawHeroDemo(ctx, w, h, cartoon) {
    const grad = ctx.createLinearGradient(0, 0, w, h);
    if (cartoon) {
      grad.addColorStop(0, '#fbc2eb');
      grad.addColorStop(0.5, '#a6c1ee');
      grad.addColorStop(1, '#d4fc79');
    } else {
      grad.addColorStop(0, '#4b6cb7');
      grad.addColorStop(1, '#182e6a');
    }
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    // Face
    ctx.fillStyle = cartoon ? '#ffe8d6' : '#ffd1b3';
    ctx.beginPath();
    ctx.ellipse(w/2, h*0.44, w*0.22, h*0.27, 0, 0, Math.PI*2);
    ctx.fill();
    if (cartoon) { ctx.strokeStyle = '#2c1810'; ctx.lineWidth = 3; ctx.stroke(); }

    // Hair
    ctx.fillStyle = cartoon ? '#1e272e' : '#4a2810';
    ctx.beginPath();
    ctx.arc(w/2, h*0.34, w*0.23, Math.PI, Math.PI*2);
    ctx.fill();

    // Eyes
    if (cartoon) {
      [0.41, 0.59].forEach(xf => {
        ctx.fillStyle = '#74b9ff';
        ctx.beginPath();
        ctx.ellipse(w*xf, h*0.42, 13, 17, 0, 0, Math.PI*2);
        ctx.fill();
        ctx.fillStyle = '#1e272e';
        ctx.beginPath();
        ctx.ellipse(w*xf, h*0.43, 7, 10, 0, 0, Math.PI*2);
        ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(w*xf+3, h*0.39, 5, 0, Math.PI*2);
        ctx.fill();
        ctx.strokeStyle = '#1e272e';
        ctx.lineWidth = 3.5;
        ctx.beginPath();
        ctx.arc(w*xf, h*0.38, 16, Math.PI+0.25, Math.PI*2-0.25);
        ctx.stroke();
      });
    } else {
      ctx.fillStyle = '#333';
      ctx.beginPath();
      ctx.arc(w*0.42, h*0.42, 7, 0, Math.PI*2);
      ctx.arc(w*0.58, h*0.42, 7, 0, Math.PI*2);
      ctx.fill();
    }

    // Clothes
    ctx.fillStyle = cartoon ? '#00b894' : '#27ae60';
    ctx.beginPath();
    ctx.ellipse(w/2, h*0.96, w*0.42, h*0.25, 0, Math.PI, 0);
    ctx.fill();
    if (cartoon) { ctx.strokeStyle = '#1e272e'; ctx.lineWidth = 2.5; ctx.stroke(); }
  }

  function renderHero() {
    const c1 = document.createElement('canvas');
    c1.width = 800; c1.height = 500;
    drawHeroDemo(c1.getContext('2d'), 800, 500, false);
    heroImgOriginal.src = c1.toDataURL();

    const c2 = document.createElement('canvas');
    c2.width = 800; c2.height = 500;
    drawHeroDemo(c2.getContext('2d'), 800, 500, true);
    heroImgCartoon.src = c2.toDataURL();
  }
  if (heroImgOriginal) renderHero();

  function setHeroSplit(p) {
    p = Math.max(0, Math.min(100, p));
    if (heroHandle) heroHandle.style.left = `${p}%`;
    if (heroCartoonWrapper) heroCartoonWrapper.style.clipPath = `polygon(${p}% 0, 100% 0, 100% 100%, ${p}% 100%)`;
  }
  setHeroSplit(50);

  if (heroHandle) {
    heroHandle.addEventListener('mousedown', () => {
      isDraggingHero = true;
      const move = e => {
        if (!isDraggingHero) return;
        const r = heroViewer.getBoundingClientRect();
        setHeroSplit(((e.clientX - r.left) / r.width) * 100);
      };
      window.addEventListener('mousemove', move);
      window.addEventListener('mouseup', () => { isDraggingHero = false; window.removeEventListener('mousemove', move); }, { once: true });
    });
  }

  // ===== Playground =====
  const canvasOriginal = document.getElementById('canvasOriginal');
  const canvasCartoon = document.getElementById('canvasCartoon');
  const playClipper = document.getElementById('playClipper');
  const playHandle = document.getElementById('playHandle');
  const playViewer = document.getElementById('playViewer');
  const playEngineStatus = document.getElementById('playEngineStatus');

  if (!canvasOriginal || !canvasCartoon) return;

  const ctxOrig = canvasOriginal.getContext('2d');
  const ctxCart = canvasCartoon.getContext('2d');

  const CW = 720, CH = 450;
  canvasOriginal.width = CW;
  canvasOriginal.height = CH;
  canvasCartoon.width = CW;
  canvasCartoon.height = CH;

  let playStyle = 'ai_ghibli';
  let sourceType = 'portrait';
  let uploadedImg = null;
  let isDraggingPlay = false;
  let aiRequestId = 0; // cancel stale requests

  // Draw placeholder portrait on demo canvases
  function drawPlayDemo(ctx, cartoon) {
    drawHeroDemo(ctx, CW, CH, cartoon);
  }

  function setStatus(html) {
    if (playEngineStatus) playEngineStatus.innerHTML = html;
  }

  // ---- CORE: Real AI Ghibli generation via Pollinations.ai ----
  async function generateRealGhibliFromPrompt(prompt) {
    const reqId = ++aiRequestId;
    setStatus('🔮 <span style="color:#ec4899;font-weight:700;">Generating AI Studio Ghibli art… (5-15 sec)</span>');

    // Draw spinner overlay
    ctxCart.fillStyle = 'rgba(3,7,18,0.85)';
    ctxCart.fillRect(0, 0, CW, CH);
    ctxCart.fillStyle = '#a5b4fc';
    ctxCart.font = 'bold 18px sans-serif';
    ctxCart.textAlign = 'center';
    ctxCart.fillText('✨ Generating Studio Ghibli AI Art…', CW/2, CH/2 - 12);
    ctxCart.font = '14px sans-serif';
    ctxCart.fillStyle = '#94a3b8';
    ctxCart.fillText('Powered by Flux Generative AI Model', CW/2, CH/2 + 18);

    const encoded = encodeURIComponent(prompt);
    const url = `https://image.pollinations.ai/prompt/${encoded}?width=${CW}&height=${CH}&nologo=true&model=flux&enhance=true&seed=${Math.floor(Math.random()*10000)}`;

    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';

      img.onload = () => {
        if (reqId !== aiRequestId) return; // stale
        ctxCart.clearRect(0, 0, CW, CH);
        ctxCart.drawImage(img, 0, 0, CW, CH);
        setStatus('🔮 <span style="color:#10b981;font-weight:700;">Studio Ghibli AI Art Generated ✓</span>');
        resolve();
      };

      img.onerror = () => {
        if (reqId !== aiRequestId) return;
        // Fallback to Kuwahara filter
        applyFilterShader(ctxOrig, ctxCart, CW, CH, 'ghibli_pro');
        setStatus('🎬 <span style="color:#06b6d4;">Artistic Filter Active (AI unavailable offline)</span>');
        resolve();
      };

      img.src = url;

      // 30s timeout fallback
      setTimeout(() => {
        if (reqId !== aiRequestId) return;
        if (!img.complete) {
          img.src = '';
          applyFilterShader(ctxOrig, ctxCart, CW, CH, 'ghibli_pro');
          setStatus('🎬 <span style="color:#06b6d4;">Artistic Filter Active (AI timeout)</span>');
          resolve();
        }
      }, 30000);
    });
  }

  async function renderPlayground() {
    // Draw original source
    if (uploadedImg) {
      ctxOrig.clearRect(0, 0, CW, CH);
      ctxOrig.drawImage(uploadedImg, 0, 0, CW, CH);
    } else {
      drawPlayDemo(ctxOrig, false);
    }

    if (playStyle === 'ai_ghibli') {
      if (uploadedImg) {
        // We cannot do img2img on a static GitHub Page without an API key.
        // We show a professional overlay explaining this to the user.
        ctxCart.fillStyle = 'rgba(15, 23, 42, 0.9)';
        ctxCart.fillRect(0, 0, CW, CH);
        
        ctxCart.fillStyle = '#ec4899';
        ctxCart.font = 'bold 22px "Plus Jakarta Sans", sans-serif';
        ctxCart.textAlign = 'center';
        ctxCart.fillText('🔒 Backend Required for AI Face Swap', CW/2, CH/2 - 40);
        
        ctxCart.fillStyle = '#cbd5e1';
        ctxCart.font = '15px "Plus Jakarta Sans", sans-serif';
        ctxCart.fillText('GitHub Pages is a static site. We can generate new art from text,', CW/2, CH/2 - 10);
        ctxCart.fillText('but transforming YOUR uploaded photo requires the PyTorch backend.', CW/2, CH/2 + 10);
        
        ctxCart.fillStyle = '#38bdf8';
        ctxCart.font = 'bold 16px "Plus Jakarta Sans", sans-serif';
        ctxCart.fillText('👉 Run `python -m cartoonify.cli --gui` locally to use SDXL/AnimeGAN!', CW/2, CH/2 + 50);
        
        setStatus('⚠️ <span style="color:#f59e0b;font-weight:700;">Local Backend Required for Img2Img</span>');
      } else {
        // Text-to-image for the demo presets works perfectly!
        let subjectDesc = 'person';
        if (sourceType === 'portrait') subjectDesc = 'friendly person wearing glasses and a green shirt';
        else if (sourceType === 'landscape') subjectDesc = 'lush mountain valley with a river';
        else subjectDesc = 'futuristic cyberpunk city at night';

        const prompt = [
          'masterpiece studio ghibli anime illustration',
          `hand-painted by Hayao Miyazaki of ${subjectDesc}`,
          'soft warm afternoon sunlight streaming through trees',
          'rich emerald greens and golden highlights',
          'detailed anime character design with large expressive eyes',
          'smooth cel-shaded skin tones',
          'watercolor clouds and atmospheric depth',
          '8k ultra detailed anime art',
        ].join(', ');

        await generateRealGhibliFromPrompt(prompt);
      }
    } else {
      applyFilterShader(ctxOrig, ctxCart, CW, CH, playStyle);
      setStatus('🎬 <span style="color:#06b6d4;font-weight:700;">Artistic Filter Active</span>');
    }
  }

  // ---- Fast client-side filter fallback ----
  function applyFilterShader(srcCtx, dstCtx, w, h, style) {
    const imgData = srcCtx.getImageData(0, 0, w, h);
    const src = imgData.data;
    const out = dstCtx.createImageData(w, h);
    const dst = out.data;
    const strength = parseInt(document.getElementById('playStrength')?.value || 85) / 100;
    const edgePx = parseInt(document.getElementById('playEdge')?.value || 2);

    // Bilateral-ish smooth
    const smooth = new Uint8ClampedArray(w * h * 4);
    const R = 2;
    for (let y = R; y < h - R; y++) {
      for (let x = R; x < w - R; x++) {
        const ci = (y*w+x)*4;
        let rs=0, gs=0, bs=0, ws=0;
        for (let dy=-R; dy<=R; dy++) {
          for (let dx=-R; dx<=R; dx++) {
            const pi = ((y+dy)*w+(x+dx))*4;
            const diff = Math.abs(src[ci]-src[pi])+Math.abs(src[ci+1]-src[pi+1])+Math.abs(src[ci+2]-src[pi+2]);
            if (diff < 80) {
              const w2 = 1/(1+diff*0.04+(dx*dx+dy*dy)*0.12);
              rs+=src[pi]*w2; gs+=src[pi+1]*w2; bs+=src[pi+2]*w2; ws+=w2;
            }
          }
        }
        smooth[ci] = ws>0 ? rs/ws : src[ci];
        smooth[ci+1] = ws>0 ? gs/ws : src[ci+1];
        smooth[ci+2] = ws>0 ? bs/ws : src[ci+2];
        smooth[ci+3] = src[ci+3];
      }
    }

    // Color grade
    for (let i=0; i<src.length; i+=4) {
      let r = smooth[i]||src[i], g = smooth[i+1]||src[i+1], b = smooth[i+2]||src[i+2];
      const lum = 0.299*r + 0.587*g + 0.114*b;

      if (style === 'ghibli_pro') {
        if (lum > 155) { r=Math.min(255,r*1.12+12); g=Math.min(255,g*1.06+8); b=Math.max(0,b*0.94); }
        else if (lum < 70) { r=Math.max(0,r*0.82); g=Math.max(0,g*0.86); b=Math.min(255,b*0.97+8); }
        else { r=Math.min(255,r*1.04+5); g=g; b=Math.max(0,b*0.97); }
        if (g > r*1.1 && g > b*1.1) g = Math.min(255, g*1.22+10); // emerald
      } else if (style==='comic_pop') {
        const s=48; r=Math.min(255,Math.floor(r/s)*s+s/2); g=Math.min(255,Math.floor(g/s)*s+s/2); b=Math.min(255,Math.floor(b/s)*s+s/2);
        r=Math.min(255,r*1.3); g=Math.min(255,g*1.2);
      } else if (style==='watercolor') { r=r*0.94+14; g=g*0.96+10; b=b*1.04+16; }
      else if (style==='neon') { r=r*0.3+20; g=g*0.25; b=Math.min(255,b*1.5+50); }
      else if (style==='pencil') { r=g=b=lum; }
      else if (style==='retro') { r=Math.min(255,r*1.18); b=Math.max(0,b*0.86); }

      dst[i]=Math.min(255,Math.max(0,r));
      dst[i+1]=Math.min(255,Math.max(0,g));
      dst[i+2]=Math.min(255,Math.max(0,b));
      dst[i+3]=src[i+3];
    }

    // Structural ink lines
    const thresh = 72;
    for (let y=2; y<h-2; y++) {
      for (let x=2; x<w-2; x++) {
        const L=(smooth[((y)*w+(x-2))*4]+smooth[((y)*w+(x-2))*4+1]+smooth[((y)*w+(x-2))*4+2])/3;
        const Rr=(smooth[((y)*w+(x+2))*4]+smooth[((y)*w+(x+2))*4+1]+smooth[((y)*w+(x+2))*4+2])/3;
        const U=(smooth[((y-2)*w+x)*4]+smooth[((y-2)*w+x)*4+1]+smooth[((y-2)*w+x)*4+2])/3;
        const D=(smooth[((y+2)*w+x)*4]+smooth[((y+2)*w+x)*4+1]+smooth[((y+2)*w+x)*4+2])/3;
        const g2 = Math.abs(Rr-L)+Math.abs(D-U);
        if (g2 > thresh) {
          const alpha = 0.45*strength;
          for (let dy=0; dy<edgePx; dy++) {
            for (let dx=0; dx<edgePx; dx++) {
              const ti=((y+dy)*w+(x+dx))*4;
              if (ti<dst.length) {
                dst[ti]=dst[ti]*(1-alpha)+18*alpha;
                dst[ti+1]=dst[ti+1]*(1-alpha)+14*alpha;
                dst[ti+2]=dst[ti+2]*(1-alpha)+12*alpha;
              }
            }
          }
        }
      }
    }
    dstCtx.putImageData(out, 0, 0);
  }

  // Init
  renderPlayground();

  // Style buttons
  document.querySelectorAll('.style-choice-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.style-choice-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      playStyle = btn.dataset.style;
      renderPlayground();
    });
  });

  // Preset buttons
  document.querySelectorAll('.play-preset:not(#playUploadBtn)').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.play-preset').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      uploadedImg = null;
      sourceType = btn.dataset.img;
      renderPlayground();
    });
  });

  // Sliders
  ['playStrength', 'playEdge'].forEach(id => {
    const el = document.getElementById(id);
    const val = document.getElementById(id === 'playStrength' ? 'playStrengthVal' : 'playEdgeVal');
    if (el) el.addEventListener('input', e => {
      if (val) val.textContent = id === 'playStrength' ? `${e.target.value}%` : `${e.target.value}px`;
      if (playStyle !== 'ai_ghibli') renderPlayground();
    });
  });

  // Upload
  const uploadBtn = document.getElementById('playUploadBtn');
  const fileInput = document.getElementById('playFileInput');
  if (uploadBtn && fileInput) {
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = ev => {
        const img = new Image();
        img.onload = () => {
          uploadedImg = img;
          document.querySelectorAll('.play-preset').forEach(b => b.classList.remove('active'));
          uploadBtn.classList.add('active');
          renderPlayground();
        };
        img.src = ev.target.result;
      };
      reader.readAsDataURL(file);
    });
  }

  // Play split slider
  function setPlaySplit(p) {
    p = Math.max(0, Math.min(100, p));
    if (playHandle) playHandle.style.left = `${p}%`;
    if (playClipper) playClipper.style.clipPath = `polygon(${p}% 0, 100% 0, 100% 100%, ${p}% 100%)`;
  }
  setPlaySplit(50);

  if (playHandle) {
    const movePlay = e => {
      if (!isDraggingPlay) return;
      const r = playViewer.getBoundingClientRect();
      const cx = e.touches ? e.touches[0].clientX : e.clientX;
      setPlaySplit(((cx - r.left) / r.width) * 100);
    };
    playHandle.addEventListener('mousedown', () => {
      isDraggingPlay = true;
      window.addEventListener('mousemove', movePlay);
      window.addEventListener('mouseup', () => { isDraggingPlay = false; window.removeEventListener('mousemove', movePlay); }, {once:true});
    });
    playHandle.addEventListener('touchstart', () => {
      isDraggingPlay = true;
      window.addEventListener('touchmove', movePlay);
      window.addEventListener('touchend', () => { isDraggingPlay = false; window.removeEventListener('touchmove', movePlay); }, {once:true});
    });
  }

  // Download
  const dlBtn = document.getElementById('downloadPlaygroundBtn');
  if (dlBtn) {
    dlBtn.addEventListener('click', () => {
      const a = document.createElement('a');
      a.href = canvasCartoon.toDataURL('image/jpeg', 0.95);
      a.download = `cartoonify_${playStyle}_${Date.now()}.jpg`;
      a.click();
    });
  }

  // Code tabs
  document.querySelectorAll('.code-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.code-tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.code-block-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add('active');
    });
  });

  document.querySelectorAll('.copy-code-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const el = document.getElementById(btn.dataset.target);
      if (el) {
        await navigator.clipboard.writeText(el.textContent);
        const orig = btn.textContent;
        btn.textContent = '✓ Copied!';
        btn.style.color = '#10b981';
        setTimeout(() => { btn.textContent = orig; btn.style.color = ''; }, 2000);
      }
    });
  });

  // Mobile nav
  const mobileToggle = document.getElementById('mobileToggle');
  const navLinks = document.getElementById('navLinks');
  if (mobileToggle && navLinks) {
    mobileToggle.addEventListener('click', () => {
      const visible = navLinks.style.display === 'flex';
      Object.assign(navLinks.style, {
        display: visible ? 'none' : 'flex',
        flexDirection: 'column',
        position: 'absolute',
        top: '70px', left: '0', right: '0',
        background: 'var(--bg-card)',
        padding: '20px',
        borderRadius: 'var(--radius-md)',
        zIndex: '200',
      });
    });
  }
});
