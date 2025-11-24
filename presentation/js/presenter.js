/* Minimal presenter script: loads section fragments, wires keyboard navigation, and manages focus/ARIA */
(function () {
  const slidesContainer = document.getElementById('slides');
  const status = document.getElementById('slideStatus');
  const prevBtn = document.getElementById('prevBtn');
  const nextBtn = document.getElementById('nextBtn');

  async function loadSections(list) {
    for (const path of list) {
      try {
        const res = await fetch(path);
        if (!res.ok) {
          console.warn('Failed to load', path, res.status);
          // insert a placeholder note
          const note = document.createElement('section');
          note.className = 'slide p-16 col bg-base text-foreground';
          note.innerHTML = `<h2 class="accent">Missing section</h2><p class="text-secondary">Could not load ${path} (status ${res.status}).</p>`;
          slidesContainer.appendChild(note);
          continue;
        }
        const text = await res.text();
        const wrapper = document.createElement('div');
        wrapper.innerHTML = text;
        // Append child section elements
        Array.from(wrapper.children).forEach(child => {
          if (child.matches && child.matches('section.slide')) {
            slidesContainer.appendChild(child);
          }
        });
      } catch (err) {
        console.error('Error loading', path, err);
        const note = document.createElement('section');
        note.className = 'slide p-16 col bg-base text-foreground';
        note.innerHTML = `<h2 class="accent">Load error</h2><p class="text-secondary">Error loading ${path}</p>`;
        slidesContainer.appendChild(note);
      }
    }
    indexSlides();
    // If KaTeX auto-render is available, render math in the slides container
    if (window.renderMathInElement && typeof window.renderMathInElement === 'function') {
      try {
        window.renderMathInElement(slidesContainer, {
          // typical delimiters
          delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '\\(', right: '\\)', display: false}
          ],
          throwOnError: false
        });
      } catch (e) {
        console.warn('KaTeX render error', e);
      }
    }
    showSlide(0);
  }

  function indexSlides() {
    const slides = Array.from(slidesContainer.querySelectorAll('section.slide'));
    slides.forEach((s, i) => {
      s.setAttribute('data-slide', i + 1);
      s.setAttribute('role', 'group');
      const head = s.querySelector('h1,h2,h3');
      const title = head ? head.textContent.trim() : `Slide ${i + 1}`;
      s.setAttribute('aria-label', `Slide ${i + 1} of ${slides.length}: ${title}`);
      s.setAttribute('tabindex', '-1');
    });
  }

  let current = 0;

  function slides() { return Array.from(slidesContainer.querySelectorAll('section.slide')); }

  function clamp(n) {
    const s = slides();
    return Math.max(0, Math.min(n, s.length - 1));
  }

  function showSlide(index) {
    const s = slides();
    if (!s.length) return;
    index = clamp(index);
    s.forEach((el, i) => {
      el.classList.toggle('is-active', i === index);
    });
    current = index;
    const active = s[current];
    if (active) {
      active.focus();
      const h = active.querySelector('h1,h2,h3');
      const title = h ? h.textContent.trim() : `Slide ${current+1}`;
      document.title = `${title} — GraphSAGE`;
      if (status) status.textContent = `Slide ${current + 1} of ${s.length}: ${title}`;
      // visible progress element
      const prog = document.getElementById('progress');
      if (prog) prog.textContent = `Slide ${current + 1} of ${s.length}: ${title}`;
    }
  }

  function next() { showSlide(current + 1); }
  function prev() { showSlide(current - 1); }

  document.addEventListener('keydown', (ev) => {
    const key = ev.key;
    if (key === 'ArrowRight' || key === 'PageDown' || key === ' ') {
      ev.preventDefault(); next();
    } else if (key === 'ArrowLeft' || key === 'PageUp') {
      ev.preventDefault(); prev();
    } else if (key === 'Home') { ev.preventDefault(); showSlide(0); }
    else if (key === 'End') { ev.preventDefault(); const s=slides(); showSlide(s.length-1); }
  });

  prevBtn.addEventListener('click', prev);
  nextBtn.addEventListener('click', next);

  // Start loading (expects window.PRESENTATION_SLIDES to be an ordered array of file paths)
  if (window.PRESENTATION_SLIDES && window.PRESENTATION_SLIDES.length) {
    loadSections(window.PRESENTATION_SLIDES);
  } else {
    console.warn('No slides configured to load. Set window.PRESENTATION_SLIDES in index.html');
  }

})();
