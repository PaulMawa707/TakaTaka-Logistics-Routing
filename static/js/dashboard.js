function pad(n) {
  return String(n).padStart(2, '0');
}

function tickClock() {
  const now = new Date();
  const utc = now.getTime() + now.getTimezoneOffset() * 60000;
  const eat = new Date(utc + 3 * 3600000);
  const el = document.getElementById('clock-time');
  if (!el) return;
  el.textContent =
    `${eat.getFullYear()}-${pad(eat.getMonth() + 1)}-${pad(eat.getDate())} ` +
    `${pad(eat.getHours())}:${pad(eat.getMinutes())}:${pad(eat.getSeconds())}`;
}

function showTab(name) {
  document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
  document.querySelectorAll('[data-tab]').forEach((n) => n.classList.remove('active'));
  document.querySelectorAll('.mobile-tab-item').forEach((b) => b.removeAttribute('aria-current'));

  const panel = document.getElementById('tab-' + name);
  if (panel) panel.classList.add('active');
  document.querySelectorAll(`[data-tab="${name}"]`).forEach((n) => n.classList.add('active'));

  const mob = document.querySelector(`.mobile-tab-bar [data-tab="${name}"]`);
  if (mob) {
    mob.setAttribute('aria-current', 'page');
    mob.scrollIntoView({ inline: 'center', behavior: 'smooth', block: 'nearest' });
  }

  const url = new URL(window.location.href);
  url.searchParams.set('tab', name);
  window.history.replaceState({}, '', url);
}

function showToast(message, ok) {
  let toast = document.getElementById('app-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'app-toast';
    toast.className = 'toast';
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.className = 'toast show ' + (ok ? 'ok' : 'err');
  setTimeout(() => toast.classList.remove('show'), 5000);
}

function setAlert(el, message, type) {
  if (!el) return;
  el.className = 'alert ' + type;
  el.innerHTML = message;
  el.classList.remove('hidden');
}

async function submitDispatchForm(form, endpoint) {
  const alertEl = form.querySelector('.result-alert');
  const btn = form.querySelector('[type="submit"]');
  const fd = new FormData(form);
  btn.disabled = true;
  if (alertEl) {
    alertEl.className = 'alert info hidden';
    alertEl.textContent = 'Processing…';
    alertEl.classList.remove('hidden');
  }
  try {
    const res = await fetch(endpoint, { method: 'POST', body: fd, credentials: 'same-origin' });
    const data = await res.json();
    if (data.error === 0) {
      let html = `<strong>Success.</strong> ${data.message || 'Route created.'}`;
      if (data.summary) {
        html += `<br>Delivery points: ${data.summary.delivery_points} · Tonnage: ${data.summary.tonnage} · Amount: ${data.summary.amount}`;
      }
      if (data.planning_url) {
        html += `<br><a href="${data.planning_url}" target="_blank" rel="noopener">Open Wialon Logistics</a>`;
      }
      setAlert(alertEl, html, 'ok');
      showToast('Route created successfully', true);
    } else {
      setAlert(alertEl, `<strong>Failed.</strong> ${data.message || 'Unknown error'}`, 'err');
      showToast(data.message || 'Dispatch failed', false);
    }
  } catch (err) {
    setAlert(alertEl, `<strong>Error.</strong> ${err.message}`, 'err');
    showToast(err.message, false);
  } finally {
    btn.disabled = false;
  }
}

let weeklyState = { routes: [], assets: [], days: [] };

function renderWeeklyRoutes(day) {
  const tbody = document.querySelector('#weekly-routes-table tbody');
  const routeSelect = document.getElementById('weekly-route-select');
  if (!tbody || !routeSelect) return;

  const filtered = weeklyState.routes.filter((r) => r.day === day);
  tbody.innerHTML = filtered.map((r) =>
    `<tr><td>${r.route_name}</td><td>${r.stops ?? 0}</td><td>${r.total_tonnage}</td><td>${r.total_amount}</td><td>${r.source_zip}</td></tr>`
  ).join('');

  routeSelect.innerHTML = filtered.map((r) =>
    `<option value="${r.route_name}">${r.route_name}</option>`
  ).join('');

  const firstRoute = routeSelect.value || (filtered[0] && filtered[0].route_name);
  if (firstRoute) updateWeeklyStops(day, firstRoute);
}

async function updateWeeklyStops(day, routeName) {
  const stopsBody = document.querySelector('#weekly-stops-table tbody');
  if (!stopsBody || !day || !routeName) return;
  stopsBody.innerHTML = '<tr><td colspan="5">Loading stops…</td></tr>';
  try {
    const params = new URLSearchParams({ day, route_name: routeName });
    const res = await fetch(`/api/weekly/route?${params}`, { credentials: 'same-origin' });
    const data = await res.json();
    if (data.error !== 0) throw new Error(data.message || 'Could not load stops');
    const orders = data.orders || [];
    if (!orders.length) {
      stopsBody.innerHTML = '';
      return;
    }
    stopsBody.innerHTML = orders.map((o) =>
      `<tr><td>${o.PRIORITY ?? ''}</td><td>${o['CUSTOMER NAME'] ?? ''}</td><td>${o.LOCATION ?? ''}</td><td>${o.TONNAGE ?? ''}</td><td>${o.AMOUNT ?? ''}</td></tr>`
    ).join('');
  } catch (err) {
    stopsBody.innerHTML = `<tr><td colspan="5">${err.message}</td></tr>`;
  }
}

async function loadWeeklyRoutes(form) {
  const alertEl = form.querySelector('.result-alert');
  const btn = form.querySelector('[type="submit"]');
  const fd = new FormData(form);
  btn.disabled = true;
  setAlert(alertEl, 'Loading route templates…', 'info');
  try {
    const res = await fetch('/api/weekly/load', { method: 'POST', body: fd, credentials: 'same-origin' });
    const data = await res.json();
    if (data.error !== 0) throw new Error(data.message || 'Load failed');

    weeklyState.routes = data.routes || [];
    weeklyState.assets = data.assets || [];
    weeklyState.days = data.days || [];

    const daySelect = document.getElementById('weekly-day-select');
    const assetSelect = document.getElementById('weekly-asset-select');
    const panel = document.getElementById('weekly-dispatch-panel');

    daySelect.innerHTML = weeklyState.days.map((d) => `<option value="${d}">${d}</option>`).join('');
    assetSelect.innerHTML = weeklyState.assets.map((a) =>
      `<option value="${a.itemid}" data-name="${a.asset_name}">${a.asset_name}</option>`
    ).join('');

    panel.classList.remove('hidden');
    renderWeeklyRoutes(weeklyState.days[0]);
    setAlert(alertEl, data.message, 'ok');
    showToast('Routes loaded', true);
  } catch (err) {
    const msg = err.message === 'Failed to fetch'
      ? 'Network error — loading routes may have timed out. Please try again.'
      : err.message;
    setAlert(alertEl, msg, 'err');
    showToast(msg, false);
  } finally {
    btn.disabled = false;
  }
}

async function dispatchWeeklyRoute() {
  const alertEl = document.getElementById('weekly-dispatch-alert');
  const day = document.getElementById('weekly-day-select').value;
  const routeName = document.getElementById('weekly-route-select').value;
  const assetSelect = document.getElementById('weekly-asset-select');
  const assetItemId = assetSelect.value;
  const assetName = assetSelect.selectedOptions[0]?.dataset.name || '';

  setAlert(alertEl, 'Dispatching route…', 'info');
  try {
    const res = await fetch('/api/weekly/dispatch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ day, route_name: routeName, asset_item_id: assetItemId, asset_name: assetName }),
    });
    const data = await res.json();
    if (data.error === 0) {
      let html = `<strong>Success.</strong> ${data.message || 'Route dispatched.'}`;
      if (data.planning_url) {
        html += `<br><a href="${data.planning_url}" target="_blank" rel="noopener">Open Wialon Logistics</a>`;
      }
      setAlert(alertEl, html, 'ok');
      showToast('Route dispatched', true);
    } else {
      setAlert(alertEl, data.message || 'Dispatch failed', 'err');
      showToast(data.message || 'Dispatch failed', false);
    }
  } catch (err) {
    setAlert(alertEl, err.message, 'err');
    showToast(err.message, false);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  tickClock();
  setInterval(tickClock, 1000);

  const sidebar = document.querySelector('.sidebar');
  if (sidebar) {
    sidebar.addEventListener('mouseleave', () => sidebar.classList.remove('force-collapse'));
    sidebar.querySelectorAll('.nav-item').forEach((item) => {
      item.addEventListener('click', () => sidebar.classList.add('force-collapse'));
    });
  }

  const active = document.body.dataset.activeTab;
  if (active) showTab(active);

  document.getElementById('optimized-form')?.addEventListener('submit', (e) => {
    e.preventDefault();
    submitDispatchForm(e.target, '/api/optimized/dispatch');
  });

  document.getElementById('strict-form')?.addEventListener('submit', (e) => {
    e.preventDefault();
    submitDispatchForm(e.target, '/api/strict/dispatch');
  });

  document.getElementById('weekly-load-form')?.addEventListener('submit', (e) => {
    e.preventDefault();
    loadWeeklyRoutes(e.target);
  });

  document.getElementById('weekly-day-select')?.addEventListener('change', (e) => {
    const routeSelect = document.getElementById('weekly-route-select');
    renderWeeklyRoutes(e.target.value);
    updateWeeklyStops(e.target.value, routeSelect.value);
  });

  document.getElementById('weekly-route-select')?.addEventListener('change', (e) => {
    const day = document.getElementById('weekly-day-select').value;
    updateWeeklyStops(day, e.target.value);
  });

  document.getElementById('weekly-dispatch-btn')?.addEventListener('click', dispatchWeeklyRoute);
});
