const form = document.querySelector('#fare-form');
const result = document.querySelector('#result-content');
const submitButton = form.querySelector('button[type="submit"]');

function pad(number) {
  return String(number).padStart(2, '0');
}

const now = new Date();
document.querySelector('#pickup-date').value = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
document.querySelector('#pickup-time').value = `${pad(now.getHours())}:${pad(now.getMinutes())}`;

async function loadModelMetrics() {
  try {
    const response = await fetch('/api/metrics');
    const metrics = await response.json();
    if (!response.ok) throw new Error(metrics.error || 'Model information is unavailable.');
    document.querySelector('#mae-value').textContent = `$${Number(metrics.mae).toFixed(2)}`;
    document.querySelector('#r2-value').textContent = Number(metrics.r2).toFixed(3);
    document.querySelector('#model-period').textContent =
      `${Number(metrics.test_rows).toLocaleString()} later trips in the chronological test set.`;
  } catch {
    document.querySelector('#model-period').textContent = 'Model performance details are unavailable.';
  }
}

function showError(message) {
  result.innerHTML = `<div class="result-icon" aria-hidden="true">!</div><h2>Could not estimate</h2><p class="error-message"></p>`;
  result.querySelector('.error-message').textContent = message;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;
  const values = new FormData(form);
  const payload = {
    pickup_latitude: values.get('pickup_latitude'),
    pickup_longitude: values.get('pickup_longitude'),
    dropoff_latitude: values.get('dropoff_latitude'),
    dropoff_longitude: values.get('dropoff_longitude'),
    pickup_datetime: `${values.get('pickup_date')}T${values.get('pickup_time')}`,
    passenger_count: values.get('passenger_count'),
  };

  submitButton.disabled = true;
  submitButton.querySelector('span:first-child').textContent = 'Calculating…';
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The prediction could not be completed.');
    result.innerHTML = `<div class="result-icon" aria-hidden="true">$</div><p class="eyebrow">ESTIMATED FARE</p><div class="fare-value"></div><p class="fare-subtitle">Model estimate for your trip</p><span class="distance-chip"></span>`;
    result.querySelector('.fare-value').textContent = `$${Number(data.estimated_fare).toFixed(2)}`;
    result.querySelector('.distance-chip').textContent = `${Number(data.distance_km).toFixed(1)} km straight-line distance`;
  } catch (error) {
    showError(error.message);
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector('span:first-child').textContent = 'Estimate fare';
  }
});

loadModelMetrics();
