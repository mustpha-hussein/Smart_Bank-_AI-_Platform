const $$ = (selector) => document.querySelectorAll(selector);
const $ = (selector) => document.querySelector(selector);
let toastTimer;

function showToast(message) {
  const toast = $('.toast');
  toast.textContent = message;
  toast.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('visible'), 2600);
}

function showPage(pageId) {
  $$('.page').forEach((page) => page.classList.toggle('active', page.id === pageId));
  $$('.nav-link').forEach((link) => link.classList.toggle('active', link.dataset.page === pageId));
  $('.sidebar').classList.remove('open');
  history.replaceState(null, '', `#${pageId}`);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

$$('.nav-link').forEach((button) => button.addEventListener('click', () => showPage(button.dataset.page)));
$$('[data-goto]').forEach((button) => button.addEventListener('click', () => showPage(button.dataset.goto)));
$('.menu-button').addEventListener('click', () => $('.sidebar').classList.toggle('open'));

$('#loan-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  const result = $('#loan-result');
  const payload = {person_age:+data.get('age'), person_gender:data.get('gender'), person_education:data.get('education'), person_income:+data.get('income'), person_emp_exp:+data.get('employment'), person_home_ownership:data.get('home'), loan_amnt:+data.get('amount'), loan_intent:data.get('intent'), loan_int_rate:+data.get('rate'), loan_percent_income:+data.get('percent'), cb_person_cred_hist_length:+data.get('history'), credit_score:+data.get('score'), previous_loan_defaults_on_file:data.get('default')};
  requestPrediction('/api/loan/predict', payload, result, (response) => ({positive: response.approved, title: response.decision, description: response.approved ? 'The model indicates this application is eligible for approval.' : `The model recommends a manual lending review. Approval requires at least ${Math.round(response.decision_threshold * 100)}%.`, score: response.approval_probability, scoreLabel: 'Approval probability'}));
});

$('#fraud-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  const result = $('#fraud-result');
  const payload = {amt:+data.get('amount'), category:data.get('category'), merchant:data.get('merchant'), gender:data.get('gender'), job:data.get('job'), state:data.get('state'), date:data.get('date'), time:data.get('time'), date_of_birth:data.get('birth')};
  requestPrediction('/api/fraud/predict', payload, result, (response) => ({positive: !response.flagged, title: response.decision, description: response.flagged ? 'The model flagged this transaction for manual fraud review.' : 'The model found no high-risk fraud signal.', score: response.fraud_probability, scoreLabel: 'Fraud probability'}));
});

async function requestPrediction(endpoint, payload, result, format) {
  result.className = 'result-card';
  result.innerHTML = '<span class="result-icon">⌛</span><p class="eyebrow">AI inference result</p><h2>Analyzing…</h2><p>Running the saved model.</p>';
  try {
    const response = await fetch(endpoint, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'Prediction failed.');
    const data = format(body);
    result.className = `result-card ${data.positive ? 'approved' : 'alert'}`;
    result.innerHTML = `<span class="result-icon">${data.positive ? '✓' : '!'}</span><p class="eyebrow">AI inference result</p><h2 class="decision">${data.title}</h2><p>${data.description}</p><p class="confidence">${data.scoreLabel || 'Decision confidence'}: <b>${Math.round(data.score * 100)}%</b></p>`;
  } catch (error) {
    result.className = 'result-card declined';
    result.innerHTML = `<span class="result-icon">!</span><p class="eyebrow">Connection error</p><h2>Prediction unavailable</h2><p>${error.message} Start the SmartBank API and try again.</p>`;
  }
}

$$('.theme-choice').forEach((button) => button.addEventListener('click', () => {
  $$('.theme-choice').forEach((choice) => choice.classList.remove('selected'));
  button.classList.add('selected');
  const theme = button.dataset.theme;
  const dark = theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  document.body.classList.toggle('dark', dark);
  localStorage.setItem('smartbank-theme', theme);
}));

$('#save-settings').addEventListener('click', () => showToast('Settings saved successfully.'));
$('#global-search').addEventListener('keydown', (event) => {
  if (event.key !== 'Enter') return;
  const query = event.currentTarget.value.toLowerCase();
  const match = ['dashboard', 'loan', 'fraud', 'settings'].find((page) => page.includes(query));
  if (match) showPage(match); else showToast('No matching workspace found.');
});

const savedTheme = localStorage.getItem('smartbank-theme');
if (savedTheme) { const choice = $(`.theme-choice[data-theme="${savedTheme}"]`); if (choice) choice.click(); }
const initialPage = location.hash.slice(1);
if (['dashboard', 'loan', 'fraud', 'settings'].includes(initialPage)) showPage(initialPage);
