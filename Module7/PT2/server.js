// server.js
import express from 'express';
import bodyParser from 'body-parser';
import { validateEmail, validatePassword, validatePhone } from './validation.js';
import { getValidationRulesFromDB } from './db/rulesRepository.js';

const app = express();
app.use(bodyParser.json());

// Cache for validation rules
let cachedRules = null;
const RULES_CACHE_TTL = 60 * 1000; // 1 minute
let lastCacheTime = 0;

/**
 * Get cached rules, refresh if TTL expired
 */
async function getCachedRules() {
  const now = Date.now();
  if (!cachedRules || now - lastCacheTime > RULES_CACHE_TTL) {
    try {
      cachedRules = await getValidationRulesFromDB();
      lastCacheTime = now;
    } catch (err) {
      console.error('Failed to fetch validation rules', err);
      cachedRules = [];
    }
  }
  return cachedRules;
}

// GET /validation-rules
app.get('/validation-rules', async (req, res) => {
  try {
    const rules = await getCachedRules();
    res.json(rules);
  } catch (err) {
    res.status(500).json({ error: 'Failed to fetch validation rules.' });
  }
});

// POST /validate
app.post('/validate', async (req, res) => {
  try {
    const { email, password, phone } = req.body;

    const fields = { email, password, phone };
    const validators = { email: validateEmail, password: validatePassword, phone: validatePhone };
    const errors = [];

    for (const [field, value] of Object.entries(fields)) {
      const result = validators[field](value);
      result.errors.forEach(msg => errors.push({ field, message: msg }));
    }

    res.json({ valid: errors.length === 0, errors });
  } catch (err) {
    console.error('Validation failed', err);
    res.status(500).json({ error: 'Internal server error.' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Validation API running on port ${PORT}`));
