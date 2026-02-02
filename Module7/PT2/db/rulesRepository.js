// db/rulesRepository.js
import pkg from 'pg';
const { Pool } = pkg;

// Use environment variable DATABASE_URL for secure connection
const pool = new Pool({ connectionString: process.env.DATABASE_URL });

/**
 * Fetch validation rules from database (parameterized, secure)
 */
export async function getValidationRulesFromDB() {
  const query = 'SELECT id, field_name, rule_name, regex_pattern, error_message FROM validation_rules';
  const { rows } = await pool.query(query);
  return rows;
}
