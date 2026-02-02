// db/rulesRepository.js
import sqlite3 from 'sqlite3';
import { open } from 'sqlite';

// Open SQLite database
export async function openDB() {
  return open({
    filename: './validation.db',
    driver: sqlite3.Database
  });
}

/**
 * Fetch validation rules from SQLite
 */
export async function getValidationRulesFromDB() {
  const db = await openDB();
  try {
    const rows = await db.all('SELECT id, field_name, rule_name, regex_pattern, error_message FROM validation_rules');
    return rows;
  } finally {
    await db.close();
  }
}
