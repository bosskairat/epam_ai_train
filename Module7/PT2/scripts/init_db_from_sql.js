import fs from 'fs/promises';
import sqlite3 from 'sqlite3';
import { open } from 'sqlite';

async function initFromSql() {
  const sqlPath = './validation_rules.sql';
  const sql = await fs.readFile(sqlPath, { encoding: 'utf8' });

  const db = await open({ filename: './validation.db', driver: sqlite3.Database });
  try {
    await db.exec(sql);
    console.log('Executed SQL from', sqlPath);
  } finally {
    await db.close();
  }
}

initFromSql().catch(err => {
  console.error('Failed to initialize DB from SQL', err);
  process.exit(1);
});
