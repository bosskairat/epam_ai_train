// validation.test.js
import { validateEmail, validatePassword, validatePhone } from './validation.js';

describe('Validation Module', () => {

  // --------------------------
  // Email Validation Tests
  // --------------------------
  test('Valid email passes', () => {
    const result = validateEmail('test@example.com');
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  test('Invalid email fails', () => {
    const result = validateEmail('invalid-email');
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Invalid email format.');
  });

  test('Empty email fails', () => {
    const result = validateEmail('');
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Email is required.');
  });

  test('Null email fails', () => {
    const result = validateEmail(null);
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Email is required.');
  });

  // --------------------------
  // Password Validation Tests
  // --------------------------
  test('Valid password passes', () => {
    const result = validatePassword('Passw0rd!');
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  test('Password missing number fails', () => {
    const result = validatePassword('Password!');
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Password must include at least one number.');
  });

  test('Password too short fails', () => {
    const result = validatePassword('P1!');
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Password must be at least 8 characters long.');
  });

  test('Password missing special character fails', () => {
    const result = validatePassword('Password1');
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Password must include at least one special character.');
  });

  // --------------------------
  // Phone Validation Tests
  // --------------------------
  test('Valid international phone passes', () => {
    const result = validatePhone('+12345678901');
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  test('Phone missing + fails', () => {
    const result = validatePhone('1234567890');
    expect(result.valid).toBe(false);
    expect(result.errors).toContain("Phone number must start with '+' followed by country code.");
  });

  test('Phone with letters fails', () => {
    const result = validatePhone('+123ABC7890');
    expect(result.valid).toBe(false);
    expect(result.errors).toContain("Phone number must contain only digits and be 8-15 digits long after '+'.");
  });

  test('Empty phone fails', () => {
    const result = validatePhone('');
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Phone number is required.');
  });

  test('Null phone fails', () => {
    const result = validatePhone(null);
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Phone number is required.');
  });

});
