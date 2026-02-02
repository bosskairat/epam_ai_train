// validation.js
// ES6+ modern validation functions
const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const numberRegex = /\d/;
const specialCharRegex = /[!@#$%^&*(),.?":{}|<>]/;
const phoneRegex = /^\+\d{8,15}$/;

export function validateEmail(email) {
  const errors = [];
  if (!email) errors.push('Email is required.');
  else if (!emailRegex.test(email)) errors.push('Invalid email format.');
  return { valid: errors.length === 0, errors };
}

export function validatePassword(password) {
  const errors = [];
  if (!password) errors.push('Password is required.');
  else {
    if (password.length < 8) errors.push('Password must be at least 8 characters long.');
    if (!numberRegex.test(password)) errors.push('Password must include at least one number.');
    if (!specialCharRegex.test(password)) errors.push('Password must include at least one special character.');
  }
  return { valid: errors.length === 0, errors };
}

export function validatePhone(phone) {
  const errors = [];
  if (!phone) errors.push('Phone number is required.');
  else {
    if (!phone.startsWith('+')) errors.push("Phone number must start with '+' followed by country code.");
    if (!phoneRegex.test(phone)) errors.push("Phone number must contain only digits and be 8-15 digits long after '+'.");
  }
  return { valid: errors.length === 0, errors };
}
