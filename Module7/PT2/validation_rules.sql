CREATE TABLE validation_rules (
    id SERIAL PRIMARY KEY,
    field_name VARCHAR(50) NOT NULL,
    rule_name VARCHAR(50) NOT NULL,
    regex_pattern TEXT,
    error_message VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO validation_rules (field_name, rule_name, regex_pattern, error_message)
VALUES
('email', 'valid_format', '^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$', 'Invalid email format.'),
('password', 'min_length', '.{8,}', 'Password must be at least 8 characters long.'),
('password', 'has_number', '.*\\d.*', 'Password must include at least one number.'),
('password', 'has_special', '.*[!@#$%^&*].*', 'Password must include at least one special character.'),
('phone', 'intl_format', '^\\+\\d{8,15}$', 'Phone number must be in international format.');
