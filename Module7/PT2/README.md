# Validation API

## Overview
The **Validation API** provides endpoints to validate user data including **email**, **password**, and **international phone numbers**. Validation rules are configurable and stored in the database, allowing dynamic updates without changing code.  

**Key Features:**
- Validate email, password, and phone number fields.
- Return structured error messages.
- Retrieve all validation rules.
- Support dynamic rule configuration.

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/bosskairat/epam_ai_train.git
cd Module7/PT2
```

2. **Install dependencies:**
```bash
npm install
```

3. **Set up SQLite database:**
```bash
node scripts/init_db_from_sql.js
```

4. **Start the server:**
```bash
npm start
```

## Usage Examples

```javascript
import { validateAll } from './validation.js';

const result = validateEmail({
  email: 'test@example.com',
  password: 'Passw0rd!',
  phone: '+12345678900'
});
console.log(result);
```

## API Endpoints
1. **GET /validation-rules**
Retrieve all validation rules stored in the database.

Request:
```http
GET /validation-rules
```

Response Example:
```json
[
  {
    "id": 1,
    "field_name": "email",
    "rule_name": "valid_format",
    "regex_pattern": "^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$",
    "error_message": "Invalid email format."
  },
  {
    "id": 2,
    "field_name": "password",
    "rule_name": "min_length",
    "regex_pattern": ".{8,}",
    "error_message": "Password must be at least 8 characters long."
  }
]
```

2. **POST /validate**
Validate incoming user data according to the rules stored in the database.

Request:
```http
POST /validate
Content-Type: application/json

{
  "email": "test@example.com",
  "password": "Passw0rd!",
  "phone": "+12345678901"
}
```

Response Example (Valid Data):
```json
{
  "valid": true,
  "errors": []
}
```
Response Example (Invalid Data):
```json
{
  "valid": false,
  "errors": [
    { "field": "email", "message": "Invalid email format." },
    { "field": "password", "message": "Password must include at least one special character." }
  ]
}
```

## Error Codes

| Code | Meaning                | Description                                                        |
|------|------------------------|--------------------------------------------------------------------|
| 400  | Bad Request            | Invalid request body or missing fields                             |
| 404  | Not Found              | Resource not found (e.g., validation rules not found)             |
| 500  | Internal Server Error  | Unexpected server error during validation or database query       |


## Example Code
See validation.js and validation.test.js for full examples