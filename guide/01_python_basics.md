# Part 1 — Python basics (what each idea does)

Python is a language where you write **instructions** top to bottom. The computer runs them and can **store data**, **organize data**, and **reuse logic** with functions.

---

## 1. Variables — named boxes in memory

```python
name = "Arash"
age = 25
```

| Piece | What happens |
|--------|----------------|
| `name` | A **label** you choose. It points to a value in memory. |
| `=` | **Assignment**: “store the right side in the left side.” |
| `"Arash"` | A **string** (`str`): text. Quotes tell Python this is text. |
| `25` | An **integer** (`int`): whole number. |

You can **read** a variable later: `print(name)` calls the built-in function `print`, which shows text in the terminal.

```python
print(type(name))   # <class 'str'>  — type() tells you what kind of value it is
```

Other common types:

- `1.75` → **float** (decimal number)
- `True` / `False` → **bool** (yes/no)

**Mental model:** variables are sticky notes on boxes; the box holds one value at a time (you can replace it: `age = 26`).

---

## 2. Data structures — how you group many values

### List — ordered, changeable

```python
fruits = ["apple", "banana"]
fruits.append("orange")
first = fruits[0]      # "apple" — index 0 is the FIRST item
count = len(fruits)    # 3
```

- **`[ ]`** creates a **list**.
- **Order matters**: `fruits[0]` is first, `fruits[1]` is second.
- **`.append(x)`** adds at the end.
- Use a list when you have a **sequence** you will change (shopping cart, list of files).

### Dictionary — lookup by name (key)

```python
user = {"name": "Arash", "email": "a@example.com"}
print(user["name"])    # Arash
user["role"] = "dev"   # add or update one field
```

- **`{ key: value }`** — like a row in a database or JSON object.
- Use a dict when each item has **named fields** (user profile, job status, symbol metadata).

### Tuple — ordered, fixed

```python
bbox = (100, 50, 40, 30)   # x, y, width, height
x, y, w, h = bbox          # unpacking into four variables
```

- Use a tuple when the group **should not change** (coordinates, RGB color).

### Set — unique values only

```python
tags = {"python", "ml", "python"}
# only one "python" inside
```

- Use a set when duplicates must disappear (unique symbol IDs seen on a page).

---

## 3. Functions — reusable steps with a name

```python
def greet(name):
    return "Hello, " + name

message = greet("Arash")
```

| Line | Meaning |
|------|---------|
| `def greet(name):` | **Define** a function named `greet`. `name` is a **parameter** (input placeholder). |
| Indented body | Everything indented belongs to this function. |
| `return ...` | **Send back** a result to whoever called the function. |
| `greet("Arash")` | **Call** the function: `name` becomes `"Arash"` for that call. |

Functions let you:

- Avoid copying the same code
- Test one piece of logic alone
- Build a **pipeline**: `load_image()` → `find_symbols()` → `save_json()`

```python
if __name__ == "__main__":
    print(greet("Arash"))
```

This block runs **only** when you execute this file directly (`python myfile.py`), not when another file imports it. Good place for tests and demos.

---

## 4. How Python files become a “project”

1. **One file** — learn and experiment (`see_ml_flow.py`).
2. **Many files** — `utils.py`, `models.py`, `api.py` import each other.
3. **Service** — .NET or FastAPI receives uploads; **Python worker** runs your functions on a queue.

Your interview architecture is exactly that: **frontend uploads → backend creates job → Python processes image → status updates**.

---

## 5. What you should type yourself (minimum practice)

Create `my_first.py`:

```python
name = "Arash"
skills = ["python", "ml"]

def show_skills(person, items):
    return person + " knows: " + ", ".join(items)

if __name__ == "__main__":
    print(show_skills(name, skills))
```

Run: `python my_first.py`

Understand each line before moving to Part 2 (machine learning).
