import ast
src = open("server/app.py").read()

# Syntax check
ast.parse(src)
print("syntax: valid")

checks = [
    ("import uvicorn", "import uvicorn" in src),
    ("from api.app import app", "from api.app import app" in src),
    ("def main():", "def main()" in src),
    ("uvicorn.run present", "uvicorn.run" in src),
    ('uses "api.app:app"', '"api.app:app"' in src),
    ("if __name__ guard", 'if __name__ == "__main__"' in src),
]
for desc, ok in checks:
    print(f'  {"OK" if ok else "MISSING"}  {desc}')

from server.app import main, app
print("main callable:", callable(main))
print("app type:", type(app).__name__)
