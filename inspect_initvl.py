# Read and analyze transformation_core.py
with open('transformation_core.py', 'r') as f:
    content = f.read()

# Show first 150 lines to understand structure
lines = content.split('\n')
print("="*70)
print("TRANSFORMATION_CORE.PY STRUCTURE")
print("="*70)
for i, line in enumerate(lines[:150], 1):
    print(f"{i:4d}: {line}")

print("\n" + "="*70)
print("SEARCHING FOR KEY TRANSFORMATION FUNCTIONS")
print("="*70)

# Find key functions
import re
functions = re.findall(r'^def (\w+)\(.*?\):', content, re.MULTILINE)
print(f"\nFunctions found: {len(functions)}")
for func in functions[:20]:  # Show first 20
    print(f"  - {func}")