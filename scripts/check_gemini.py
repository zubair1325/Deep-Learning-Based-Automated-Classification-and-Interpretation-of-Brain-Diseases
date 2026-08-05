import os
import sys
import json
from pathlib import Path
# Minimal test script to validate GEMINI_API_KEY and make a lightweight GenAI call.
# It avoids printing the full key.

env_path = Path(__file__).resolve().parents[1] / '.env'
key = None
if env_path.exists():
    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            k, v = line.split('=', 1)
            if k.strip() == 'GEMINI_API_KEY':
                key = v.strip()
                break

if not key:
    print('ERROR: GEMINI_API_KEY not found in .env')
    sys.exit(2)

masked = key[:4] + '...' + key[-4:]
print('Found GEMINI_API_KEY in .env (masked):', masked)

try:
    from google import genai
    from google.genai import types
except Exception as e:
    print('ERROR: google.genai not installed or import failed:', repr(e))
    sys.exit(3)

client = genai.Client(api_key=key)

# Prepare a tiny test prompt
contents = [
    types.Content(parts=[
        types.Part.from_text(text='Return a single short JSON object with key "ok": true'),
    ])
]

for model_name in ('gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-1.5-mini'):
    try:
        print('\nTesting model:', model_name)
        resp = client.models.generate_content(model=model_name, contents=contents, temperature=0, max_output_tokens=128)
        print('Call succeeded for', model_name)
        # Print candidate text(s) safely
        if getattr(resp, 'candidates', None):
            for i, c in enumerate(resp.candidates):
                cont = getattr(c, 'content', None)
                text = getattr(cont, 'text', None) if cont is not None else None
                if text:
                    print(f'candidate[{i}].content.text:', text[:2000])
                elif getattr(cont, 'parts', None):
                    parts_texts = [getattr(p, 'text', '') for p in cont.parts if getattr(p, 'text', None)]
                    print(f'candidate[{i}].parts.text:', ' '.join(parts_texts)[:2000])
        else:
            print('No candidates in response (unexpected).')
        sys.exit(0)
    except Exception as e:
        print('Model', model_name, 'failed:', repr(e))
        continue

print('All tested models failed. See errors above.')
sys.exit(4)
