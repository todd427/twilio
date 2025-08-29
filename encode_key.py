#!/usr/bin/env python3
import base64
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <api_key>")
    sys.exit(1)

api_key = sys.argv[1]

# encode
encoded = base64.b64encode(api_key.encode("utf-8")).decode("utf-8")

print("Base64 encoded key:")
print(encoded)

# optional sanity check
decoded = base64.b64decode(encoded).decode("utf-8")
print("\nDecoded back (sanity check):")
print(decoded)
