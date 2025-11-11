#!/usr/bin/env python3
"""
Extract player JSON files from the markdown artifact.
Usage: python extract_players.py players_collection.md
"""

import re
import os
import sys
import json

def extract_players(markdown_text):
    """Extract all JSON blocks and their filenames from markdown."""
    # Pattern to match: ### filename.json followed by ```json ... ```
    pattern = r'###\s+([a-z_]+\.json)\s*\n```json\s*\n(.*?)\n```'
    
    matches = re.findall(pattern, markdown_text, re.DOTALL | re.MULTILINE)
    
    players = []
    for filename, json_content in matches:
        try:
            # Validate JSON
            parsed = json.loads(json_content)
            players.append((filename, json_content))
            print(f"✓ Found valid JSON for {filename}")
        except json.JSONDecodeError as e:
            print(f"✗ Invalid JSON in {filename}: {e}")
    
    return players

def save_players(players, output_dir="players"):
    """Save player JSON files to directory."""
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_count = 0
    for filename, json_content in players:
        filepath = os.path.join(output_dir, filename)
        
        # Check if file exists
        if os.path.exists(filepath):
            response = input(f"⚠️  {filename} already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print(f"Skipped {filename}")
                continue
        
        # Save file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_content)
        
        print(f"💾 Saved {filepath}")
        saved_count += 1
    
    return saved_count

def main():
    # Read markdown from file or stdin
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            markdown_text = f.read()
    else:
        print("Paste the markdown content, then press Ctrl+D (Unix) or Ctrl+Z (Windows):")
        markdown_text = sys.stdin.read()
    
    # Extract players
    print("\n🔍 Extracting player JSONs...\n")
    players = extract_players(markdown_text)
    
    if not players:
        print("❌ No valid player JSONs found!")
        return
    
    print(f"\n📦 Found {len(players)} player files\n")
    
    # Save players
    saved_count = save_players(players)
    
    print(f"\n✅ Successfully saved {saved_count}/{len(players)} player files to players/")
    print(f"\nRestart uvicorn to load the new players!")

if __name__ == "__main__":
    main()
