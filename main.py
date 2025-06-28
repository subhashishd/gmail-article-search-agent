#!/usr/bin/env python3
"""
Gmail Article Search Agent - Main Application Entry Point

A multi-agent AI system for Gmail article discovery and search.
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

if __name__ == "__main__":
    from backend.main import main
    main()
