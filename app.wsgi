import os
import sys

# Add the path to your project
sys.path.insert(0, os.path.dirname(__file__))

from app import app as application  # Change 'app' to the name of your main Python file without the .py extension

if __name__ == "__main__":
    application.run()
