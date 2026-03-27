# Copilot Instructions

## Project Overview
This is an AnisoraV3 video generation project using Wan2.1 models for image-to-video and text-to-video generation.

## Custom Instructions

### get_error Tool Behavior
When `get_error` tool is called, it should return the original source code along with any error messages, not just the error information. This allows Copilot to better understand and fix issue in context.

### Development Guidelines
- Python 3.10+ project
- Uses conda for environment management
- Main components: fastvideo/, wan/, api/
- Inference scripts in root directory
