---
name: venv311 dependency issues
description: Known missing/broken packages in the .venv311 environment
type: project
---

`google-generativeai` was missing from requirements.txt and not installed in `.venv311`. Added `google-generativeai>=0.8.0` to requirements.txt and installed v0.8.6.

The vector store is Pinecone-only. Keep `PINECONE_API_KEY` configured for indexing/search paths.

**Why:** google-generativeai was never added to requirements.txt, so fresh venv installs miss it.

**How to apply:** If the user reports terminal noise on startup or Gemini not working, these are the two known culprits.
