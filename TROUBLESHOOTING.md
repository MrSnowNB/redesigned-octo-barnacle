***
error_id: <unique-id>
date: <yyyy-mm-dd>
component: <module/script>
context: <short description>
symptom: <actual error/bug>
error_snippet: <stack/output>
probable_cause: <one-line>
quick_fix: <if any>
permanent_fix: <if any>
prevention: <how to stop recurrence>
***

# TROUBLESHOOTING LOG

## Seeded Issues

### Dependency Resolution Loop
- **Context:** pip install requirements.txt
- **Symptom:** Package version conflicts during installation
- **Error Snippet:** ERROR: Cannot install - conflicting versions
- **Probable Cause:** Indirect dependency version conflicts
- **Quick Fix:** Use constraints.txt with pinned versions
- **Permanent Fix:** Update requirements.txt with compatible versions
- **Prevention:** Test installations in clean virtual environments

### Embedding Model OOM (Out of Memory)
- **Context:** Viseme extraction on large videos
- **Symptom:** CUDA out of memory or system freeze
- **Error Snippet:** RuntimeError: CUDA out of memory
- **Probable Cause:** Batch size too large for available VRAM
- **Quick Fix:** Reduce batch size, use CPU fallback
- **Permanent Fix:** Implement dynamic batch sizing
- **Prevention:** Add memory usage checks before processing

### Socket Port Conflicts
- **Context:** Local orchestrator startup
- **Symptom:** Port already in use errors
- **Error Snippet:** OSError: [Errno 48] Address already in use
- **Probable Cause:** Previous process didn't clean up
- **Quick Fix:** Kill existing processes on port
- **Permanent Fix:** Randomized port binding with retries
- **Prevention:** Proper cleanup in signal handlers
