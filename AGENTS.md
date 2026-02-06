# Code Style

1. **No bloat.** Every line, function, and class must earn its place. Minimize code without sacrificing clarity.
2. **Long functions are fine** if they're cohesive—don't split logic artificially.
3. **Self-documenting code over comments.** Use clear names and structure. Remove scaffolding comments after writing.
4. **Match the file's style.** Don't introduce patterns inconsistent with surrounding code.
5. **Avoid sloppy code:**
   - No unnecessary defensive checks or try/catch in trusted codepaths
   - No `any` casts to bypass type issues
   - Inline variables used only once immediately after declaration
