---
jupyter:
  jupytext:
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
---

```python
# Tests that CI sets the test mode environment variable
import os
test_mode = os.getenv('AI_STACK_TEST_MODE', 'false')
assert test_mode == 'true', test_mode
```
