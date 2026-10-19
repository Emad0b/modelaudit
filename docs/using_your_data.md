# Reusable API

```python
from modelaudit.core import audit
# See the JSON evidence schema in reports/healthy_input.json.
result = audit(evidence_bundle)
```

Validate units, timestamps, missingness and evaluation assumptions. The simulator is not a substitute for representative labelled data.
