# Report schema

All structured reports have `schema_version: "1"` and this shape:

```json
{
  "schema_version": "1",
  "tool": {"name": "packproof", "version": "0.1.0"},
  "summary": {
    "records_checked": 1,
    "lines_read": 1,
    "errors": 0,
    "warnings": 0,
    "infos": 0,
    "truncated": false,
    "status": "PASS"
  },
  "config": {"position_mode": "auto"},
  "finding_counts": {},
  "findings": [
    {
      "code": "POS003",
      "severity": "error",
      "message": "position_ids do not follow reset at each segment boundary",
      "line": 4,
      "field": "position_ids",
      "record_id": "pack-7"
    }
  ]
}
```

`record_id`, `field`, and `line` are optional. Finding messages are designed to
be useful without including input payloads. New codes should be additive and
documented in the README/changelog.
