# ReportDisc
A methodical API which report statistics through the discord server.

## Usage:
Copy file `ReportDisc.py` to the appropriate code prompt.

### Import Function:

```py
from ReportDisc import ReportDisc, TorchReportDisc, TFReportDisc
```

### Loop Reporter:

```py
rd = ReportDisc(webhook_url)
rd.report("Iterating through: "+str(array))
for index, dictionary in enumerate(array):
  rd.report_stats(dictionary)
```

### PyTorch Reporter:


```py
trd = TorchReportDisc(webhook_url)
trd.report("Starting PyTorch...")
for epoch in range(10):
  for batch_x, batch_y in dataloader:
    ...
    trd.report_stats({"loss": loss, "acc": acc})
```

### TensorFlow Reporter:

```py
reporter_callback = TFReportDisc(webhook_url)
reporter_callback.reporter.report("Starting TensorFlow...")
model.compile(...)
model.fit(x, y, batch_size=64, epochs=5, validation_split=0.2, callbacks=[reporter_callback])
```

#### OUTPUT:



## Embedding Reports:

```py
reporter_callback = TFReportDisc(webhook_url, embed_reports=True)
```

#### OUTPUT:

