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

#### OUTPUT:
![loop-without-embed](/images/loop_without_embed.png)

### PyTorch Reporter:


```py
trd = TorchReportDisc(webhook_url)
trd.report("Starting PyTorch...")
for epoch in range(10):
  for batch_x, batch_y in dataloader:
    ...
    trd.report_stats({"loss": loss, "acc": acc})
```


#### OUTPUT:
![torch-without-embed](/images/torch_without_embed.png)

### TensorFlow Reporter:

```py
reporter_callback = TFReportDisc(webhook_url)
reporter_callback.reporter.report("Starting TensorFlow...")
model.compile(...)
model.fit(x, y, batch_size=64, epochs=5, validation_split=0.2, callbacks=[reporter_callback])
```

#### OUTPUT:
![tensorflow-without-embed](/images/tensorflow_without_embed.png)

## Embedding Reports:

```py
reporter_callback = TFReportDisc(webhook_url, embed_reports=True)
```


#### OUTPUT:
![loop-with-embed](/images/loop_with_embed.png)
![torch-with-embed](/images/torch_with_embed.png)
![tensorflow-with-embed](/images/tensorflow_with_embed.png)
