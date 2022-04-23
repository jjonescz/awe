# Release process

1. Train a model (see [Training](train.md)).

2. Add `description` and `examples` to `info.json` in the model's version dir.

3. Create release on GitHub (at the code the model was trained with).

4. Pack and upload the checkpoint.

   ```bash
   tar czf logs.tar.gz logs/1-version-name/
   gh auth login
   gh release upload v0.1 logs.tar.gz
   ```