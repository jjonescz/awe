import awe.data.set.swde
import awe.training.params
import awe.training.trainer
import awe.data.sampling

params = awe.training.params.Params.load_user(normalize=True)
trainer = awe.training.trainer.Trainer(params)
trainer.ds = awe.data.set.swde.Dataset(suffix='-exact', convert=False, only_verticals=('auto',))
trainer.prepare_features()
