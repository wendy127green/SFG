from . import layer
from . import MaskFlownet
from . import pipeline

def get_pipeline(network, **kwargs):
	if network == 'MaskFlownet':
		return pipeline.PipelineFlownet(**kwargs)
	else:
		raise NotImplementedError

def get_pipeline_SFG(network, **kwargs):
	if network == 'MaskFlownet':
		return pipeline_SFG.PipelineFlownet(**kwargs)
	else:
		raise NotImplementedError
