from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import RasaNLUInterpreter
import os
import yaml
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
yaml.warnings({'YAMLLoadWarning': False})

logger = logging.getLogger(__name__)

def train_dialogue(domain_file = 'gst_domain.yml',
					model_path = './models/dialogue',
					training_data_file = 'data/stories.md'):
					
	agent = Agent(domain_file, policies = [MemoizationPolicy(), KerasPolicy()])
	
	agent.train(
				training_data_file,
				epochs = 300,
				batch_size = 50,
				validation_split = 0.2)
				
	agent.persist(model_path)
	return agent
	
def run_gst_bot(serve_forever=True):
	interpreter = RasaNLUInterpreter('./models/nlu/default/gst_bot')
	agent = Agent.load('./models/dialogue', interpreter = interpreter)
	if serve_forever:
		agent.handle_channel(ConsoleInputChannel())
        
if __name__ == '__main__':
	#train_dialogue()
	run_gst_bot()