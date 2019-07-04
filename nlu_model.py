from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter
import os
import yaml
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
yaml.warnings({'YAMLLoadWarning': False})
def train_nlu(data, configs, model_dir):
    training_data = load_data(data)
    trainer = Trainer(config.load(configs))
    trainer.train(training_data)
    model_directory = trainer.persist(model_dir, fixed_model_name = 'gst_bot')
    
def run_nlu():
    interpreter = Interpreter.load('./models/nlu/default/gst_bot')
    print(interpreter.parse("Hello!"))
    
if __name__ == '__main__':
    train_nlu('./data/data.json', 'config_spacy.json', './models/nlu')
    run_nlu()