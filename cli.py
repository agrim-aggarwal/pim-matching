import click
import time
from utils.Initialize import Initialize
from constants.FileNameConstants import llm_configs_directory
Initialize().download_config()
Initialize().download_s3_folder('ciq-marketshare', llm_configs_directory)

from entrypoints.DataProcessing import DataProcessing
from entrypoints.SKUMatching import SKUMatching

from utils.log import Log    
log = Log(__name__)

@click.group()
@click.version_option('1.0.0', prog_name='pimmatching')
def main():
	"""pimmatching[pm] is used to run entrypoint files for pim matching pipeline"""
	pass

@main.command()
@click.option('--entrypoint', type=click.Choice(['DataProcessing', 
                                                 'SKUMatching', 
                                                ], case_sensitive=False), help='Entrypoint file to run')
@click.option('--client-name', '-c', help='The client name', type=str)
@click.option('--run-date', '-rd', help='Run date for the client', type=str)
@click.option('--override', '-ov', help='Override of base files, default option is False, to override use the agrument "y" or "yes" or "true" ', type=str)
@click.option('--max_n', '-mn', help='Maximum number of neighbors to search in for low confidence matches, The default value is 50 ', type=str)




def run_entrypoint(entrypoint: str, client_name: str, run_date: str, override: str, max_n:int):
    
	""" run entrypoint """
	if entrypoint == 'DataProcessing':
		dp = DataProcessing(client_name, run_date, override)
		dp.run() 
	elif entrypoint == 'SKUMatching':
		sm = SKUMatching(client_name, run_date, max_n)
		sm.run()
        
if __name__ == '__main__':
	main()
