'''
## meta information for the project would be stored here
## author  : anmol jindal
## github  : https://github.com/anmoljindal/fathom
## created : 23rd February 2021
'''
import os

import yaml

## Paths
working_dir = os.path.dirname(os.path.realpath(__file__))
projects_path = os.path.join(working_dir, 'projects')
metadata_path = os.path.join(working_dir, "metadata")
logs_path = os.path.join(working_dir, 'logs')

if not os.path.exists(projects_path):
    os.mkdir(projects_path)

if not os.path.exists(metadata_path):
    os.mkdir(metadata_path)

if not os.path.exists(logs_path):
    os.mkdir(logs_path)

## Log Configuration
app_log_file = os.path.join(logs_path, 'app.log')

## Proxy Information
proxy_file = r'proxy_config.yaml'
proxy_config = None
if not os.path.exists(proxy_file):
    open(proxy_file, 'a').close()

def load_proxy_config():
    global proxy_file, proxy_config
    with open(proxy_file, 'r') as file:
        proxy_config = yaml.load(file, Loader=yaml.FullLoader)

def update_proxy_config(proxy, proxy_type):
    global proxy_file, proxy_config
    if proxy_config is None:
        proxy_config = {}
    
    proxy_config['proxy'] = proxy
    proxy_type['proxy_type'] = proxy_type

    with open(proxy_file, 'w') as file:
        yaml.dump(proxy_config, file)

load_proxy_config()
download_timeout = 10

## Available Options
augmentation_options = [
    "random flip",
    "random rotation"
]

model_options = [
    "mobilenet_v2"
]

def scan_projects():
    projects = {}
    for project in os.listdir(projects_path):
        project_dir = os.path.join(projects_path, project)
        projects[project] = {}
        dataset_file = os.path.join(project_dir, 'dataset.csv')
        if os.path.exists(dataset_file):
            projects[project]['image_details'] = dataset_file
    
    return projects