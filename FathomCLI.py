import os
import json
import shutil
import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

import Meta
import Trainer
import GoogleImagesDownloader

def read_json_file(filepath):
    with open(filepath, 'r') as file:
        json_data = json.load(file)
    return json_data

def write_json_file(json_data, filepath):
    with open(filepath, 'w') as file:
        json.dump(json_data, file)

def get_project_details(project_name):

    project_path = os.path.join(Meta.projects_path, project_name)
    if not os.path.exists(project_path):
        raise Exception("project does not exists")
    
    json_filename = os.path.join(project_path, '{}.json'.format(project_name))
    project_json = read_json_file(json_filename)
    return project_json

def update_project_json(project_name, project_json):

    project_path = os.path.join(Meta.projects_path, project_name)
    if not os.path.exists(project_path):
        raise Exception("project does not exists")
    
    json_filename = os.path.join(project_path, '{}.json'.format(project_name))
    write_json_file(project_json, json_filename)

def create_project(project_name: str, 
        groups: dict, splits: list, augmentations: list, batch_size: int, image_size: int,
        model_name: str, base_learning_rate: float, epochs: int):
    project_json = {}
    project_json['project'] = project_name

    project_path = os.path.join(Meta.projects_path, project_name)
    project_json['working_dir'] = project_path
    if not os.path.exists(project_path):
        os.mkdir(project_path)
    
    project_json['groups'] = deepcopy(groups)
    
    if sum(splits) > 100:
        raise Exception('value error')
    data_splits = [
        ["train", int(splits[0])],
        ["validation", int(splits[1])],
        ["test", int(splits[2])]
    ]
    project_json['data_splits'] = data_splits
    project_json['augmentations'] = list(map(lambda x: x.replace(' ','_'), augmentations))
    project_json['batch_size'] = batch_size
    project_json['model'] = model_name
    project_json['image_size'] = image_size
    project_json['base_learning_rate'] = base_learning_rate
    project_json['epochs'] = epochs

    update_project_json(project_name, project_json)
    return project_json

def validate_project_json(project_json):
    if not isinstance(project_json['project']) or len(project_json.strip())==0:
        error_message = "invalid project name"
        return error_message
    return

def get_project_dataset(project_json):

    project_path = project_json['working_dir']
    if not os.path.exists(project_path):
        os.mkdir(project_path)
    
    images_path = os.path.join(project_path, 'images')
    if not os.path.exists(images_path):
        os.mkdir(images_path)

    dataset_file = os.path.join(project_path, 'dataset.csv')
    if os.path.exists(dataset_file):
        dataset = pd.read_csv(dataset_file)
    else:
        dataset = None
    
    details = {'images_path':images_path,"dataset":dataset, "dataset_path":dataset_file}
    return details

def get_split_path(existing_path, new_path):
    filename = os.path.basename(existing_path)
    new_path = new_path/filename
    return new_path

def move_file(source, destination):
    if source.exists():
        source.replace(destination)

def train_test_split(images_df, splits:list, working_dir):
    total = 0
    for split in splits:
        total += split[1]
    
    if total != 100:
        raise Exception('invalid split - does not equal 100')

    splits_df = {split[0]:[] for split in splits}
    for category in images_df['category'].unique().tolist():
        cat_images_df = images_df[images_df['category']==category]
        cat_images_df = cat_images_df.sample(frac=1.0)
        total = len(cat_images_df)
        num_splits = len(splits) - 1
        for i, [split_name, split] in enumerate(splits):
            limit = (split*total)//100
            if i == num_splits:
                subset = cat_images_df.copy()
            else:
                subset = cat_images_df.sample(limit)
            
            cat_images_df = cat_images_df[~cat_images_df['path'].isin(subset['path'].tolist())]
            splits_df[split_name].append(subset)
    
    splits_df = {split:pd.concat(dfs, sort=False).assign(split=split) for split, dfs in splits_df.items()}
    splits_df = pd.concat(splits_df.values())
    
    ##move to folders
    frame_list = []
    working_dir = Path(working_dir)
    splits_df = splits_df[pd.notnull(splits_df['path'])]
    splits_df['path'] = splits_df['path'].apply(Path)
    for split in splits:
        split_folder = working_dir/split[0]
        if not os.path.exists(split_folder):
            os.mkdir(split_folder)
        
        subset = splits_df[splits_df['split']==split[0]]
        for category in subset.category.unique().tolist():
            category_split_folder = split_folder/category
            if not os.path.exists(category_split_folder):
                os.mkdir(category_split_folder)
            
            category_subset = subset[subset['category']==category]
            category_subset['new_path'] = category_subset['path'].apply(lambda x: get_split_path(x, category_split_folder))
            category_subset.apply(lambda row: move_file(row['path'], row['new_path']), axis=1)
            frame_list.append(category_subset)
    
    splits_df = pd.concat(frame_list, sort=False)
    splits_df['path'] = splits_df['new_path'].apply(str)
    splits_df.drop(['new_path'], axis=1, inplace=True)

    #folder cleanup
    for x in os.listdir(working_dir):
        if os.path.isfile(working_dir/x):
            os.unlink(working_dir/x)
            continue
        
        if x not in [split[0] for split in splits]:
            shutil.rmtree(working_dir/x)
    
    return splits_df

def add_to_dataset(project_json, category, imagefiles):
    details = get_project_dataset(project_json)
    dataset = []
    
    for imagefile in imagefiles:
        image_filename = os.path.join(details['images_path'], imagefile.name)
        image = Image.oepn(imagefile)
        image.save(image_filename)
        dataset.append({"category":category, "path":image_filename})
    
    dataset = pd.DataFrame(dataset)
    if details['dataset'] is not None:
        dataset = pd.concat([details['dataset'], dataset], sort=False)
    dataset = train_test_split(dataset, project_json['data_splits'], details['images_path'])
    dataset.to_csv(details['dataset_path'], index=False)
    return dataset

def remove_from_dataset(project_json, imagepaths):
    details = get_project_dataset(project_json)
    dataset = details['dataset']
    dataset = dataset[~dataset['path'].isin(imagepaths)]
    dataset = train_test_split(dataset, project_json['data_splits'], details['images_path'])
    dataset.to_csv(details['dataset_path'], index=False)
    return dataset

def download_dataset(project_json):
    details = get_project_dataset(project_json)

    proxy, proxy_type = None, None
    if Meta.proxy_config is not None:
        proxy = Meta.proxy_config['proxy']
        proxy_type = Meta.proxy_config['proxy_type']

    dataset = GoogleImagesDownloader.download_images(details['images_path'], groups=project_json['groups'],
        proxy=proxy, proxy_type=proxy_type, download_timeout=Meta.download_timeout)
    dataset = GoogleImagesDownloader.validate_images(dataset)
    if details['dataset'] is not None:
        dataset = pd.concat([dataset, details['dataset']], sort=False)
    dataset = train_test_split(dataset, project_json['data_splits'], details['images_path'])
    dataset.to_csv(details['dataset_path'], index=False)
    return dataset

def train_model(project_json, custom_callbacks=[]):

    project_path = project_json['working_dir']
    images_path = os.path.join(project_path, 'images')
    models_path = os.path.join(project_path, "models")
    reports_path = os.path.join(project_path, "reports")
    if "current_model_version" not in project_json:
        project_json['current_model_version'] = 0
    if not os.path.exists(models_path):
        os.mkdir(models_path)
    if not os.path.exists(reports_path):
        os.mkdir(reports_path)

    datasets = Trainer.get_datasets(
        working_dir=images_path,
        batch_size=project_json['batch_size'],
        image_size=project_json['image_size']
    )

    model = Trainer.get_model(
        model_name=project_json['model'], 
        n_classes=len(project_json['groups']), 
        image_size=project_json['image_size'], 
        augmentations=project_json['augmentations'], 
        base_learning_rate=project_json['base_learning_rate']
    )
    
    project_json['current_model_version'] += 1
    model_filename = os.path.join(models_path, str(project_json['current_model_version']))
    tblog_filename = "{}.logs".format(project_json['current_model_version'])
    tblog_filename = os.path.join(reports_path, tblog_filename)

    model, history = Trainer.train_model(
        model,
        datasets['train'], 
        epochs=project_json['epochs'],
        validation_dataset=datasets['validation'],
        logs=tblog_filename,
        custom_callbacks=custom_callbacks
    )

    
    model.save(model_filename)

    training_report = Trainer.get_training_report(history)
    training_report_filename = "{}.report.csv".format(project_json['current_model_version'])
    training_report_filename = os.path.join(reports_path, training_report_filename)
    training_report.to_csv(training_report_filename, index=False)
    update_project_json(project_json['project'], project_json)

    return history, model, project_json['current_model_version']

def main(args):

    if args.operation in ["create","update","all"]:
        project_json = create_project(
            args.project, args.groups,
            splits=args.splits, augmentations=args.augmentations,
            batch_size=args.batch_size, image_size=args.image_size,
            model_name=args.model, base_learning_rate=args.learning_rate,
            epochs=args.epochs
        )
        print("project created")

    if args.operation in ["download",'all']:
        project_json = get_project_details(args.project)
        dataset = download_dataset(project_json)
        print('dataset of {} images created'.format(len(dataset)))
    
    if args.operation in ["train","all"]:
        project_json = get_project_details(args.project)
        _, _, model_version = train_model(project_json)
        print("model version {} trained".format(model_version))
        
    if args.operation in ["details","all"]:
        project_json = get_project_details(args.project)
        print(project_json)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", help="operation to be performed", type=str)
    parser.add_argument("project", help="project name", type=str)
    parser.add_argument("--groups", help="categories and there corresponding keywords")
    parser.add_argument("--splits", help="dataset splits", type=str)
    parser.add_argument("--augmentations", help="data augmentations", type=str)
    parser.add_argument("--batch_size", help="batch size", type=int)
    parser.add_argument("--image_size", help="image size in pixels", type=int)
    parser.add_argument("--model", help="which model to use", type=str)
    parser.add_argument("-lr","--learning_rate", help="base learning rate", type=float)
    parser.add_argument("-ep","--epochs", help="number of epochs", type=int)

    args = parser.parse_args()
    if args.splits:
        args.splits = [int(split) for split in args.splits.split(",")]
    if args.groups:
        args.groups = args.groups.split("|")
        groups = {}
        for group in args.groups:
            category = group.split(":")[0]
            keywords = group.split(":")[-1]
            if ">>" in category:
                max_images = category.split(">>")[-1]
                category = category.split(">>")[0]
            else:
                max_images = 100
            keywords = keywords.split(",")
            groups[category] = {
                "keywords":keywords,
                "max_image_per_keyword":[0]*len(keywords),
                "max_images":max_images
            }
        args.groups = groups
    if args.augmentations:
        args.augmentations = args.augmentations.split(",")
    
    main(args)