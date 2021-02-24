import os
import sys
# import json
import shutil
import asyncio
from pathlib import Path

import pandas as pd
from PIL import Image

import Crawler
import AsyncDownloader

pd.options.mode.chained_assignment = None

# def read_config(filename):
#     with open(filename, "r") as file:
#         data = json.load(file)
#     return data

# def download_images(config):
def download_images(working_dir, groups, proxy=None, proxy_type=None, download_timeout=20):

    # working_dir = Path(config['working_dir'])
    # if 'proxy' in config:
    #     proxy_type = config['proxy']['proxy_type']
    #     proxy = config['proxy']['proxy']
    # else:
    #     proxy_type = None
    #     proxy = None

    # download_timeout = 20 #standard
    # if 'download_timeout' in config:
    #     download_timeout = config['download_timeout']

    working_dir = Path(working_dir)
    details = {
        "category":[],
        "keyword":[],
        "path":[],
        "image_url":[]
    }
    
    for category, category_config in groups.items():
        print('running for category: {}'.format(category))
        category_path = working_dir/category
        if not os.path.exists(category_path):
            os.mkdir(category_path)
        
        keywords = category_config['keywords']
        keyword_image_limits = category_config['max_image_per_keyword']
        category_image_limit = category_config['max_images']
        default_keyword_image_limit = category_image_limit//len(keywords)
        
        for keyword, keyword_image_limit in zip(keywords, keyword_image_limits):
            print('running for keyword: {}'.format(keyword))
            if keyword_image_limit == 0:
                keyword_image_limit = default_keyword_image_limit

            images = Crawler.crawl_image_urls(keyword, max_number=keyword_image_limit, proxy=proxy, proxy_type=proxy_type)
            print('images crawled')
            image_filenames = list(map(lambda x: "{}_{}.jpg".format(keyword,x),range(len(images))))
            image_filenames = list(map(lambda x: str(category_path/x), image_filenames))
            images_detail = {filename: image for image, filename in zip(images, image_filenames)}
            asyncio.run(AsyncDownloader.fetch_all(images_detail, timeout=download_timeout, proxy_type=proxy_type, proxy=proxy))
            print('images downloaded')

            details["category"].extend([category]*len(images))
            details["keyword"].extend([keyword]*len(images))
            details["path"].extend(image_filenames)
            details["image_url"].extend(images)

    details = pd.DataFrame(details)
    return details

def is_valid_path(image_path):
    if not isinstance(image_path, str):
        return False
    
    if not os.path.exists(image_path):
        return False
    
    statfile = os.stat(image_path)
    filesize = statfile.st_size
    if filesize == 0:
        return False
    
    try:
        im = Image.open(image_path)
        im.verify() #I perform also verify, don't know if he sees other types o defects
        im.close()
    except Exception as e:
        return False
    
    return True

def validate_images(images_df):
    images_df['is_valid_filepath'] = images_df['path'].apply(is_valid_path)
    valid = len(images_df[images_df['is_valid_filepath']])
    invalid = len(images_df[~images_df['is_valid_filepath']])
    total = len(images_df)
    print('Total {} - valid {}, invalid {}'.format(total, valid, invalid))

    images_df = images_df[images_df['is_valid_filepath']]
    images_df.drop(['is_valid_filepath'], axis=1, inplace=True)
    return images_df

def move_file(source, destination):
    if not destination.exists():
        source.replace(destination)

def get_split_path(existing_path, new_path):
    filename = str(existing_path).split('/')[-1].split('\\')[-1]
    new_path = new_path/filename
    return new_path

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

    for category in splits_df.category.unique().tolist():
        shutil.rmtree(working_dir/category)     #folder cleanup
    
    return splits_df

# def main(config_filename):
#     # if 'json' not in config_filename:
#         # return
    
#     # config = read_config(config_filename)
#     details_filename = config_filename.replace('.json','.report.csv')
#     images_df = download_images(config)
#     images_df = validate_images(images_df)
#     images_df = train_test_split(images_df, config['datasets'], config['working_dir'])
#     images_df.to_csv(details_filename, index=False)
#     return images_df

# if __name__ == '__main__':
#     details_frame = main(sys.argv[1])