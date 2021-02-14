import os
import sys
import json
import asyncio
from pathlib import Path

import pandas as pd

import crawler
import async_downloder

def read_config(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data

def download_images(config):

    working_dir = Path(config['working_dir'])
    if 'proxy' in config:
        proxy_type = config['proxy']['proxy_type']
        proxy = config['proxy']['proxy']
    else:
        proxy_type = None
        proxy = None

    download_timeout = 20 #standard
    if 'download_timeout' in config:
        download_timeout = config['download_timeout']

    details = {
        "category":[],
        "keyword":[],
        "path":[],
        "image_url":[]
    }
    
    for category, category_config in config['groups'].items():
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

            images = crawler.crawl_image_urls(keyword, max_number=keyword_image_limit, proxy=proxy, proxy_type=proxy_type)
            print('images crawled')
            image_filenames = list(map(lambda x: "{}_{}.jpg".format(keyword,x),range(len(images))))
            image_filenames = list(map(lambda x: str(category_path/x), images))
            images_detail = {filename: image for image, filename in zip(images, image_filenames)}
            asyncio.run(async_downloder.fetch_all(images_detail, timeout=download_timeout, proxy_type=proxy_type, proxy=proxy))
            print('images downloaded')

            details["category"].extend([category]*len(images))
            details["keyword"].extend([keyword]*len(images))
            details["path"].extend(image_filenames)
            details["image_url"].extend(images)

    details = pd.DataFrame(details)
    return details

def is_valid_path(image_path):
    return os.path.exists(image_path)

def validate_images(images_df):
    images_df['is_valid_filepath'] = images_df['path'].apply(is_valid_path)
    valid = len(images_df[images_df['is_valid_filepath']])
    invalid = len(images_df[~images_df['is_valid_filepath']])
    total = len(images_df)
    print('Total {} - valid {}, invalid {}'.format(total, valid, invalid))

    images_df = images_df[images_df['is_valid_filepath']]
    return images_df

def main(config_filename):
    if 'json' not in config_filename:
        return
    
    config = read_config(config_filename)
    images_df = download_images(config)
    images_df = validate_images(images_df)
    details_filename = config_filename.replace('.json','.csv')
    images_df.to_csv(details_filename, index=False)
    return details_filename

if __name__ == '__main__':

    print('done', main(sys.argv[1]))