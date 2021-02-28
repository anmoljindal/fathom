import os
import sys
import shutil
import asyncio
from pathlib import Path

import pandas as pd
from PIL import Image

import Crawler
import AsyncDownloader

pd.options.mode.chained_assignment = None

def download_images(working_dir, groups, proxy=None, proxy_type=None, download_timeout=20):

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