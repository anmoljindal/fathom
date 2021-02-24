import os
import re
import time

from urllib.parse import unquote, quote
from selenium import webdriver

import Meta

def google_gen_query_url(keywords, search_options=["safe_mode"]):
    base_url = "https://www.google.com/search?tbm=isch&hl=en"
    keywords_str = "&q=" + quote(keywords)
    query_url = base_url + keywords_str
    if "face_only" in search_options:
        query_url += "&tbs=itp:face"
    if "safe_mode" in search_options:
        query_url += "&safe=on"
    else:
        query_url += "&safe=off"
    return query_url

def google_image_url_from_webpage(driver, max_number):
    thumb_elements_old = []
    thumb_elements = []
    while True:
        try:
            thumb_elements = driver.find_elements_by_class_name("rg_i")
            # print("Find {} images.".format(len(thumb_elements)))
            if len(thumb_elements) >= max_number:
                break
            if len(thumb_elements) == len(thumb_elements_old):
                break
            thumb_elements_old = thumb_elements
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            show_more = driver.find_elements_by_class_name("mye4qd")
            if len(show_more) == 1 and show_more[0].is_displayed() and show_more[0].is_enabled():
                # print("Click show_more button.")
                show_more[0].click()
            time.sleep(3)
        except Exception as e:
            # print("Exception ", e)
            pass
    
    if len(thumb_elements) == 0:
        return []

    # print("Click on each thumbnail image to get image url, may take a moment ...")

    retry_click = []
    for i, elem in enumerate(thumb_elements):
        try:
            if i != 0 and i % 50 == 0:
                pass
                # print("{} thumbnail clicked.".format(i))
            if not elem.is_displayed() or not elem.is_enabled():
                retry_click.append(elem)
                continue
            elem.click()
        except Exception as e:
            # print("Error while clicking in thumbnail:", e)
            retry_click.append(elem)

    if len(retry_click) > 0:    
        # print("Retry some failed clicks ...")
        for elem in retry_click:
            try:
                if elem.is_displayed() and elem.is_enabled():
                    elem.click()
            except Exception as e:
                pass
                # print("Error while retrying click:", e)
    
    image_elements = driver.find_elements_by_class_name("islib")
    image_urls = list()
    url_pattern = r"imgurl=\S*&amp;imgrefurl"

    for image_element in image_elements[:max_number]:
        outer_html = image_element.get_attribute("outerHTML")
        re_group = re.search(url_pattern, outer_html)
        if re_group is not None:
            image_url = unquote(re_group.group()[7:-14])
            image_urls.append(image_url)
    return image_urls

def crawl_image_urls(keywords, max_number=10000, search_options=["safe_mode"],
                     proxy=None, proxy_type="http"):
    chrome_path = os.path.join(Meta.metadata_path, 'chromedriver')
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("headless")
    if proxy is not None and proxy_type is not None:
        chrome_options.add_argument("--proxy-server={}://{}".format(proxy_type, proxy))

    query_url = google_gen_query_url(keywords, search_options)
    driver = webdriver.Chrome(chrome_path, chrome_options=chrome_options)
    driver.set_window_size(1920, 1080)
    driver.get(query_url)
    image_urls = google_image_url_from_webpage(driver, max_number)

    if max_number > len(image_urls):
        output_num = len(image_urls)
    else:
        output_num = max_number

    # print("\n== {0} out of {1} crawled images urls will be used.\n".format(output_num, len(image_urls)))
    return image_urls[0:output_num]