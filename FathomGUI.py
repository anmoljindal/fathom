import os
from re import split
import time
import shutil
from copy import deepcopy

import SessionState
import streamlit as st

working_dir = os.path.dirname(os.path.realpath(__file__))
state = SessionState.get(groups={})

st.title("Fathom")
st.markdown(
    'create your own image classifier'
)

navigation_choice = st.sidebar.radio("go to",("New Project","Project Explorer"))

def get_category_group_text(groups):

    details_list = []
    for category, details in groups.items():
        keywords = ",".join(details['keywords'])
        details_list.append("- **category**: {}\t **keywords**: {}\t **max images**: {}".format(category, keywords, details['max_images']))
    
    return "\n ".join(details_list)

def create_project():
    project_json = {}
    project_json['project'] = project_name

    project_path = os.path.join(working_dir, project_name)
    project_json['working_dir'] = project_path
    if os.path.exists(project_path):
        shutil.rmtree(project_path)
        os.mkdir(project_path)
    
    project_json['groups'] = deepcopy(state.groups)
    state.groups.clear()
    
    if sum(splits) > 100:
        raise Exception('value error')
    datasets = [
        ["train", int(splits[0])],
        ["validation", int(splits[1])],
        ["test", int(splits[2])]
    ]
    project_json['datasets'] = datasets
    
    project_json['augmentations'] = augmentations
    project_json['batch_size'] = batch_size
    project_json['model'] = model_name
    project_json['image_size'] = image_size
    project_json['base_learning_rate'] = base_learning_rate
    project_json['epochs'] = epochs

    return project_json

if navigation_choice == "New Project":
    st.header("New Project")
    project_name = st.text_input("Project Name")

    ##Dataset information
    st.subheader("Dataset Information")
    category_name = st.text_input("category name")
    keywords = st.text_input("keywords seperated by comma")
    max_images = st.number_input("total number of images needed", value=1000, step=100)
    
    if st.button("+ Category") and len(category_name) != 0:
        keywords_list = keywords.split(',')
        category_group = {
            category_name:{
                "keywords":keywords_list,
                "max_image_per_keyword":[0]*len(keywords_list),
                "max_images":max_images
            }
        }
        state.groups.update(category_group)

    category_groups_text = get_category_group_text(state.groups)
    st.markdown(category_groups_text)

    split_columns = st.beta_columns(3)
    splits = [None]*3
    with split_columns[0]:
        splits[0] = st.number_input("train split", min_value=0, max_value=100, step=1)
    with split_columns[1]:
        splits[1] = st.number_input("validation split", min_value=0, max_value=100, step=1)
    with split_columns[2]:
        splits[2] = st.number_input("test split", min_value=0, max_value=100, step=1)

    if sum(splits) > 100:
        st.error("total of data splits should be less than 100")

    image_size = st.number_input("image length", min_value=64, max_value=720, step=1)

    #Model information
    st.subheader("Model Information")
    model_options = [
        "mobilenet_v2"
    ]
    model_name = st.selectbox("model",model_options)
    
    augmentation_options = [
        "random flip",
        "random rotation"
    ]
    augmentations = st.multiselect("augmentations", augmentation_options)
    
    hyperparam_container = st.beta_columns(3)
    with hyperparam_container[0]:
        batch_size = st.number_input("batch size", min_value=4, max_value=126, step=2)
    with hyperparam_container[1]:
        base_learning_rate = st.number_input("base learning rate", min_value=0.0, max_value=1.0, step=0.0001)
    with hyperparam_container[2]:
        epochs = st.number_input("epochs", min_value=1, max_value=1000, step=1)

    if st.button("+ Project"):
        project_json = create_project()
        st.write(project_json)

if navigation_choice == "Project Explorer":
    st.header("Project Explorer")



