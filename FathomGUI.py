import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2

import Meta
import SessionState
import FathomCLI

state = SessionState.get(groups={})

st.title("Fathom")
st.markdown(
	'create your own image classifier'
)

navigations = (
	"Home",
	"Create",
	"Datasets",
	"Models",
	"Deploy"
)
navigation_choice = st.sidebar.radio("go to",navigations)
st.sidebar.markdown("---")

def paginator(label, items, items_per_page=10, on_sidebar=True):
	"""Lets the user paginate a set of items.
	Parameters
	----------
	label : str
		The label to display over the pagination widget.
	items : Iterator[Any]
		The items to display in the paginator.
	items_per_page: int
		The number of items to display per page.
	on_sidebar: bool
		Whether to display the paginator widget on the sidebar.
		
	Returns
	-------
	Iterator[Tuple[int, Any]]
		An iterator over *only the items on that page*, including
		the item's index.
	Example
	-------
	This shows how to display a few pages of fruit.
	>>> fruit_list = [
	...     'Kiwifruit', 'Honeydew', 'Cherry', 'Honeyberry', 'Pear',
	...     'Apple', 'Nectarine', 'Soursop', 'Pineapple', 'Satsuma',
	...     'Fig', 'Huckleberry', 'Coconut', 'Plantain', 'Jujube',
	...     'Guava', 'Clementine', 'Grape', 'Tayberry', 'Salak',
	...     'Raspberry', 'Loquat', 'Nance', 'Peach', 'Akee'
	... ]
	...
	... for i, fruit in paginator("Select a fruit page", fruit_list):
	...     st.write('%s. **%s**' % (i, fruit))
	"""

	# Figure out where to display the paginator
	if on_sidebar:
		location = st.sidebar.empty()
	else:
		location = st.empty()

	# Display a pagination selectbox in the specified location.
	items = list(items)
	n_pages = len(items)
	n_pages = (len(items) - 1) // items_per_page + 1
	page_format_func = lambda i: "Page %s" % i
	page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

	# Iterate over the items in the page to let the user display them.
	min_index = page_number * items_per_page
	max_index = min_index + items_per_page
	import itertools
	return itertools.islice(enumerate(items), min_index, max_index)

def get_category_group_text(groups):

	details_list = []
	for category, details in groups.items():
		keywords = ",".join(details['keywords'])
		details_list.append("- **category**: {}\t **keywords**: {}\t **max images**: {}".format(category, keywords, details['max_images']))
	
	return "\n ".join(details_list)

if navigation_choice == "Home":
	pass
elif navigation_choice == "Create":
	st.header("Create New Project")
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

	split_container = st.beta_columns(3)
	splits = [None]*3
	with split_container[0]:
		splits[0] = st.number_input("train split", min_value=10, max_value=100, step=1, value=70)
	with split_container[1]:
		splits[1] = st.number_input("validation split", min_value=5, max_value=100, step=1, value=15)
	with split_container[2]:
		splits[2] = st.number_input("test split", min_value=0, max_value=100, step=1, value=15)

	if sum(splits) > 100:
		st.error("total of data splits should be less than 100")

	image_size = st.number_input("image length", min_value=64, max_value=720, step=1)

	#Model information
	st.subheader("Model Information")
	model_name = st.selectbox("model",Meta.model_options)
	augmentations = st.multiselect("augmentations", Meta.augmentation_options)
	
	hyperparam_container = st.beta_columns(3)
	with hyperparam_container[0]:
		batch_size = st.number_input("batch size", min_value=4, max_value=128, step=2)
	with hyperparam_container[1]:
		base_learning_rate = st.number_input("base learning rate", min_value=0.0, max_value=1.0, step=0.0001)
	with hyperparam_container[2]:
		epochs = st.number_input("epochs", min_value=1, max_value=1000, step=1)

	if st.button("+ Project"):
		project_json = FathomCLI.create_project(project_name, 
			groups=state.groups, splits=splits, augmentations=augmentations,
			batch_size=batch_size, image_size=image_size,
			model_name=model_name, base_learning_rate=base_learning_rate,
			epochs=epochs
		)
		state.groups.clear()
elif navigation_choice == "Datasets":
	projects = Meta.scan_projects()
	selection_row = st.beta_columns(2)
	with selection_row[0]:
		selected_project = st.selectbox("Select Project", list(projects.keys()))
	project_json = FathomCLI.get_project_details(selected_project)
	with selection_row[1]:
		selected_category = st.selectbox("Select Category", list(project_json['groups'].keys()))
	
	uploaded_files = st.file_uploader("Upload images", type=['png', 'jpg'], accept_multiple_files=True)
	operation_row = st.beta_columns(2)
	with operation_row[0]:
		if st.button("Upload Images"):
			FathomCLI.add_to_dataset(project_json, selected_category, uploaded_files)

	with operation_row[1]:
		if st.button("Download from google"):
			with st.spinner('Crawling and Downloading...'):
				dataset = FathomCLI.download_dataset(project_json)

	if 'image_details' in projects[selected_project]:
		dataset = pd.read_csv(projects[selected_project]['image_details'])
		grouping = dataset.groupby(['category','split']).size().reset_index(drop=False)
		grouping.rename(columns={0:"images"}, inplace=True)
		st.dataframe(grouping)

		columns_per_page = 4
		rows_per_page = 10
		dataset_list = dataset[dataset['category']==selected_category].to_dict(orient='records')
		data_chunks = [dataset_list[x:x+columns_per_page] for x in range(0, len(dataset_list), columns_per_page)]
		selected_images = []

		for _, chunk in paginator("select a page", data_chunks, items_per_page=rows_per_page):
			image_row = st.beta_columns(len(chunk))
			for item, image_cell in zip(chunk, image_row):
				with image_cell:
					check = st.checkbox(item['split'], key=item['path'])
					try:
						image = Image.open(item['path']).convert('RGB')
						st.image(image)
					except:
						continue
					if check:
						selected_images.append(item['path'])

		if st.sidebar.button("remove images"):
			dataset = FathomCLI.remove_from_dataset(project_json, selected_images)

elif navigation_choice == "Models":
	pass
elif navigation_choice == "Deploy":
	pass