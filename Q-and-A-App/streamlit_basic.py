# pip install streamlit
# streamlit run streamlit_basic.py
import streamlit as st
import pandas as pd

# Displaying data on the screen:
st.title("Hello there! Streamlit Basic :100: ")
st.write('I am learning Streamlit')

# Using st.write
# list
l1 = [1, 2, 3]
st.write(l1)

# dictionary
l2 = list('abc')
d1 = dict(zip(l1, l2))
st.write(d1)

# Using magic
'Displaying using magic :smile:!'
df = pd.DataFrame({
    'first_column': [1, 2, 3, 4],
    'second_column': [30, 40, 50, 10]
})

df

st.divider()

# Using Widgets
# Text input
name = st.text_input('Your name: ')
if name:
    st.write(f'Hello {name}')

st.divider()

# Number input
x = st.number_input('Enter a number', min_value=1, max_value=50, step=2)
st.write(f'The current number is {x}')

st.divider()

# Button 
click = st.button('Click Me')
if click:
    st.write(':ghost:' * 3)

st.divider()

# Checkbox
agree = st.checkbox('I agree')
if agree: 
    'Great, you agreed!'

check = st.checkbox('Continue', value=True)
if check:
    ':+1:' * 3


df = pd.DataFrame({
    'Name': ['Anne', 'Mario', 'Sarah'],
    'Age': [20, 30, 35]
})

if st.checkbox('Show data'):
    st.write(df)

st.divider()

# Radio
pets = ['cat', 'dog', 'fish', 'parrot']
pet = st.radio('Favorite pet', pets, index=1, key='your_pet')
st.write(f'Your favorite pet: {pet}')
st.write(f'Your favorite pet: {st.session_state.your_pet}')

st.divider()

# Select
cities = ['London', 'Berlin', 'Paris', 'New York']
city = st.selectbox('Your city', cities, index=2)
st.write(f'You live in {city}')

st.divider()

# Slider
x = st.slider('x', value=24, min_value=18, max_value=80, step=2)
st.write(f'x is {x}')

st.divider()

# File uploader
uploaded_file = st.file_uploader('Upload a file:', type=['txt', 'csv', 'xlsx'])
if uploaded_file:
    st.write(uploaded_file)
    if uploaded_file.type == 'text/plain':
        from io import StringIO
        stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
        string_data = stringio.read()
        st.write(string_data)
    elif uploaded_file.type == 'text/csv':
        df = pd.read_csv(uploaded_file)
        st.write(df)
    else:
        df = pd.read_excel(uploaded_file)
        st.write(df)

st.divider()

# Camera input
camera_photo = st.camera_input('Take a photo')
if camera_photo:
    st.image(camera_photo)

st.image('https://images.unsplash.com/photo-1700359396628-9bb38cc3d7e2?q=80&w=3087&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')


# Sidebar select box
my_select_box = st.sidebar.selectbox('Select', ['US', 'UK', 'DE', 'FR'])

st.sidebar.divider()

# Sidebar slider
my_slider = st.sidebar.slider('Amount of Oranges')
st.sidebar.write(f'I have {my_slider} oranges')

st.divider()

# Columns
left_column, right_column = st.columns(2)

import random
data = [random.random() for _ in range(100)]

with left_column:
    st.subheader('A Line Chart')
    st.line_chart(data)

right_column.subheader('Data')
right_column.write(data[:10])

# columns with different width
col1, col2, col3 = st.columns([0.2, 0.5, 0.3])

col1.markdown('Hello World!')
col2.write(data[5:10])

with col3:
    st.subheader('A cat')
    st.image('https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8Y2F0fGVufDB8fDB8fHww')

st.divider()

# Expander
with st.expander('Click to expand'):
    st.bar_chart({'Data': [random.randint(2, 10) for _ in range(25)]})
    st.write('A dog image below')
    st.image('https://images.unsplash.com/photo-1517849845537-4d257902454a?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8ZG9nfGVufDB8fDB8fHwwv')

st.divider()

# # Displaying a progress Bar
# import time

# st.write('Starting an extensive computation ...')
# latest_iteration = st.empty()

# progress_text = 'Operation in progress! Please wait ...'
# my_bar = st.progress(0, text=progress_text)
# time.sleep(2)

# for i in range(100):
#     my_bar.progress(i+1)
#     latest_iteration.text(f'Iteration {i+1}')
#     time.sleep(0.1)

# st.write('We are done!')

st.divider()

# Session state
st.header('Streamlit Session')
st.write(st.session_state)

if 'counter' not in st.session_state:
    st.session_state['counter'] = 0
else:
    st.session_state.counter += 1

st.write(f'Counter: {st.session_state.counter}')

button = st.button('Update state')
if 'clicks' not in st.session_state:
    st.session_state['clicks'] = 0

if button:
    st.session_state['clicks'] += 1
    f'After pressing button {st.session_state}'

number = st.slider('Value', 1, 10, key='my_slider')
st.write(st.session_state)
st.write(number)

st.divider()

# Callbacks
st.subheader('Distance converter')

def miles_to_km():
    st.session_state.km = st.session_state.miles * 1.609

def km_to_miles():
    st.session_state.miles = st.session_state.km * 0.621


col4, buff, col5 = st.columns([2, 1, 2])
with col4: 
    miles = st.number_input('Miles:', key='miles', on_change=miles_to_km)

with col5:
    km = st.number_input('Km:', key='km', on_change=km_to_miles)