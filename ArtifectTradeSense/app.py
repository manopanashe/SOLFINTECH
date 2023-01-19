import datetime
import time
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from  keras.layers import  Dense
from keras.layers import Dropout
import sqlite3


#Connect to Database
conn = sqlite3.connect('data.db')
c = conn.cursor()
def create_user_table():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_user_data(username,password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
    user_data = c.fetchall()
    return  user_data

def view_all_users():
    c.execute('SELECT * FROM usertable')
    user_info = c.fetchall()
    return user_info
#------------------------------------------------------------------------------------------------------------------


st.set_page_config(layout='wide',initial_sidebar_state='expanded')
st.title('Trade Sense')
st.write('Welcome to SOLFINTECH Stock Prediction App. Helping you make informative and beneficial Stock trading choices ')
#Download Tickers
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

with st.sidebar:
    st.header('Menu')
    select_stck = st.selectbox('Please Select the Company for prediction:',tickers)
    START = st.sidebar.date_input('Please Enter the Start date', datetime.date(2016, 1, 1))
    TODAY = st.sidebar.date_input('Please Enter the End Date', datetime.date.today())
    menu = ['Login', 'SignUp']
    choice = st.sidebar.selectbox('Further Access', menu)

@st.cache()
def load_data(ticker):
    train_data = yf.download(ticker,START,TODAY )
    train_data.isna()
    return train_data

train_data = load_data(select_stck)
train_data.to_csv('stock_data.csv')

df = pd.read_csv('stock_data.csv')
df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x.split()[0]))
df.set_index('Date',drop=True,inplace=True)


def plot_Close_raw_data():
    # Create subplots to plot graph and control axes
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df['Close'])
    # Set figure title
    plt.title('Close Stock Price History [2016 - Present Day]', fontsize=16)
    # Set x label
    plt.xlabel('Date', fontsize=14)
    # Set y label
    plt.ylabel('Closing Stock Price in $', fontsize=14)
    # Rotate and align the x labels
    fig.autofmt_xdate()
    # Show plot
    st.pyplot(fig)


st.subheader('Raw data')
st.write(df.tail(5))
plot_Close_raw_data()




# Create a new DataFrame with only closing price and date
data = df[['Open','Close']]
predicting_state = st.text('Calculating Prediction')
# Scaling our data
from sklearn.preprocessing import MinMaxScaler
Ms = MinMaxScaler()
data[data.columns] = Ms.fit_transform(data)

# Split data into train and test data
training_size = round(len(data) * 0.80)
train_data = data[:training_size]
test_data  = data[training_size:]

#Create sequences
def create_sequence(dataset):
    sequences = []
    labels = []
    start_idx = 0

    for stop_idx in range(50,len(dataset)):
        sequences.append(dataset.iloc[start_idx:stop_idx])
        labels.append(dataset.iloc[stop_idx])
        start_idx += 1
    return (np.array(sequences),np.array(labels))

train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)
# Our Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))
model.add(Dropout(0.1))
model.add(LSTM(units=50))
model.add(Dense(2))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(train_seq, train_label, epochs=80,validation_data=(test_seq, test_label), verbose=1)
test_predicted = model.predict(test_seq)

# Inversing scaling on predicted data
test_inverse_predicted = Ms.inverse_transform(test_predicted)
#Merging predicted data to our dataset

new_stock_data = pd.concat([data.iloc[-320:].copy(),pd.DataFrame(test_inverse_predicted,columns=['Open_predicted','Close_predicted'],index=data.iloc[-305:].index)], axis=1)
# Inverse scaling new dataset
new_stock_data[['Open','Close']] = Ms.inverse_transform(new_stock_data[['Open','Close']])

# Creating a dataframe and adding 10 days to existing index
new_stock_data = new_stock_data.append(pd.DataFrame(columns=new_stock_data.columns,index=pd.date_range(start=new_stock_data.index[-1], periods=11, freq='D', closed='right')))

#Adding Predictions to dataset
up_coming_pred = pd.DataFrame(columns=['Open','Close'],index=new_stock_data.index)
up_coming_pred.index=pd.to_datetime(up_coming_pred.index)
up_coming_pred.shape
curr_seq = test_seq[-1:]
for i in range(-10,0):
  up_pred = model.predict(curr_seq)
  up_coming_pred.iloc[i] = up_pred
  curr_seq = np.append(curr_seq[0][1:],up_pred,axis=0)
  curr_seq = curr_seq.reshape(test_seq[-1:].shape)

#Inversing predicted data
up_coming_pred[['Open','Close']] = Ms.inverse_transform(up_coming_pred[['Open','Close']])
predicting_state.text('Prediction Complete!')

price = up_coming_pred.iloc[329]['Close']
st.write('The Predicted Price for tomorrow is $',price)

fig2, ax = plt.subplots(figsize=(10, 4))
ax.plot(new_stock_data.loc['2016-01-01':, 'Close'], label='Current close Price')
ax.plot(up_coming_pred.loc['2016-01-01':, 'Close'], label='Upcoming close Price')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.set_xlabel('Date', size=15)
ax.set_ylabel('Stock Price', size=15)
ax.set_title('Upcoming close price prediction', size=15)
ax.legend()
st.pyplot(fig2)

# Plotting our data
fig = plt.figure(figsize=(10, 6))
plt.plot(new_stock_data['Close'], color='blue', label='Close')
plt.plot(new_stock_data['Close_predicted'], color='red', label='Predicted Close')
plt.xticks(rotation=45)
plt.xlabel('Date', size=15)
plt.ylabel('Stock Price', size=15)
plt.title('Actual vs Predicted for Close Price', size=15)
plt.legend()
st.pyplot(fig)

st.title('To access further features Please Sign Up or Log In')
if choice == 'Login':
    st.sidebar.subheader('Log In')
    username = st.sidebar.text_input('User Name')
    password = st.sidebar.text_input('Password', type='password',key='log_in_password')
    login_btn = st.sidebar.checkbox('Login')
    if login_btn:
            #if Password == Password:
            create_user_table()
            result = login_user(username,password)
            if result :
                st.success('Logged in as {}'.format(username))

                #Second Plot
                fg, ax = plt.subplots(figsize=(10, 4))
                ax.plot(new_stock_data.loc['2021-04-01':, 'Open'], label='Current Open Price')
                ax.plot(up_coming_pred.loc['2021-04-01':, 'Open'], label='Upcoming Open Price')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                ax.set_xlabel('Date', size=15)
                ax.set_ylabel('Stock Price', size=15)
                ax.set_title('Upcoming Open price prediction', size=15)
                ax.legend()
                st.pyplot(fg)

            else:
                st.warning('Incorrect Username/Password')
elif choice == 'SignUp':
    st.sidebar.subheader('Create New Account')
    new_user = st.sidebar.text_input('Username')
    new_password = st.sidebar.text_input('Password',type='password',key='sign_in_password')
    if st.sidebar.button('Signup'):
            create_user_table()
            add_user_data(new_user,new_password)
            st.sidebar.success('You have successfully created an account')
            st.sidebar.info('Go to Access Point to log into your account')

