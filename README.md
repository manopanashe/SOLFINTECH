# SOLFINTECH

Simple Stock Prediction Web application

# How to Run
- 1. Clone the repo 
  - git clone https://github.com/manopanashe/SOLFINTECH.git
- 2. Change directory to the application
  - cd app
- Update iloc values to curent Row Index 
  - Merging predicted data to our dataset
  new_stock_data = pd.concat([data.iloc[-320:].copy(),pd.DataFrame(test_inverse_predicted,columns=['Open_predicted','Close_predicted'],index=data.iloc[-304:].index)], axis=1)
- 3. Run the Application
   - streamlit run app.py