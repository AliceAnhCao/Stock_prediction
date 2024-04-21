## Project: Web lấy giá chứng khoán online và mô phỏng biến động giá chứng khoán sử dụng thuật toán LSTM

## Thành viên:
1. Cao Hồng Vân Anh
2. Nguyễn Quốc Trung
3. Ngô Khánh Linh


## Install Library 
pip install -r requirements.txt

## To run the code: 
- cd to folder của project
- streamlit run app_FE.py

## Dự án bao gồm các file sau:
- Stock_model.ipynb: File code phân tích và xây dựng mô hình bằng thuật toán LSTM
- app_FE.py: File code trực quan hóa kết quả phân tích và xây dựng mô hình lên web
- stock_pre.h5: File chứa thông tin mô hình được save xuống
- best_model_stock_pre.h5: chứa thông tin mô hình sau khi được tunning  
