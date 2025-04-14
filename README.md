# LITECOIN PRICE PREDICTION USING LSTM ALGORITHM
**Proyek Analisis Prediksi Harga Mata Uang Kripto Litecoin Menggunakan Algoritma Long Short Term Memory**

📌 **Dibuat untuk**: Skripsi gelar Sarjana S1 Teknik Informatika

-------------------------------------------------------------------


## 📝 Deskripsi Proyek
Proyek ini mengimplementasikan Algoritma **Long Short Term Memory (LSTM)** untuk melakukan analisis prediksi **Time Series** berdasarkan 2 kategori:
- Tanggal
- Harga Penutupan

Dataset yang digunakan: <a href="https://raw.githubusercontent.com/curzedb/LSTM_LTC-USD/refs/heads/main/Data/LTC-USD%20LSTM.csv">LTC-USD LSTM.csv</a> (700+ Data).

-----------------------------------------------------------------------

## 🛠️ Tools
- **Bahasa Pemrograman**: Python 3
- **Framework**: TensorFlow
- **Libraries**: datetime, itertools, math, matplotlib, numpy, os, pandas, plotly, seaborn, sklearn, warnings 
- **Platform**: Jupyter Notebook

-----------------------------------------------------------------------
## 📊 Struktur Dataset  
```
dataset/  
  ├── Date  
  ├── Open
  ├── High
  ├── Low
  ├── Close
  ├── Adj_Close
  └── Volume 
   
```
| Date | Open | High | Low | Close | Adj Close | Volume |
|------|------|------|-----|-------|-----------|--------|
| 2022-03-01	| 113.476082	| 115.579834	| 110.730995	| 112.544044	| 112.544044	| 951568702 |
| 2022-03-02	| 112.540298	| 114.243759	| 109.672516	| 110.351357	| 110.351357	| 838169853 |
| ... | ... | ... | ... | ... | ... | ... |
| 2024-03-31	| 102.866302	| 106.548813	| 101.554459	| 105.183403	| 105.183403	| 638798519 |
| 2024-04-01	| 105.183403	| 112.315865	| 97.518333	| 99.375343	| 99.375343	| 1354224503 |

Disederhanakan menjadi: 
```
dataset/  
  ├── Date  
  └── Close
   
```

| NO | Date | Close |
|----|------|-------|
| 30 |	2022-03-31	| 123.716011 |
| 31 |	2022-04-01	| 124.883179 |
| 32 |	2022-04-02	| 124.923485 |
| 33 |	2022-04-03	| 128.946213 |
| 34 |	2022-04-04	| 124.861702 |
| ... |	... |	... |
| 758 |	2024-03-28	| 94.219116 |
| 759	| 2024-03-29	| 109.258972 |
| 760	| 2024-03-30	| 102.863113 |
| 761	| 2024-03-31	| 105.183403 |
| 762	| 2024-04-01	| 99.375343 |

**Pembagian Data**:
| Jenis Data  | Jumlah Data | Persentase Data |
|-------------|-------------|-----------------|
| Data Latih  | 513         | 70%             |
| Data Uji    | 220         | 30%             |

---------------------------------------------------------------------
## 🧠 Arsitektur Model LSTM
```python
time_step = 10
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)
```
```python
# membentuk ulang input menjadi [samples, time steps, features] yang diperlukan untuk LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
```
```python
#menentukan model, neuron, loss, dan optimizer
model=Sequential()

model.add(LSTM(10,input_shape=(None,1),activation="relu"))

model.add(Dense(1))

model.compile(loss="mean_squared_error",optimizer="adam")
```
```python
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=500,batch_size=64,verbose=1)
```
**Hyperparameter**:  
- Optimizer: `Adam`  
- Loss: `mean_squared_error`  
- Epochs: `500`  
- Batch Size: `64`
- Dense: `1`
- Activation: `relu`

-----------------------------------------------------------------------
## 📈 Hasil Evaluasi Berdasarkan R2 Data Uji(Acurracy) dan RMSE(Loss) 
| Metric      | Validation |
|-------------|------------|
| Accuracy    | 91.28%     |
| Loss        | 2.69       |

-----------------------------------------------------------------------
## ☑️Implementasi
![image](https://github.com/user-attachments/assets/0f0854bd-96b9-4abe-8a61-a893fcd12183)



