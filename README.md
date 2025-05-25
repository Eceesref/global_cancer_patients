Bu proje, kanser hastalarına ait multidisipliner verilerin işlenerek bireysel sağlık risklerinin tahmini üzerine etkili bir çözüm üretmiştir.

Projenin başarıyla gösterdiği noktalar:
 - Gerçek dünya sağlık verisi üzerinde anlamlı sonuçlar elde edilmesi
 - Regresyon modellemesi ve model açıklanabilirliği süreçlerinin uygulanması
 - Tahminleme doğruluğu yüksek, overfitting riski düşük bir model kurulması

İleride proje şu yönlerde genişletilebilir:
* Tavsiye sistemleri: Hasta profiline göre tedavi öneri motoru kurulabilir
* Anomali tespiti: Aykırı veya beklenmedik skorlar erken fark edilebilir
* Survival Analysis: Survival_Years değişkeni ile hayatta kalma süresi analizi yapılabilir
* Clustering: Unsupervised learning ile hasta segmentasyonu uygulanabilir
* Model deploy: Model, API formatında dış sistemlere entegre edilebilir

Kullandığım datasetin Kaggle linki : https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024

Kaggle notebook linki : https://www.kaggle.com/code/eceeref/notebook7f33250f3c

Visual Studio Code ile aldığım Output:


Data columns (total 15 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   Patient_ID             50000 non-null  object 
 1   Age                    50000 non-null  int64  
 2   Gender                 50000 non-null  object 
 3   Country_Region         50000 non-null  object 
 4   Year                   50000 non-null  int64  
 5   Genetic_Risk           50000 non-null  float64
 6   Air_Pollution          50000 non-null  float64
 7   Alcohol_Use            50000 non-null  float64
 8   Smoking                50000 non-null  float64
 9   Obesity_Level          50000 non-null  float64
 10  Cancer_Type            50000 non-null  object 
 11  Cancer_Stage           50000 non-null  object 
 12  Treatment_Cost_USD     50000 non-null  float64
 13  Survival_Years         50000 non-null  float64
 14  Target_Severity_Score  50000 non-null  float64
dtypes: float64(8), int64(2), object(5)
memory usage: 5.7+ MB
Pearson correlation between Age and Severity Score: -0.001
Mean Squared Error: 0.03
Root Mean Squared Error: 0.17
R² Score: 0.98
Cross-Validated RMSE Scores (fold bazlı): [0.17658423 0.17133507 0.16822222 0.17532885 0.17431679]
Average CV RMSE: 0.1732
En iyi parametreler: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None}
Mean Squared Error: 0.07
Root Mean Squared Error: 0.27
R² Score: 0.95
