import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("C:/Users/Lenovo/Downloads/global_cancer_patients_2015_2024.csv")

# -- Datasetimin neye benzediği ile ilgili fikir sahibi olmak için df.head veya df.sample komutunu çalıştırıyorum
df.head(5)


# -- Datasetimin sütunlarının null/not null değerlerini ve veri tipini görmek için df.info kullanıyorum
df.info()


# -- Unique olan hasta numarasına ihtiyacım olmayacağından doğrudan dropluyorum
df.drop(columns=['Patient_ID'], inplace = True)



# -- Sütunlarımı kategorik ve nümerik olarak sınıflandırıyorum, böylece model eğitimi yaparken işimi kolaylaştıracak
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()



# -- Cancer_stage sütunumu modele sokmak için kolay bir hale getiriyorum

stage_mapping = {
    'Stage 0': 0,
    'Stage I': 1,
    'Stage II': 2,
    'Stage III': 3,
    'Stage IV': 4
}

df['Cancer_Stage'] = df['Cancer_Stage'].str.strip().str.upper().replace({
    'STAGE 0': 0,
    'STAGE I': 1,
    'STAGE II': 2,
    'STAGE III': 3,
    'STAGE IV': 4
})




# -- Hasta sayısı ve yaş arasındaki korelasyonu görebilmek için görselleştirme kullanıyorum
plt.figure(figsize=(10,6))
df['Age'].hist(bins=20, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.title('Distribution of Patients by Age')
plt.grid(True)
plt.savefig("C:/Users/Lenovo/Downloads/age_distribution.png")
plt.close()
# -- Verimin yaşlar arasında fark oluşturmayacak şekilde dağılım gösterdiğini gözlemliyorum




# -- Bir box-plot yardımı q1,q2(median),q3 gibi değerleri görerek ülkeler ve target severity score arasındaki korelasyonu görmek istiyorum
plt.figure(figsize=(14,6))
df.boxplot(column='Target_Severity_Score', by='Country_Region', rot=90)
plt.title('Target Severity Score Distribution by Country')
plt.suptitle('')
plt.xlabel('Country')
plt.ylabel('Target Severity Score')
plt.tight_layout()
plt.savefig("C:/Users/Lenovo/Downloads/severity_by_country.png")
plt.close()
# -- Verimin ülkeler arasında fark oluşturmayacak şekilde dağılım gösterdiğini gözlemliyorum




# -- Hasta sayısı ve severity score korelasyonuna göz atalım
severity_counts = df['Target_Severity_Score'].value_counts().sort_index()
plt.figure(figsize=(10,6))
plt.plot(severity_counts.index, severity_counts.values, marker='o', linestyle='-')
plt.xlabel('Target Severity Score')
plt.ylabel('Number of Patients')
plt.title('Number of Patients by Severity Score')
plt.grid(True)
plt.savefig("C:/Users/Lenovo/Downloads/patients_by_severity.png")
plt.close()
# -- Hastalarımın büyük çoğunluğunun target severity score'u 4-6 arasında olduğunu gözlemliyorum.




# -- Target severity score ve hasta sayısına ait bir gözlem yaptıktan sonra yaş için de bir korelasyon var mı bakalım
correlation = df['Age'].corr(df['Target_Severity_Score'])
print(f'Pearson correlation between Age and Severity Score: {correlation:.3f}')
# -- Neredeyse hiçbir ilişki olmadığını görüyorum




# -- Çok nadir olan kategorik değişkenlerimi overfitting'e ve curse of dimensionality'e sebep vermemek için elimine ediyorum

def group_rare_categories(df, column, threshold=0.01):
    value_counts = df[column].value_counts(normalize=True)
    rare_categories = value_counts[value_counts < threshold].index
    df[column] = df[column].apply(lambda x: 'Other' if x in rare_categories else x)
    return df
df = group_rare_categories(df, 'Country_Region', threshold=0.01)
df = group_rare_categories(df, 'Cancer_Type', threshold=0.01)




# -- Model eğitimi için One-hot encoded sütunlar oluşturuyorum.
df = pd.get_dummies(df, columns=['Gender', 'Country_Region', 'Cancer_Type'], drop_first=True)




from sklearn.preprocessing import StandardScaler

# -- Veri setimizi feature columns ve label olarak ikiye ayırabiliriz
x = df.drop(columns=['Target_Severity_Score'])
y = df['Target_Severity_Score']



# -- Boolean sütunları int'e çevirelim
bool_cols = x.select_dtypes(include='bool').columns
x[bool_cols] = x[bool_cols].astype(int)



# -- Sayısal verilerimi modele sokmadan önce ölçekliyorum
from sklearn.preprocessing import StandardScaler

numeric_cols = x.select_dtypes(include='number').columns
scaler = StandardScaler()
x[numeric_cols] = scaler.fit_transform(x[numeric_cols])




# -- Veriyi eğitim ve test olmak üzere ikiye ayıralım, burada random_state yani rastgele veri kümesi için gelenek olduğu için 42 seçiyorum
# -ancak başka bir değer de seçilebilirdi, verimi ise 80/20 olarak ikiye ayırıyorum
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



# -- Verimi bir decision tree based olan random forest regressor classification modeline sokuyorum
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train)


# -- Değerlerimin Mean Squared Error ve Root Mean Squared Error, R Square gibi değerlerini hesaplıyorum, bu sayede
# - ne kadar doğru tahmin yapabildiğimi göreceğim.
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")



# -- Cross validation aşamasına geçelim ve modelimizin istikrarını görelim.
from sklearn.model_selection import cross_val_score

model = RandomForestRegressor(random_state=42)

# -- Cross-validation (5 katlı)
mse_scores = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=5)
rmse_scores = np.sqrt(-mse_scores)

print("Cross-Validated RMSE Scores (fold bazlı):", rmse_scores)
print("Average CV RMSE:", round(np.mean(rmse_scores), 4))
# -- Cross validation sürecinden ve average cv rmse ölçümünden başarılı bir sonuç aldığımızı gözlemliyoruz.


# -- Modeli tüm veri üzerinde uygulayalım

model = RandomForestRegressor(random_state=42)
model.fit(x, y)

# -- Hangi feature'ların bu doğru tahminleme yapmamızı sağladığını çıkartabilmek için feature importance çıkartalım
importances = model.feature_importances_

# Özellik isimleri ile birleştir
feature_importance_df = pd.DataFrame({
    'Feature': x.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)


# -- Görselleştirelim
plt.figure(figsize=(12,6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance Score')
plt.title('Random Forest Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("C:/Users/Lenovo/Downloads/feature_importance_rf.png")
plt.close()



# -- Daha da doğru bir tahminleme yapabilir miyiz görebilmek için Hyperparameter Tuning yapalım
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

rf = RandomForestRegressor(random_state=42)

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

search.fit(x_train, y_train)

print("En iyi parametreler:", search.best_params_)
best_model = search.best_estimator_


# -- Tekrardan mse,rmse,r square değerlerimizi hesaplayalım.
from sklearn.metrics import mean_squared_error, r2_score
y_pred = best_model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


# - Hyperparameter tuning modelimizin overfitting ihtimalini azaltabilir ve genelleştirebilir dolayısıyla
# Dolayısıyla R-Squared değeri düşerken, RMSE değeri artabilir. (istemediğimiz bir durum)
# Ancak bizim işimizin en iyi genelleyen modeli seçmek olduğunu unutmamalıyız.




# -- Yorum

# - Modelin hem test setinde hem de cross-validation sürecinde düşük RMSE (~0.17)
#   ve yüksek R² (~0.98) skorları üretmesi, genel olarak oldukça başarılı ve
#   genellenebilir bir regresyon modeli ortaya koyduğumuzu göstermektedir.

# Random Forest tarafından hesaplanan feature importance skorlarına göre;
# - Sigara kullanımı (Smoking), genetik yatkınlık (Genetic_Risk) ve tedavi masrafları
#   (Treatment_Cost_USD), modelin hedef değişkeni tahminlemesinde en belirleyici
#   değişkenler olmuştur.
# - Alkol kullanımı (Alcohol_Use), hava kirliliği (Air_Pollution) ve obezite seviyesi
#   (Obesity_Level) gibi çevresel ve yaşam tarzı faktörleri de önemli katkılar
#   sağlamıştır.
# - Bu sonuçlar; sağlık politikalarının planlanması, risk yönetimi ve önleyici
#   müdahaleler açısından stratejik içgörüler sunmaktadır.

