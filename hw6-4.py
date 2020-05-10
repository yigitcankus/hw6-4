import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import bartlett
from scipy.stats import levene
from statsmodels.tsa.stattools import acf
from scipy.stats import jarque_bera
from scipy.stats import normaltest
import warnings
warnings.filterwarnings('ignore')

hava_durumu = pd.read_csv("weatherHistory.csv")

###########################################################################################
#Hedef değişkeninizin görünür sıcaklık ve sıcaklık arasındaki fark olduğu doğrusal
# bir regresyon modeli oluşturun. Açıklayıcı değişkenler olarak nem ve rüzgar hızı kullanın.
# Şimdi, modelinizi OLS kullanarak tahmin edin. Tahmin edilen katsayılar istatistiksel olarak anlamlı mıdır?
# Tahmini katsayılar önceki beklentileriniz doğrultusunda mı? Tahmin edilen katsayıları yorumlayınız.
# Hedef ve açıklayıcı değişkenler arasındaki ilişkiler nelerdir?


# hava_durumu["hedef_degisken"] = hava_durumu["Sicaklik"]-hava_durumu["gorunur_sicaklik"]
#
# Y = hava_durumu["hedef_degisken"]
# X = hava_durumu[["Nem","RuzgarHizi"]]
#
# lrm = linear_model.LinearRegression()
# lrm.fit(X, Y)
#
# print('Değişkenler: \n', lrm.coef_)
# print('Sabit değer (bias): \n\n\n', lrm.intercept_)
#
# X = sm.add_constant(X)
# results = sm.OLS(Y, X).fit()
# print(results.summary())


# Kullanılan özelliklerimizin p değerleri 0 olduğu için anlamlılar. Katsayılarda nemin katsayısı daha fazla çıktı.
# Nem hava sıcaklığına rüzgar hızından daha çok etki eder. Yani beklentimi karşıladı.
# Adj R-squared değeri 0.288. Bu değer, bu iki özellik ile modelimizi tam olarak anlamlandıramadığımızı söylüyor.

#####################################################################################################
# Ardından, yukarıdaki modele nem ve rüzgar hızı etkileşimini dahil edin ve OLS'yi kullanarak modeli tahmin edin.
# Katsayılar istatistiksel olarak anlamlı mıdır? Nem ve rüzgar hızı için tahmini katsayıların işaretleri değişti mi?
# Tahmin edilen katsayıları yorumlayınız.


# hava_durumu["hedef_degisken"] = hava_durumu["Sicaklik"]-hava_durumu["gorunur_sicaklik"]
#
# hava_durumu["nem_RuzgarHızı_iliskisi"] = hava_durumu["Nem"] * hava_durumu["RuzgarHizi"]
#
# Y = hava_durumu["hedef_degisken"]
# X = hava_durumu[["Nem","RuzgarHizi","nem_RuzgarHızı_iliskisi"]]
#
# lrm = linear_model.LinearRegression()
# lrm.fit(X, Y)
#
# print('Değişkenler: \n', lrm.coef_)
# print('Sabit değer (bias): \n\n\n', lrm.intercept_)
#
# X = sm.add_constant(X)
# results = sm.OLS(Y, X).fit()
# print(results.summary())

# Adj R-squared değerimiz yükseldi. Bu veriyi daha iyi yorumladığımız anlamına geliyor. Değişkenlerimizin
# p değerleri sıfır ama constant'ın p değeri biraz arttı. Nem ve rüzgar hızının katsayıları negatif iken yeni
# yarattığımız nem_ve_RuzgarHızı_iliskisi ilişkisi değişkeni positif. Bu durumda nem ve rüzgar hızının artması,
# sıcaklığa negatif bir katkı mı yapıcak? Emin değilim. Rüzgar hızının artmasını havanın soğumasına bağlayabiliriz.
# Bu da negatif değer almasını açıklar. Ama nem için anlamlandıramadım.



########################################################################################################################
# Ev fiyatları modelinizi tekrar çalıştırın ve sonuçları yorumlayın.
# Hangi özellikler istatistiksel olarak anlamlı ve hangileri değildir?

# house = pd.read_csv("train.csv")
#
# house['yeni_mi'] = np.where(house['YearBuilt']>=2005, 1, 0)
#
# Y = house["SalePrice"]
# X = house[["BedroomAbvGr","yeni_mi","FullBath","KitchenAbvGr","GarageCars","WoodDeckSF","OverallQual","LotArea"]]
#
# lrm = linear_model.LinearRegression()
# print(lrm.fit(X, Y))
#
# print('Değişkenler: \n', lrm.coef_)
# print('Sabit değer (bias): \n', lrm.intercept_)
#
# X = sm.add_constant(X)
# results = sm.OLS(Y, X).fit()
# print(results.summary())

# KitchenAbvGr özelliğimiz negatif katsayılı çıktı. Zaten p değeri de 0.025 . Kabul edilebilir değerler arasında ama
# diğer özelliklerimizin p değeri 0 iken Kitchen'ın olması dikkatleri ona çekiyor. Onun dışında bütün özelliklerimiz
# istatistiksel olarak anlamlıdır.

########################################################################################################################
#Şimdi, anlamsız özellikleri modelinizden hariç tutun. Bir şey değişti mi?

# house = pd.read_csv("train.csv")
#
# house['yeni_mi'] = np.where(house['YearBuilt']>=2005, 1, 0)
#
# Y = house["SalePrice"]
# X = house[["BedroomAbvGr","yeni_mi","FullBath","GarageCars","WoodDeckSF","OverallQual","LotArea"]]
#
# lrm = linear_model.LinearRegression()
# print(lrm.fit(X, Y))
#
# print('Değişkenler: \n', lrm.coef_)
# print('Sabit değer (bias): \n', lrm.intercept_)
#
# X = sm.add_constant(X)
# results = sm.OLS(Y, X).fit()
# print(results.summary())

# Kitchen değerini çıkardıktan sonra Adj-R squared değerimiz 0.715 ten 0.714e düştü. Yani hiç bir şey kaybetmedik.

########################################################################################################################
#Ev fiyatları ile olan ilişkilerini belirleyerek istatistiksel olarak anlamlı katsayıları yorumlayın.
# Ev fiyatları üzerinde hangi özellikler daha belirgin bir etkiye sahiptir?

# house = pd.read_csv("train.csv")
#
# house['yeni_mi'] = np.where(house['YearBuilt']>=2005, 1, 0)
#
# Y = house["SalePrice"]
# X = house[["BedroomAbvGr","yeni_mi","FullBath","GarageCars","WoodDeckSF","OverallQual","LotArea"]]
#
# lrm = linear_model.LinearRegression()
# print(lrm.fit(X, Y))
#
# print('Değişkenler: \n', lrm.coef_)
# print('Sabit değer (bias): \n', lrm.intercept_)
#
# X = sm.add_constant(X)
# results = sm.OLS(Y, X).fit()
# print(results.summary())

# Sırasıyla OverallQuality, GarageCars, FullBath, yeni_mi değerleri ön planda olan özellikler.
# OverallQuality'nin katsayısının çok olması fazlasıyla anlamlı. Constant değerimiz negatif çıktı. Bu durumda
# sadece constant'ı yorumlarken sale_price değerimiz negatif olur? Anlamlandıramadım.


# print results.summary yaptıktan sonra console'un en altında ""The condition number is large, 8.4e+04. This might indicate that there are
# strong multicollinearity or other numerical problems."" Yazıyor. Bunun anlamı verimizdeki küçük hatalar büyük
#sonuçlara yok açabilir demek mi?

