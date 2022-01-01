# Used Car Price Prediction
## İkinci El Araba Fiyat Tahmini

### Artificial Intelligence
Bu projede, daha önceden satılmış olan ikinci el araçların özniteliklerinin ve satış fiyatlarının yer aldığı bir [veri seti](https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes?select=bmw.csv) kullanıldı.

Veri seti üzerinde gerekli [ön işleme adımları](./artificial_intelligence/dataset.txt) uygulandıktan sonra aşağıdaki algoritmaların eğitiminde kullanıldı:
- Decision Tree Regressor
- Gradient Boosting Regressor
- XGB Regressor
- Random Forest Regressor
- LGBM (LightGBM) Regressor
- Cat Boost Regressor

>Bu projenin bilimsel makalesi de [LaTeX](./LaTeX/) konumundaki [PDF](./LaTeX/Used_Car_Price_Prediction.pdf) ve [TeX](./LaTeX/Used_Car_Price_Prediction.tex) dosyalarında yer almakta.

### Pattern Recognition
Bu projede, daha önceden satılmış olan ikinci el araçların özniteliklerinin ve satış fiyatlarının yer aldığı sentetik bir veri seti kullanıldı.

Farklı algoritmalara sahip farklı kod bloklarıyla denemeler yapılarak veri setleri test edildi.

> Sentetik verilerin bir kısmı el ile oluşturulurken bir kısmı da belli bir kurala göre rastgele bir şekilde oluşturuldu. Verileri rastgele bir şekilde oluşturan kod bloğu da [burada](./pattern_recognition/random_data_generator.py) yer almakta.