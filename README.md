# Real-time Object Recognition with KNN and HOG Features

Bu proje OpenCV ve scikit-learn KNN kullanarak gerçek zamanlı nesne tanıma yapar.  
Fare ile ROI seçip nesne örnekleri toplayabilir ve KNN modeli ile sınıflandırma yapılabilir.

## Özellikler

- Canlı video akışı
- Fare ile nesne seçimi (ROI)
- HOG özellik çıkarımı
- KNN sınıflandırıcı
- Eğitim ve tahmin işlemleri
- Eğitim verisi dosyasında veri saklama ve yükleme

## Gereksinimler

- Python 3.x
- opencv-python
- numpy
- scikit-learn
- pickle (Python standard)

## Kurulum

'''bash
pip install opencv-python numpy scikit-learn
## Kullanım
Fare ile nesnenin ROI’sini seçin.

's' tuşuna basarak seçilen nesnenin HOG özelliklerini çıkarıp etiketleyin.

En az 3 farklı örnek ekledikten sonra 'r' tuşuna basarak gerçek zamanlı tanıma yapabilirsiniz.

'q' tuşu ile çıkabilirsiniz.
