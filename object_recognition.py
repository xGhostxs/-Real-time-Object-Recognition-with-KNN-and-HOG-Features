import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

drawing = False
start_point = None
end_point = None
roi_defined = False

features = []
labels = []

knn = KNeighborsClassifier(n_neighbors=3)
model_trained = False

def extract_features(image):
    win_size = (64, 128)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    image = cv2.resize(image, win_size)
    return hog.compute(image).flatten()

def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point, end_point, roi_defined
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = None
        roi_defined = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        roi_defined = True

try:
    with open('features_labels.pkl', 'rb') as f:
        features, labels = pickle.load(f)
        if len(features) >= 3:
            knn.fit(features, labels)
            model_trained = True
    print("Önceki eğitim verileri yüklendi ve model eğitildi.")
except FileNotFoundError:
    print("Eğitim verisi dosyası bulunamadı. Yeni veriler oluşturulacak.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_rectangle)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if start_point and end_point:
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

    if roi_defined and start_point and end_point:
        x1, y1 = start_point
        x2, y2 = end_point
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        selected_region = frame[y1:y2, x1:x2]
        if selected_region.size > 0:
            cv2.imshow("Selected Region", selected_region)

            if selected_region.shape[0] >= 128 and selected_region.shape[1] >= 64:
                gray = cv2.cvtColor(selected_region, cv2.COLOR_BGR2GRAY)
                features_vector = extract_features(gray)

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    label = input("Bu nesnenin adı nedir?: ")
                    features.append(features_vector)
                    labels.append(label)
                    knn.fit(features, labels)
                    model_trained = True
                    print(f"{label} kaydedildi!")

                elif cv2.waitKey(1) & 0xFF == ord('r'):
                    if model_trained:
                        prediction = knn.predict([features_vector])[0]
                        print(f"Tanımlanan nesne: {prediction}")
                    else:
                        print("Model henüz eğitilmedi. En az 3 örnek ekleyin ve 's' ile kaydedin.")

            else:
                print("Seçilen bölge çok küçük! Daha büyük bir bölge seçin.")

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(features) >= 3:
    with open('features_labels.pkl', 'wb') as f:
        pickle.dump((features, labels), f)
    print("Eğitim verileri kaydedildi.")
else:
    print("Yeterli eğitim verisi yok, kaydedilmedi.")
