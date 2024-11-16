import cv2
import torch


# Завантаження моделі YOLOv5
def load_model():
    # Завантаження моделі YOLOv5s з локального репозиторію
    model = torch.hub.load('yolov5', 'yolov5s', source='local')  # Використання локальної моделі
    return model


def detect_objects(model, image):
    # Виконання детекції об'єктів
    results = model(image)
    return results


def visualize_results(results, image):
    # Візуалізація результатів
    results.render()  # Додає рамки до зображення
    return image


def main(image_path):
    # Завантаження моделі
    model = load_model()

    # Зчитування зображення
    image = cv2.imread(image_path)
    if image is None:
        print("Не вдалося завантажити зображення.")
        return

    # Детекція об'єктів
    results = detect_objects(model, image)

    # Візуалізація результатів
    image_with_boxes = visualize_results(results, image)

    # Показ зображення
    cv2.imshow("Detected Objects", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def real_time_detection():
    # Завантаження моделі
    model = load_model()

    cap = cv2.VideoCapture(0)  # Використання веб-камери
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не вдалося отримати кадр.")
            break

        # Детекція об'єктів
        results = detect_objects(model, frame)

        # Візуалізація результатів
        frame_with_boxes = visualize_results(results, frame)

        # Показ кадру
        cv2.imshow("Detected Objects", frame_with_boxes)

        # Вихід з циклу при натисканні 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Вказати шлях до вашого зображення або запустити детекцію в реальному часі
    image = "/Users/erihkoh/GitHub/object_detect_yolo5/images/group-of-asia-animals-photo.jpg"
    # main(image)
    real_time_detection()
